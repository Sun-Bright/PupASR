# server.py - 部署在华为云的服务端
# 功能: 接收客户端音频流并进行识别

import os
import numpy as np
import time
import threading
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel
import socket
import asyncio
import base64
import io

# 导入语音识别模型
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2ProcessorWithLM
)
import torch
from pyctcdecode import build_ctcdecoder
from pathlib import Path

# === 热词后处理模块 ===
HOTWORD_DICT = {
    "小智": ["小志", "小治", "小治理", "小志你", "小致", "乔治"],
    #"停下": ["天下", "听下", "亭下", "听下"],
    #"站立": ["战力", "占理"],
    #"打招呼": ["打招虎", "打照顾", "打照护"],
    #"坐下": ["做下", "作下"],
    #"前进": ["前见","全景","影响"],
    #"后退": ["猴腿", "后腿"],
    #"打开灯": ["打开凳", "打开等"]
}

def correct_hotword(text: str) -> str:
    for correct, wrong_list in HOTWORD_DICT.items():
        for wrong in wrong_list:
            if wrong in text:
                print(f"[热词纠错] {wrong} → {correct}")
                text = text.replace(wrong, correct)
    return text

# === 全局配置 ===
MODEL_PATH = "/root/projects/wav2vec2-live/my_model/final_stage2_output/checkpoint-463"
LM_PATH = "./lm/lm.binary"
UNIGRAMS_PATH = "./unigrams_wbbb.txt"

HOTWORDS = {
    "小智": 20.0,
    "前进": 12.0,
    "后退": 12.0,
    "往后退": 12.0,
    "站立": 12.0,
    "站起来": 12.0,
    "坐下": 12.0,
    "睡觉": 12.0,
    "休息": 12.0,
    "左转": 12.0,
    "向左转": 12.0,
    "右转": 12.0,
    "向右转": 12.0,
    "挥手": 12.0,
    "打招呼": 12.0,
    "停止": 12.0,
    "停下": 12.0,
}

HOTWORD_WEIGHT = 50.0
LM_ALPHA = 0.8
LM_BETA = 1.2
BEAM_WIDTH = 80
SAMPLE_RATE = 16000
MIN_AUDIO_S = 0.5
MIN_CONFIDENCE = -5.0

def validate_hotwords(hotwords: dict, unigrams_path: str) -> dict:
    """
    检查热词中所有字符是否都在字符表中，剔除非法热词。
    """
    try:
        with open(unigrams_path, "r", encoding="utf-8") as f:
            vocab_set = set(line.strip() for line in f if line.strip())
    except Exception as e:
        print(f"[错误] 加载字符表失败: {str(e)}")
        return {}

    valid = {}
    invalid = []
    for word, score in hotwords.items():
        if all(c in vocab_set for c in word):
            valid[word] = score
        else:
            invalid.append(word)

    print(f"[热词] 有效热词（{len(valid)} 个）: {list(valid.keys())}")
    if invalid:
        print(f"[热词] ⚠️ 无效热词（{len(invalid)} 个，被忽略）: {invalid}")
    return valid

def expand_hotwords(hotwords: dict) -> list:
    expanded = []
    for word, score in hotwords.items():
        expanded.append((tuple(word), score))
    return expanded

class Wave2Vec2Inference:
    def __init__(self, model_name, lm_path=None, unigrams_path=None,
                 lm_alpha=1.0, lm_beta=0.0, beam_width=20, hotwords=None):
        print(f"[初始化] 加载模型: {model_name}")
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.tokenizer = self.processor.tokenizer
        self.decoder = None
        self.using_lm = False
        self.hotwords = expand_hotwords(hotwords if hotwords else {})
        print(f"[设备] 模型当前设备: {next(self.model.parameters()).device}")

        if lm_path:
            self._initialize_language_model(lm_path, unigrams_path, lm_alpha, lm_beta, beam_width)

    def _initialize_language_model(self, lm_path, unigrams_path, lm_alpha, lm_beta, beam_width):
        lm_file = Path(lm_path)
        if not lm_file.is_file():
            print(f"[错误] 语言模型文件不存在: {lm_path}")
            return

        unigrams = self._load_unigrams(unigrams_path)
        vocab_dict = self.tokenizer.get_vocab()
        sorted_labels = [k for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])]

        valid_hotwords = []
        for word_tuple, score in self.hotwords:
            if all(char in sorted_labels for char in word_tuple):
                valid_hotwords.append((word_tuple, score))

        print("[热词] 有效热词({}个): {}".format(len(valid_hotwords), ["".join(k) for k, _ in valid_hotwords]))

        try:
            decoder_args = {
                'labels': sorted_labels,
                'kenlm_model_path': str(lm_file),
                'unigrams': unigrams,
                'alpha': lm_alpha,
                'beta': lm_beta,
                #'hotwords': valid_hotwords,
                #'hotword_weight': HOTWORD_WEIGHT
            }
            self.decoder = build_ctcdecoder(**decoder_args)
            self.processor = Wav2Vec2ProcessorWithLM(
                feature_extractor=self.processor.feature_extractor,
                tokenizer=self.processor.tokenizer,
                decoder=self.decoder
            )
            self.beam_width = beam_width
            self.using_lm = True
            print("[成功] 语言模型加载完成（含热词配置）")
        except Exception as e:
            print(f"[错误] 解码器初始化失败: {str(e)}")
            self.using_lm = False

    def _load_unigrams(self, unigrams_path):
        unigrams_file = Path(unigrams_path)
        if not unigrams_file.is_file():
            print(f"[警告] 字符表文件不存在: {unigrams_path}")
            return None

        try:
            with open(unigrams_file, 'r', encoding='utf-8') as f:
                unigrams = [line.strip() for line in f if line.strip()]
            print(f"[加载] 已加载字符表 {len(unigrams)} 个字符")
            return unigrams
        except Exception as e:
            print(f"[错误] 读取字符表失败: {str(e)}")
            return None

    def buffer_to_text(self, buffer: np.ndarray):
        if len(buffer) == 0:
            return "", None

        float_buffer = buffer.astype(np.float32)

        # 预处理
        inputs = self.processor(
            float_buffer,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}  # ← 新增：推理数据转移到GPU
        with torch.no_grad():
            logits = self.model(inputs.input_values).logits  # shape: [1, T, V]

        text = ""
        confidence = None

        if self.using_lm:
            try:
                logits_np = logits[0].cpu().numpy()  # shape: [T, V]
                result = self.processor.decode(logits_np, beam_width=self.beam_width)
                # OutputBeam 类型：使用 .text 和 .logit_score
                text = result.text
                confidence = result.logit_score
            except Exception as e:
                print(f"[警告] LM解码失败: {str(e)}, 使用贪婪解码")
                predicted_ids = torch.argmax(logits, dim=-1)
                text = self.processor.decode(predicted_ids[0])
        else:
            predicted_ids = torch.argmax(logits, dim=-1)
            text = self.processor.decode(predicted_ids[0])

        text = correct_hotword(text)
        return text, confidence


if __name__ == "__main__":
    print(" 测试模型和热词初始化中...")
    inference = Wave2Vec2Inference(
        model_name=MODEL_PATH,
        lm_path=LM_PATH,
        unigrams_path=UNIGRAMS_PATH,
        lm_alpha=LM_ALPHA,
        lm_beta=LM_BETA,
        beam_width=BEAM_WIDTH,
        hotwords=HOTWORDS
    )

# FastAPI 应用
app = FastAPI(title="远程语音识别API")

# 配置 CORS，允许任意来源访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ASR引擎单例
asr_engine = None
asr_lock = threading.Lock()


# Pydantic模型
class ASRResponse(BaseModel):
    text: str
    duration: float
    inference_time: float
    confidence: float = None


# 连接管理
class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"[WebSocket] 新客户端连接，当前连接数: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"[WebSocket] 客户端断开，剩余连接数: {len(self.active_connections)}")


manager = ConnectionManager()


# 初始化ASR引擎
def initialize_asr_engine():
    global asr_engine
    with asr_lock:
        if asr_engine is None:
            try:
                asr_engine = Wave2Vec2Inference(
                    model_name=MODEL_PATH,
                    lm_path=LM_PATH,
                    unigrams_path=UNIGRAMS_PATH,
                    lm_alpha=LM_ALPHA,
                    lm_beta=LM_BETA,
                    beam_width=BEAM_WIDTH,
                    hotwords=HOTWORDS
                )
                print("[服务器] ASR引擎初始化成功")
            except Exception as e:
                print(f"[错误] ASR引擎初始化失败: {str(e)}")
                raise RuntimeError(f"ASR引擎初始化失败: {str(e)}")
    return asr_engine


# 启动事件
@app.on_event("startup")
async def startup_event():
    initialize_asr_engine()
    local_ip = get_local_ip()
    print(f"[服务器] 已启动 - HTTP接口：http://{local_ip}:8000/recognize")
    print(f"[服务器] 已启动 - WebSocket接口：ws://{local_ip}:8000/ws/asr")


# 获取本机IP
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


# HTTP接口：处理上传的音频文件
@app.post("/recognize", response_model=ASRResponse)
async def recognize_audio(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="音频文件未提供")

    # 读取文件内容
    audio_bytes = await file.read()

    # 转换为NumPy数组
    try:
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"音频解析失败: {str(e)}")

    # 进行识别
    start_time = time.perf_counter()
    try:
        # 确保ASR引擎已初始化
        if asr_engine is None:
            initialize_asr_engine()

        text, confidence = asr_engine.buffer_to_text(audio_np)
        inference_time = time.perf_counter() - start_time
        duration = len(audio_np) / SAMPLE_RATE

        return ASRResponse(
            text=text,
            duration=duration,
            inference_time=inference_time,
            confidence=confidence if confidence is not None else 0.0
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"识别处理失败: {str(e)}")


# 接收Base64编码的音频数据
@app.post("/recognize_base64", response_model=ASRResponse)
async def recognize_audio_base64(data: dict):
    if 'audio' not in data:
        raise HTTPException(status_code=400, detail="未提供音频数据")

    try:
        # 解码Base64
        audio_bytes = base64.b64decode(data['audio'])
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16)

        # 确保ASR引擎已初始化
        if asr_engine is None:
            initialize_asr_engine()

        # 进行识别
        start_time = time.perf_counter()
        text, confidence = asr_engine.buffer_to_text(audio_np)
        inference_time = time.perf_counter() - start_time
        duration = len(audio_np) / SAMPLE_RATE

        return ASRResponse(
            text=text,
            duration=duration,
            inference_time=inference_time,
            confidence=confidence if confidence is not None else 0.0
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"识别处理失败: {str(e)}")


# WebSocket接口：处理实时音频流
@app.websocket("/ws/asr")
async def websocket_asr(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        print("[WebSocket] 等待音频数据...")

        # 确保ASR引擎已初始化
        if asr_engine is None:
            initialize_asr_engine()

        while True:
            # 接收客户端发送的音频数据
            data = await websocket.receive_bytes()

            # 处理接收到的音频数据
            try:
                audio_np = np.frombuffer(data, dtype=np.int16)
                duration = len(audio_np) / SAMPLE_RATE

                # 丢弃过短的音频片段
                if duration < MIN_AUDIO_S:
                    continue

                # 执行识别
                start_time = time.perf_counter()
                text, confidence = asr_engine.buffer_to_text(audio_np)
                inference_time = time.perf_counter() - start_time

                # 只发送有内容的结果
                if text.strip():
                    await websocket.send_json({
                        "text": text,
                        "duration": duration,
                        "inference_time": inference_time,
                        "confidence": confidence if confidence is not None else 0.0
                    })
                    print(f"[识别结果] {text}")
            except Exception as e:
                print(f"[WebSocket] 处理异常: {str(e)}")
                await websocket.send_json({"error": str(e)})

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"[WebSocket] 连接异常: {str(e)}")
        manager.disconnect(websocket)


# 健康检查接口
@app.get("/health")
async def health_check():
    return {"status": "ok", "asr_engine": asr_engine is not None}


# 主程序
if __name__ == "__main__":
    uvicorn.run(
        "wbbb_gpu_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
