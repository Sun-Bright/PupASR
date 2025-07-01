# ======== 分块预处理版：断点恢复 + 避免 OOM 的安全方案（支持加速） ========
# 文件名建议保存为 preprocess_magicdata.py

import os
import warnings
warnings.filterwarnings("ignore")

import torch
import torchaudio
import pynvml
import psutil
import re
import multiprocessing
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from datetime import datetime
from transformers import Wav2Vec2Processor

# 设置多进程启动模式为 fork，加速子进程（Linux 专用）
multiprocessing.set_start_method('fork', force=True)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["DATASETS_DISABLE_MULTIPROCESSING"] = "1"

# ========== 路径配置 ==========
TRAIN_INDEX = "./data/MDT2021S003/prepared_bigdata/train.txt"
DEV_INDEX = "./data/MDT2021S003/prepared_bigdata/dev.txt"
MODEL_LOCAL_PATH = "/mnt/data/wav2vec2-wbbb-offline"
OUTPUT_DIR = "./preprocessed_magicdata_wbbb"
CHUNK_DIR = os.path.join(OUTPUT_DIR, "train_chunks")
os.makedirs(CHUNK_DIR, exist_ok=True)

processor = Wav2Vec2Processor.from_pretrained(MODEL_LOCAL_PATH, local_files_only=True)

# ========== 文本清洗 ==========
def clean_text(text):
    return re.sub(r"[^\u4e00-\u9fff]+", "", text)

def load_custom_data(txt_path):
    audio_paths, texts = [], []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() == "":
                continue
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            audio_paths.append(parts[0])
            texts.append(clean_text(parts[1]))
    return Dataset.from_dict({"path": audio_paths, "text": texts})

# ========== 监控函数 ==========
def print_memory_usage(example_index):
    if example_index % 1000 == 0:
        print(f"[监控] 当前处理到第 {example_index} 条样本")
        print(f"[内存] 使用: {psutil.virtual_memory().used / 1024**3:.2f} GB / {psutil.virtual_memory().total / 1024**3:.2f} GB")
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            print(f"[显存] 使用: {mem_info.used / 1024**2:.2f} MiB / {mem_info.total / 1024**2:.2f} MiB")
        except:
            print("[警告] 无法读取显卡信息")

# ========== 预处理函数 ==========
def preprocess(example, idx=None):
    try:
        if idx is not None:
            print_memory_usage(idx)
        audio_tensor, sr = torchaudio.load(example["path"])
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            audio_tensor = resampler(audio_tensor)
        if audio_tensor.shape[1] < 1600 or audio_tensor.shape[1] / 16000 > 15.0:
            raise ValueError("音频过短或过长，跳过")
        audio = audio_tensor.squeeze().numpy()
        inputs = processor(audio, sampling_rate=16000, return_attention_mask=True)
        inputs["labels"] = processor.tokenizer(example["text"]).input_ids
        return inputs
    except Exception as e:
        with open("preprocess_error.log", "a", encoding="utf-8") as logf:
            logf.write(f"[{datetime.now()}] 跳过: {example['path']} - 错误: {str(e)}\n")
        return {"input_values": None, "attention_mask": None, "labels": None}

# ========== 加载原始数据 ==========
print("[加载] 原始文本索引中...")
dataset = DatasetDict({
    "train": load_custom_data(TRAIN_INDEX),
    "validation": load_custom_data(DEV_INDEX),
})

# ========== 分块处理训练集（断点恢复） ==========
chunk_size = 10000
for start in range(0, len(dataset["train"]), chunk_size):
    end = min(start + chunk_size, len(dataset["train"]))
    chunk_path = os.path.join(CHUNK_DIR, f"chunk_{start}_{end}")
    if os.path.exists(chunk_path):
        print(f"[跳过] 已存在: {chunk_path}")
        continue

    print(f"[分块] 处理训练样本 {start} 到 {end}")
    chunk = dataset["train"].select(range(start, end))
    processed_chunk = chunk.map(
        preprocess,
        with_indices=True,
        remove_columns=["path", "text"],
        load_from_cache_file=False,
        desc=f"训练集块 {start}-{end}",
        num_proc=4,  # ✅ 开启 4 进程加速
    )
    processed_chunk = processed_chunk.filter(
        lambda x: x["input_values"] is not None and x["labels"] is not None,
        num_proc=4  # ✅ 加速过滤
    )
    processed_chunk.save_to_disk(chunk_path)

# ========== 合并所有块 ==========
print("[合并] 读取所有训练分块...")
from glob import glob
all_chunks = sorted(glob(os.path.join(CHUNK_DIR, "chunk_*")))
processed_train = concatenate_datasets([load_from_disk(p) for p in all_chunks])

# ========== 验证集正常处理 ==========
print("[处理] 验证集样本...")
dataset["validation"] = dataset["validation"].map(
    preprocess,
    with_indices=True,
    remove_columns=["path", "text"],
    load_from_cache_file=False,
    desc="验证集",
    num_proc=4,
)
dataset["validation"] = dataset["validation"].filter(lambda x: x["input_values"] is not None and x["labels"] is not None, num_proc=4)

# ========== 合并保存 ==========
dataset["train"] = processed_train
print("[保存] 开始写入磁盘...")
dataset.save_to_disk(OUTPUT_DIR)
print(f"[完成] 数据已保存到: {OUTPUT_DIR}")
