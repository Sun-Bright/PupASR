import asyncio
import websockets
import json
import numpy as np
import wave
import io
import os
import socket
import threading
from datetime import datetime
import opuslib  # 使用opuslib替代opus
import time
import aiohttp  # 添加 aiohttp 用于异步 HTTP 请求
import base64  # 添加 base64 用于 Base64 编码
from pypinyin import lazy_pinyin  # 添加拼音转换功能
import pyaudio
import webrtcvad

# === 全局配置 ===
SAMPLE_RATE = 16000  # 采样率
CHANNELS = 1  # 音频通道数
CHUNK_DURATION_MS = 30  # 音频块时长（毫秒）
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)  # 块大小计算
AUDIO_FORMAT = pyaudio.paInt16  # 音频格式

OPUS_FRAME_MS = 60  # 60ms帧长度
OPUS_SAMPLE_RATE = 16000  # 16kHz采样率
OPUS_CHANNELS = 1
OPUS_FRAME_SAMPLES = int(OPUS_SAMPLE_RATE * OPUS_FRAME_MS / 1000)
OPUS_ENCODER_BITRATE = 24000

# 自己的语音识别服务器配置
ASR_SERVER_URL = "ws://1.94.205.149:8000/ws/asr"


class VoiceService:
    def __init__(self, host='0.0.0.0', port=8081, udp_port=8082):
        self.host = host
        self.port = port
        self.udp_port = udp_port

        self.running = False
        self.clients = {}
        self.last_speech_time = 0

        # 初始化Opus编码器和解码器
        self.opus_encoder = opuslib.Encoder(OPUS_SAMPLE_RATE, OPUS_CHANNELS, opuslib.APPLICATION_VOIP)
        self.opus_encoder.bitrate = OPUS_ENCODER_BITRATE
        self.opus_encoder.frame_size = OPUS_FRAME_SAMPLES
        self.opus_decoder = opuslib.Decoder(OPUS_SAMPLE_RATE, OPUS_CHANNELS)
        self.opus_decoder.frame_size = OPUS_FRAME_SAMPLES

        # 语音识别相关 - 简化版本
        self.audio_buffer = []  # 音频缓冲区
        self.buffer_lock = asyncio.Lock()
        self.last_send_time = 0
        self.min_audio_duration = 0.5  # 最小音频时长（秒），低于此时长不发送

        # 初始化VAD（Voice Activity Detection）
        self.vad = webrtcvad.Vad(1)  # 敏感度设置为1（0-3，0最不敏感，3最敏感）
        self.vad_frame_duration = 30  # VAD帧时长30ms
        self.vad_frame_samples = int(OPUS_SAMPLE_RATE * self.vad_frame_duration / 1000)  # 480 samples for 30ms at 16kHz
        self.vad_frame_size = self.vad_frame_samples * 2  # 16-bit samples = 2 bytes per sample

        # 动作状态管理
        self.current_action = None
        self.is_noisy_action_running = False
        self.noisy_actions = ["Walk", "Walk back", "turn left", "turn right"]

    async def _send_audio_to_asr_simple(self, audio_data, client_id):
        """简单直接的ASR识别 - 单次请求响应模式"""
        # 检查音频时长
        duration = len(audio_data) / (SAMPLE_RATE * 2)  # 16位=2字节
        if duration < self.min_audio_duration:
            return False

        max_retries = 3
        timeout = 5.0

        for attempt in range(max_retries):
            try:

                # 创建临时WebSocket连接
                async with websockets.connect(ASR_SERVER_URL) as websocket:
                    # 发送音频数据
                    await websocket.send(audio_data)

                    # 等待识别结果
                    response = await asyncio.wait_for(websocket.recv(), timeout=timeout)

                    # 解析结果
                    if isinstance(response, str):
                        try:
                            result = json.loads(response)
                            text = result.get('text', '').strip()
                            if text:
                                print(f"识别结果: {text}")
                                # 直接处理识别结果
                                await self._handle_asr_result(client_id, text)
                                return True
                        except json.JSONDecodeError:
                            print(f"解析JSON失败: {response}")

                return False  # 没有有效结果

            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5)  # 重试前等待
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5)  # 重试前等待
        return False

    def _audio_quality_check(self, audio_data):
        """音频质量预检查，过滤明显的噪音"""
        try:
            # 将bytes转换为numpy数组进行分析
            import numpy as np

            # 将音频数据转换为16位有符号整数数组
            audio_array = np.frombuffer(audio_data, dtype=np.int16)

            # 计算RMS (均方根) 音量
            rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))

            # 设置最低音量阈值 (可根据实际情况调整)
            min_rms_threshold = 100  # 最低有效音量 (降低阈值)
            max_rms_threshold = 15000  # 最高音量(避免过载/饱和)

            # 检查音量是否在合理范围内
            if rms < min_rms_threshold:
                print(f"❌ 音量过低: RMS={rms:.1f} < {min_rms_threshold}")
                return False

            if rms > max_rms_threshold:
                print(f"❌ 音量过载: RMS={rms:.1f} > {max_rms_threshold}")
                return False

            # 检查零交叉率 (Zero Crossing Rate) - 区分语音和纯噪音
            zero_crossings = np.sum(np.diff(np.sign(audio_array)) != 0)
            zcr = zero_crossings / len(audio_array)

            # 语音通常有适中的零交叉率
            if zcr < 0.01 or zcr > 0.3:  # 太低可能是直流偏移，太高可能是噪音

                return False

            return True

        except Exception as e:
            print(f"音频质量检查出错: {e}")
            return True  # 出错时默认通过

    async def _detect_speech(self, audio_data, client_id="default"):
        """使用VAD检测语音活动并累积音频"""
        try:
            # 检查音频数据长度
            if len(audio_data) < 960:  # 小于30ms@16kHz的音频
                return False

            # 先进行音频质量预检查，过滤明显的噪音
            if not self._audio_quality_check(audio_data):
                return False

            # 使用VAD检测语音活动
            has_voice = self._vad_detection(audio_data)

            async with self.buffer_lock:
                if has_voice:
                    # 添加音频到缓冲区，不立即发送，等待静音
                    self.audio_buffer.append(audio_data)

                else:
                    # 无语音，检查是否需要发送缓冲区的音频
                    if self.audio_buffer:
                        # 计算缓冲区音频时长
                        total_samples = sum(len(chunk) for chunk in self.audio_buffer) // 2
                        current_duration = total_samples / SAMPLE_RATE
                        current_time = time.time()

                        # 如果音频时长足够且超过静音间隔，则发送
                        if (current_duration >= self.min_audio_duration and
                                current_time - self.last_send_time > 0.5):

                            combined_audio = b''.join(self.audio_buffer)
                            print(f"🎤 发送完整语音: {current_duration:.2f}秒")

                            asyncio.create_task(
                                self._send_audio_to_asr_simple(combined_audio, client_id)
                            )

                            self.audio_buffer.clear()
                            self.last_send_time = current_time
                        elif (current_time - self.last_send_time > 3.0):
                            # 超过3秒未发送，清空缓冲区（可能是噪音）
                            if current_duration < self.min_audio_duration:
                                self.audio_buffer.clear()
                                self.last_send_time = current_time

            return has_voice

        except Exception as e:
            print(f"语音检测出错: {e}")
            return False

    def _vad_detection(self, audio_data):
        """使用WebRTC VAD进行语音活动检测"""
        try:
            # 确保音频数据长度符合VAD要求
            if len(audio_data) < self.vad_frame_size:
                # 如果数据不够，用零填充
                audio_data = audio_data + b'\x00' * (self.vad_frame_size - len(audio_data))

            # VAD检测需要固定长度的帧，我们按30ms帧进行检测
            voice_frames = 0
            total_frames = 0

            for i in range(0, len(audio_data) - self.vad_frame_size + 1, self.vad_frame_size):
                frame = audio_data[i:i + self.vad_frame_size]

                try:
                    # VAD检测，参数：音频帧数据，采样率
                    is_speech = self.vad.is_speech(frame, OPUS_SAMPLE_RATE)
                    if is_speech:
                        voice_frames += 1
                    total_frames += 1
                except Exception as e:
                    print(f"VAD检测单帧出错: {e}")
                    continue

            if total_frames == 0:
                return False

            # 如果超过10%的帧包含语音，认为这段音频包含语音 (降低阈值增加检测)
            voice_ratio = voice_frames / total_frames
            has_voice = voice_ratio > 0.10

            if has_voice:
                pass

            return has_voice
        except Exception as e:
            print(f"VAD检测出错: {e}")
            return False

    async def _handle_asr_result(self, client_id, text):
        """处理完整的语音识别结果"""
        try:
            # 找到对应的WebSocket客户端
            target_websocket = None
            target_client_info = None

            # 遍历clients字典：{client_addr: {"websocket": websocket, "client_info": client_info}}
            for client_addr, client_data in self.clients.items():
                # 正确比较元组：client_id 已经是元组 (ip, port)
                if client_addr == client_id:
                    target_websocket = client_data["websocket"]
                    target_client_info = client_data["client_info"]
                    break

            if target_websocket and target_client_info:
                print(f"🎤 识别结果: {text}")
                # 直接调用原有的语音段处理逻辑，但传入识别到的文本而不是PCM数据
                await self._process_speech_with_text(target_websocket, text, target_client_info)
            else:
                print(f"客户端连接丢失: {client_id}")
        except Exception as e:
            print(f"处理语音识别结果时出错: {e}")
            import traceback
            print(f"错误堆栈: {traceback.format_exc()}")

    async def _process_speech_with_text(self, websocket, text, client_info):
        """基于语音识别结果处理语音（跳过本地STT步骤）"""
        try:
            # 立即设置为processing状态，阻止新的语音输入
            client_info["state"] = "processing"

            # 将文本转换为拼音列表
            text_pinyin = lazy_pinyin(text)
            text_pinyin_str = ''.join(text_pinyin)

            # 优先检查停止命令（支持更多停止词汇）
            stop_keywords = ["停止", "停下", "停", "别动", "不要动", "暂停"]
            if any(keyword in text for keyword in stop_keywords):
                print("🛑 停止命令")
                stop_commands = [{"name": "Action", "method": "stop", "parameters": {}}]
                await self._send_iot_commands(websocket, stop_commands)
                # 重置动作状态
                self.current_action = None
                self.is_noisy_action_running = False
                response_text = "好的，我停下了"

                # 发送 STT 结果给客户端
                stt_response = {"type": "stt", "text": text}
                await websocket.send(json.dumps(stt_response))

                # 语音合成并发送TTS
                await self._send_tts_response(websocket, response_text, client_info)

                # 恢复到listening状态
                client_info["state"] = "listening"
                return

            # 检查是否包含唤醒词"小智"（包括拼音匹配）
            if "小智" not in text and "xiaozhi" not in text_pinyin_str:
                client_info["state"] = "listening"
                return

            # 发送 STT 结果给客户端
            stt_response = {"type": "stt", "text": text}
            await websocket.send(json.dumps(stt_response))

            # 首先检查是否是纯动作指令（优先处理，不经过大模型）
            action_commands = await self._parse_action_commands(text)
            if action_commands:
                print(f"检测到动作指令，直接执行: {action_commands}")
                await self._send_iot_commands(websocket, action_commands)
                # 记录当前动作（用于噪声过滤）
                for cmd in action_commands:
                    if cmd["name"] == "Action":
                        self.current_action = cmd["method"]
                        self.is_noisy_action_running = cmd["method"] in self.noisy_actions

                        if cmd["method"] in self.noisy_actions:
                            # 不同动作设置不同的持续时间
                            if cmd["method"] in ["Walk", "Walk back"]:
                                duration = 5.0  # 前进后退5秒
                            elif cmd["method"] in ["turn left", "turn right"]:
                                duration = 5.0  # 转向5秒
                            else:
                                duration = 5.0  # 其他动作5秒

                            # 启动自动停止任务
                            asyncio.create_task(self._schedule_auto_stop(websocket, cmd["method"], duration))
                            print(f"动作 {cmd['method']} 将在 {duration} 秒后自动停止")
                        break
                # 动作指令直接返回，不进行TTS
                client_info["state"] = "listening"
                print("动作命令处理完成，恢复语音输入接收")
                return

            # 检查是否是灯光控制指令（也直接执行，不经过大模型）
            lamp_commands = await self._parse_lamp_commands(text)
            if lamp_commands:
                print(f"检测到灯光控制指令，直接执行: {lamp_commands}")
                await self._send_iot_commands(websocket, lamp_commands)
                client_info["state"] = "listening"
                print("灯光命令处理完成，恢复语音输入接收")
                return

            # 如果不是直接控制指令，则进行普通对话处理
            print("进行普通对话处理")
            # 调用大模型获取回复
            response_text = await self._process_with_llm(text)
            if not response_text:
                print("大模型返回空结果")
                client_info["state"] = "listening"
                print("大模型处理异常，恢复语音输入接收")
                return

            # 检查大模型回复中是否包含其他IoT控制建议（如音量调节）
            other_commands = await self._parse_other_commands(text)
            if other_commands:
                await self._send_iot_commands(websocket, other_commands)

            # 进行语音合成并发送
            await self._send_tts_response(websocket, response_text, client_info)

            # 恢复到listening状态
            client_info["state"] = "listening"
            print("对话处理完成，恢复语音输入接收")

        except Exception as e:
            print(f"处理语音识别结果时出错: {e}")
            import traceback
            print(f"错误堆栈: {traceback.format_exc()}")
            client_info["state"] = "listening"
            print("处理异常，恢复语音输入接收")

    async def _send_tts_response(self, websocket, response_text, client_info):
        """发送TTS响应的通用方法"""
        try:
            # 语音合成
            wav_data = await self._text_to_speech(response_text)
            if not wav_data:
                print("语音合成失败，无法发送音频")
                return

            # 设置状态为speaking
            client_info["state"] = "speaking"

            # 发送 TTS 开始信号
            tts_start_response = {"type": "tts", "state": "start"}
            await websocket.send(json.dumps(tts_start_response))

            # 发送 TTS 句子开始信号
            tts_sentence_start_response = {"type": "tts", "state": "sentence_start", "text": response_text}
            await websocket.send(json.dumps(tts_sentence_start_response))

            # 发送音频数据
            with io.BytesIO(wav_data) as wav_io:
                with wave.open(wav_io, 'rb') as wav_file:
                    # 检查音频格式
                    if wav_file.getframerate() != OPUS_SAMPLE_RATE:
                        print(f"警告: 采样率不匹配 ({wav_file.getframerate()}Hz != {OPUS_SAMPLE_RATE}Hz)")
                    if wav_file.getnchannels() != OPUS_CHANNELS:
                        print(f"警告: 声道数不匹配 ({wav_file.getnchannels()} != {OPUS_CHANNELS})")
                    if wav_file.getsampwidth() != 2:
                        print(f"警告: 采样宽度不匹配 ({wav_file.getsampwidth()} != 2)")

                    pcm_chunk_size = OPUS_FRAME_SAMPLES * OPUS_CHANNELS * 2
                    total_frames = wav_file.getnframes()
                    sent_chunks = 0
                    total_bytes_sent = 0

                    print(
                        f"开始发送TTS音频数据，总帧数: {total_frames}, 预期时长: {total_frames / OPUS_SAMPLE_RATE:.2f}秒")

                    while True:
                        pcm_chunk = wav_file.readframes(OPUS_FRAME_SAMPLES)
                        if not pcm_chunk:
                            break

                        original_chunk_len = len(pcm_chunk)

                        # 对于最后一个不完整的块，用零填充到完整帧大小
                        if len(pcm_chunk) < pcm_chunk_size:
                            padding_needed = pcm_chunk_size - len(pcm_chunk)
                            pcm_chunk += b'\x00' * padding_needed
                            print(f"TTS最后一帧填充了 {padding_needed} 字节 (原始长度: {original_chunk_len})")

                        try:
                            # 检查是否收到中断信号
                            if client_info.get("abort_tts", False):
                                print("收到TTS中断信号，停止发送音频")
                                client_info["abort_tts"] = False  # 重置中断标志
                                break

                            opus_chunk = self.opus_encoder.encode(pcm_chunk, self.opus_encoder.frame_size)

                            # 检查WebSocket连接状态
                            if websocket.closed:
                                print("WebSocket连接已关闭，停止发送TTS音频")
                                break

                            await websocket.send(opus_chunk)
                            sent_chunks += 1
                            total_bytes_sent += len(opus_chunk)

                            # 每100帧打印一次进度
                            if sent_chunks % 100 == 0:
                                progress = (sent_chunks * OPUS_FRAME_SAMPLES) / total_frames * 100
                                print(f"TTS发送进度: {progress:.1f}% ({sent_chunks}块)")

                            # 适当延迟以保持同步，避免缓冲区溢出
                            await asyncio.sleep(OPUS_FRAME_MS / 1000 * 0.95)

                        except websockets.exceptions.ConnectionClosed:
                            print("TTS发送时WebSocket连接关闭")
                            break
                        except opuslib.OpusError as e:
                            print(f"TTS Opus编码错误: {e}")
                            continue
                        except Exception as e:
                            print(f"发送TTS音频数据时出错: {e}")
                            # 短暂延迟后继续尝试
                            await asyncio.sleep(0.01)
                            continue

                    actual_frames_sent = sent_chunks * OPUS_FRAME_SAMPLES
                    print(
                        f"TTS音频发送完成，发送块数: {sent_chunks}, 实际帧数: {actual_frames_sent}/{total_frames}, 总字节: {total_bytes_sent}")

            # 等待最后一个音频包被完全处理
            await asyncio.sleep(0.1)

            # 发送 TTS 结束信号
            tts_stop_response = {"type": "tts", "state": "stop"}
            await websocket.send(json.dumps(tts_stop_response))
            print("TTS 音频发送完毕，结束信号已发送")

        except Exception as e:
            print(f"发送TTS响应时出错: {e}")
            import traceback
            print(f"错误堆栈: {traceback.format_exc()}")

    async def _process_audio(self, websocket, opus_data, client_info):
        """处理接收到的Opus音频数据"""
        if not client_info or client_info.get("state") != "listening":
            return

        try:
            if not opus_data or len(opus_data) < 10:
                return

            try:
                # 解码ESP32发送的60ms OPUS数据为PCM
                pcm_data = self.opus_decoder.decode(opus_data, self.opus_decoder.frame_size)

                if not pcm_data:
                    return

                # 添加音频质量检测
                audio_np = np.frombuffer(pcm_data, dtype=np.int16)
                rms = np.sqrt(np.mean(audio_np.astype(np.float32) ** 2))

                # 如果音频能量过低，跳过发送（阈值可调整）
                if rms < 100:  # 提高阈值，低于100的RMS认为是静音或噪声
                    return

                # 获取客户端ID
                client_id = websocket.remote_address if websocket.remote_address else "default"

                # 使用VAD检测并发送到自己的语音识别服务器
                await self._detect_speech(pcm_data, client_id)

            except Exception as e:
                print(f"音频解码错误: {e}")
                return

        except Exception as e:
            print(f"处理音频数据时出错: {e}")

    async def handler(self, websocket, path):
        """处理WebSocket连接"""
        client_addr = websocket.remote_address
        print(f"WebSocket客户端已连接: {client_addr}")

        # 初始化 client_info
        client_info = {
            "state": "connected",
            "session_id": None,
            "abort_tts": False  # TTS中断标志
        }

        # 存储WebSocket连接和客户端信息
        self.clients[client_addr] = {"websocket": websocket, "client_info": client_info}

        try:
            while True:
                message = await websocket.recv()

                if isinstance(message, str):
                    # 处理JSON文本消息
                    try:
                        data = json.loads(message)
                        msg_type = data.get("type")
                        print(f"收到来自 {client_addr} 的JSON消息: {data}")

                        if msg_type == "hello":
                            await self._handle_hello(websocket, data, client_info)
                        elif msg_type == "control":
                            await self._handle_control(websocket, data, client_info)
                        elif msg_type == "iot":
                            await self._handle_iot(websocket, data, client_info)
                        elif msg_type == "listen":
                            await self._handle_listen(websocket, data, client_info)
                        elif msg_type == "abort":
                            await self._handle_abort(websocket, data, client_info)
                        else:
                            print(f"未知的JSON消息类型: {msg_type}")

                    except json.JSONDecodeError:
                        print(f"来自 {client_addr} 的无效JSON消息: {message}")
                    except Exception as e:
                        print(f"处理来自 {client_addr} 的JSON消息时出错: {e}")

                elif isinstance(message, bytes):
                    # 处理二进制Opus音频数据
                    # print(f"收到来自 {client_addr} 的二进制音频数据: {len(message)} 字节")
                    # 启动后台任务处理音频，避免阻塞接收循环
                    asyncio.create_task(self._process_audio(websocket, message, client_info))

        except websockets.exceptions.ConnectionClosedOK:
            print(f"WebSocket客户端 {client_addr} 正常断开连接")
        except websockets.exceptions.ConnectionClosedError as e:
            print(f"WebSocket客户端 {client_addr} 异常断开连接: {e}")
        except Exception as e:
            print(f"处理WebSocket连接 {client_addr} 时出错: {e}")
        finally:
            print(f"移除客户端: {client_addr}")
            if client_addr in self.clients:
                del self.clients[client_addr]
            # 确保在任何情况下都尝试关闭连接
            try:
                if not websocket.closed:
                    await websocket.close()
            except Exception as close_err:
                print(f"关闭连接 {client_addr} 时出错: {close_err}")

    async def _handle_hello(self, websocket, data, client_info):
        """处理客户端的 hello 消息"""
        print("处理 hello 消息")
        # 简单的 session_id 生成
        client_info["session_id"] = f"session_{datetime.now().timestamp()}"
        # 发送 hello 响应 (与 websocket_protocol.cc 匹配)
        hello_response = {
            "type": "hello",
            "session_id": client_info["session_id"],
            "version": 3,  # 与客户端匹配
            "transport": "websocket",
            "audio_params": {
                "format": "opus",
                "sample_rate": OPUS_SAMPLE_RATE,  # 16000Hz
                "channels": OPUS_CHANNELS,  # 1
                "frame_duration": OPUS_FRAME_MS  # 60ms
            }
        }
        await websocket.send(json.dumps(hello_response))
        client_info["state"] = "idle"  # 连接成功后进入空闲状态
        print(f"已向 {websocket.remote_address} 发送 hello 响应")

    async def _handle_control(self, websocket, data, client_info):
        """处理客户端的 control 消息"""
        command = data.get("command")
        print(f"处理 control 命令: {command}")
        if command == "start_listening":
            client_info["state"] = "listening"
            print("客户端开始监听")
        elif command == "stop_listening":
            client_info["state"] = "idle"
            print("客户端停止监听")
        elif command == "abort_speaking":
            print(f"客户端请求中断播报，原因: {data.get('reason')}")
            client_info["state"] = "idle"  # 假设中断后回到空闲
        elif command == "wake_word_detected":
            print(f"客户端检测到唤醒词: {data.get('wake_word')}")
        else:
            print(f"未知的 control 命令: {command}")

    async def _handle_iot(self, websocket, data, client_info):
        """处理客户端的 iot 消息"""
        try:
            # 检查是否有descriptors字段
            if "descriptors" in data:
                print(f"收到IoT描述符: {data['descriptors']}")
                # 存储设备描述符供后续使用
                self.iot_descriptors = data['descriptors']
            # 检查是否有states字段
            elif "states" in data:
                print(f"收到IoT状态: {data['states']}")
                # 处理设备状态
                for state in data['states']:
                    try:
                        name = state.get('name', '')
                        state_data = state.get('state', {})
                        print(f"设备 {name} 状态: {state_data}")
                    except Exception as e:
                        print(f"处理设备状态时出错: {e}")
            else:
                print(f"未知的IoT数据格式: {data}")
        except Exception as e:
            print(f"处理IoT消息时出错: {e}")
            # 不要中断连接，继续处理其他消息

    async def _handle_listen(self, websocket, data, client_info):
        """处理客户端的 listen 消息"""
        state = data.get("state")
        text = data.get("text", "")
        print(f"处理 listen 消息: state={state}, text={text}")

        if state == "detect":
            # 客户端检测到语音
            client_info["state"] = "listening"
            print(f"客户端检测到语音: {text}")

            # 检查是否需要发送动作命令
            action_command = None
            if action_command:
                print(f"发送动作命令: {action_command}")
                action_message = {
                    "type": "iot",
                    "commands": [
                        {
                            "name": "Action",
                            "method": action_command,
                            "parameters": {}
                        }
                    ]
                }
                await websocket.send(json.dumps(action_message))

            # 保持在listening状态，允许继续对话
            client_info["state"] = "listening"

        elif state == "start":
            # 客户端开始监听，但不发送TTS响应
            print("客户端开始监听")
            client_info["state"] = "listening"

        elif state == "end":
            # 客户端结束语音输入，但仍然保持在listening状态
            print(f"客户端结束语音输入: {text}")
            client_info["state"] = "listening"

    async def _handle_abort(self, websocket, data, client_info):
        """处理客户端的 abort 消息"""
        reason = data.get("reason", "unknown")
        print(f"客户端请求中断TTS播放，原因: {reason}")

        # 设置中断标志
        client_info["abort_tts"] = True

        # 立即设置状态为listening，准备接收新的语音输入
        client_info["state"] = "listening"
        print("TTS播放已中断，恢复语音输入接收")

        # _process_speech_segment 函数已删除，现在使用自己的实时语音识别服务器

    # 语音识别结果通过 _handle_complete_speech_recognition 函数处理

    def start_udp_server(self):
        """启动UDP服务器线程"""
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.bind((self.host, self.udp_port))
        print(f"UDP语音服务启动在 {self.host}:{self.udp_port}")

        # 设置接收超时
        self.udp_socket.settimeout(0.5)

        while self.running:
            try:
                # 接收UDP数据包
                opus_data, client_addr = self.udp_socket.recvfrom(2048)
                print(f"从 {client_addr} 接收UDP数据: {len(opus_data)} 字节")

                # 处理音频数据
                loop = asyncio.new_event_loop()
                processed_audio = loop.run_until_complete(self._process_audio(opus_data))
                loop.close()

                # 通过UDP发送处理后的音频数据
                if processed_audio:
                    self.udp_socket.sendto(processed_audio, client_addr)
                    print(f"向 {client_addr} 发送处理后的音频: {len(processed_audio)} 字节")
            except socket.timeout:
                # 超时是正常的，继续等待下一个数据包
                pass
            except Exception as e:
                print(f"UDP处理出错: {e}")

        # 关闭UDP套接字
        if self.udp_socket:
            self.udp_socket.close()
            self.udp_socket = None

    async def start_server(self):
        """启动WebSocket和UDP服务器"""
        self.running = True

        # 启动UDP线程 (同步，在独立线程运行)
        self.udp_thread = threading.Thread(target=self.start_udp_server, daemon=True)
        self.udp_thread.start()

        # 启动WebSocket服务器 (异步)
        server = await websockets.serve(self.handler, self.host, self.port)
        print(f"WebSocket语音服务启动在 ws://{self.host}:{self.port}")

        # 启动定期清理任务
        cleanup_task = asyncio.create_task(self._periodic_cleanup())

        # 保持服务器运行直到被中断
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            print("WebSocket服务器任务被取消")
        finally:
            # 清理
            cleanup_task.cancel()
            print("正在关闭WebSocket服务器...")
            server.close()
            await server.wait_closed()
            print("WebSocket服务器已关闭")
            self.running = False
            if self.udp_thread and self.udp_thread.is_alive():
                print("正在等待UDP线程停止...")
                self.udp_thread.join(timeout=1)
                print("UDP线程已停止")

    async def _periodic_cleanup(self):
        """定期清理任务"""
        while self.running:
            try:
                await asyncio.sleep(60)  # 每分钟运行一次
                # 清理音频缓冲区（如果有长时间未发送的数据）
                async with self.buffer_lock:
                    if self.audio_buffer and time.time() - self.last_send_time > 5.0:
                        print("清理长时间未发送的音频缓冲区")
                        self.audio_buffer.clear()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"定期清理任务出错: {e}")

    def start(self):
        """启动服务 (运行asyncio事件循环)"""
        loop = asyncio.get_event_loop()
        main_task = loop.create_task(self.start_server())
        try:
            loop.run_until_complete(main_task)
        except KeyboardInterrupt:
            print("\n检测到Ctrl+C，正在停止服务...")
            main_task.cancel()
            # Allow time for the task cancellation and cleanup
            loop.run_until_complete(main_task)
        finally:
            loop.close()
            self.running = False
            print("服务已停止")

    def stop(self):
        # 设置标志以停止UDP线程
        self.running = False

        # 等待UDP线程完成
        if self.udp_thread and self.udp_thread.is_alive():
            self.udp_thread.join(timeout=2)

        # WebSocket服务器通过取消asyncio任务来停止
        print("服务已停止")

    async def _parse_action_commands(self, text):
        """解析动作控制命令（优先处理，不经过大模型）"""
        commands = []
        text_lower = text.lower()

        # 动作控制
        if any(word in text_lower for word in ['前进', '向前', '走']):
            commands.append({"name": "Action", "method": "Walk", "parameters": {}})
        elif any(word in text_lower for word in ['后退', '倒退', '退']):
            commands.append({"name": "Action", "method": "Walk back", "parameters": {}})
        elif any(word in text_lower for word in ['站立', '站起来', '起来']):
            commands.append({"name": "Action", "method": "stand", "parameters": {}})
        elif any(word in text_lower for word in ['坐下', '坐']):
            commands.append({"name": "Action", "method": "sitdown", "parameters": {}})
        elif any(word in text_lower for word in ['睡觉', '睡', '休息']):
            commands.append({"name": "Action", "method": "sleep", "parameters": {}})
        elif any(word in text_lower for word in ['左转', '向左转', '左']):
            commands.append({"name": "Action", "method": "turn left", "parameters": {}})
        elif any(word in text_lower for word in ['右转', '向右转', '右']):
            commands.append({"name": "Action", "method": "turn right", "parameters": {}})
        elif any(word in text_lower for word in ['挥手', '打招呼', '招手']):
            commands.append({"name": "Action", "method": "wave", "parameters": {}})
        elif any(word in text_lower for word in ['停下', '停止', '别动']):
            commands.append({"name": "Action", "method": "stop", "parameters": {}})

        return commands

    async def _schedule_auto_stop(self, websocket, action_method, duration=3.0):
        """安排自动停止任务"""
        try:
            await asyncio.sleep(duration)
            # 检查是否还在执行同一个动作
            if self.current_action == action_method:
                stop_commands = [{"name": "Action", "method": "stop", "parameters": {}}]
                await self._send_iot_commands(websocket, stop_commands)
                print(f"动作 {action_method} 执行 {duration} 秒后自动停止")
                self.current_action = None
                self.is_noisy_action_running = False
        except Exception as e:
            print(f"自动停止任务出错: {e}")

    async def _parse_lamp_commands(self, text):
        """解析灯光控制命令（优先处理，不经过大模型）"""
        commands = []
        text_lower = text.lower()

        # 灯光控制
        if any(word in text_lower for word in ['开灯', '打开灯', '点亮', '照明']):
            commands.append({"name": "Lamp", "method": "TurnOn", "parameters": {}})
        elif any(word in text_lower for word in ['光灯','关灯', '关闭灯', '熄灭', '关掉灯']):
            commands.append({"name": "Lamp", "method": "TurnOff", "parameters": {}})
        elif any(word in text_lower for word in ['闪光', '闪烁', '闪灯']):
            commands.append({"name": "Lamp", "method": "flashlight", "parameters": {}})
        elif any(word in text_lower for word in ['呼吸灯', '呼吸']):
            commands.append({"name": "Lamp", "method": "breathe", "parameters": {}})

        return commands

    async def _parse_other_commands(self, text):
        """解析其他控制命令（如音量调节）"""
        commands = []
        text_lower = text.lower()

        # 音量控制
        if '音量' in text_lower:
            import re
            # 查找数字
            volume_match = re.search(r'(\d+)', text)
            if volume_match:
                volume = int(volume_match.group(1))
                if 0 <= volume <= 100:
                    commands.append({"name": "Speaker", "method": "SetVolume", "parameters": {"volume": volume}})
            elif any(word in text_lower for word in ['大声', '调大', '最大']):
                commands.append({"name": "Speaker", "method": "SetVolume", "parameters": {"volume": 80}})
            elif any(word in text_lower for word in ['小声', '调小', '静音']):
                commands.append({"name": "Speaker", "method": "SetVolume", "parameters": {"volume": 20}})

        return commands

    async def _send_iot_commands(self, websocket, commands):
        """发送IoT控制命令到ESP32"""
        if not commands:
            return

        try:
            message = {
                "type": "iot",
                "commands": commands
            }
            await websocket.send(json.dumps(message))
            print(f"发送IoT命令: {commands}")
        except Exception as e:
            print(f"发送IoT命令失败: {e}")

    async def _process_with_llm(self, text):
        """调用千问API进行对话"""
        try:
            # 千问API配置
            API_KEY = "sk-9db10a2b477f40a982d5ec833cde0592"
            API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
            MODEL = "qwen-plus"

            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }

            data = {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": "你是一个智能助手，请简洁、温馨地回应用户。符合一个活泼少女的声音"},
                    {"role": "user", "content": text}
                ],
                "temperature": 0.7,
                "max_tokens": 40
            }

            # 使用aiohttp发送异步请求
            async with aiohttp.ClientSession() as session:
                async with session.post(API_URL, headers=headers, json=data, timeout=15) as response:
                    if response.status == 200:
                        result = await response.json()
                        print(f"\n=== 千问API 响应结果 ===")
                        print(f"完整响应: {json.dumps(result, ensure_ascii=False, indent=2)}")

                        # 提取回答内容
                        reply = result.get("choices", [{}])[0].get("message", {}).get("content",
                                                                                      "抱歉，我现在无法回答这个问题。")
                        print(f"\n=== 提取的回答 ===")
                        print(f"回答内容: {reply}")
                        return reply
                    else:
                        error_text = await response.text()
                        print(f"\n=== 千问API 错误信息 ===")
                        print(f"状态码: {response.status}")
                        print(f"错误详情: {error_text}")
                        return "抱歉，我现在无法回答这个问题。"

        except aiohttp.ClientError as e:
            print(f"\n=== 网络错误 ===")
            print(f"错误信息: {e}")
            return "抱歉，我现在无法回答这个问题。"
        except Exception as e:
            print(f"\n=== 未知错误 ===")
            print(f"错误信息: {e}")
            import traceback
            print(f"错误堆栈: {traceback.format_exc()}")
            return "抱歉，我现在无法回答这个问题。"

    async def _text_to_speech(self, text, max_retries=3):
        """使用零样本语音合成API进行语音合成"""
        retry_count = 0
        last_error = None

        while retry_count < max_retries:
            try:
                # 准备请求参数
                url = "http://113.47.5.231:8090/tts/zero_shot"

                # 使用 aiohttp 发送异步请求
                async with aiohttp.ClientSession() as session:
                    # 准备请求数据
                    data = aiohttp.FormData()
                    data.add_field('text', text)
                    data.add_field('stream', "false")

                    async with session.post(url, data=data) as response:
                        if response.status == 200:
                            result = await response.json()
                            if "results" in result and len(result["results"]) > 0:
                                # 解码base64音频数据
                                audio_base64 = result["results"][0]["audio_base64"]
                                audio_bytes = base64.b64decode(audio_base64)

                                # 将音频数据转换为标准PCM WAV格式
                                try:
                                    # 使用 soundfile 读取音频数据
                                    import soundfile as sf
                                    import io

                                    # 从内存中读取音频数据
                                    audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))

                                    # 确保音频是单声道
                                    if len(audio_data.shape) > 1:
                                        audio_data = audio_data.mean(axis=1)

                                    # 确保采样率是16000Hz
                                    if sample_rate != OPUS_SAMPLE_RATE:
                                        from scipy import signal
                                        audio_data = signal.resample(audio_data, int(len(
                                            audio_data) * OPUS_SAMPLE_RATE / sample_rate))
                                        sample_rate = OPUS_SAMPLE_RATE

                                    # 将音频数据转换为16位整数
                                    audio_data = (audio_data * 32767).astype(np.int16)

                                    # 创建WAV文件
                                    wav_io = io.BytesIO()
                                    with wave.open(wav_io, 'wb') as wav_file:
                                        wav_file.setnchannels(OPUS_CHANNELS)
                                        wav_file.setsampwidth(2)  # 16位
                                        wav_file.setframerate(OPUS_SAMPLE_RATE)
                                        wav_file.writeframes(audio_data.tobytes())

                                    # 获取WAV数据
                                    wav_io.seek(0)
                                    return wav_io.read()

                                except Exception as e:
                                    print(f"音频格式转换失败: {e}")
                                    import traceback
                                    print(f"错误堆栈: {traceback.format_exc()}")
                                    last_error = f"音频格式转换失败: {e}"
                                    retry_count += 1
                                    continue
                            else:
                                print("语音合成返回结果为空")
                                last_error = "语音合成返回结果为空"
                        else:
                            error_text = await response.text()
                            print(f"语音合成请求失败: {response.status}")
                            print(f"错误详情: {error_text}")
                            last_error = f"请求失败: {response.status}, {error_text}"

                # 如果执行到这里，说明需要重试
                retry_count += 1
                if retry_count < max_retries:
                    print(f"第 {retry_count} 次重试...")
                    await asyncio.sleep(1)  # 等待1秒后重试
                else:
                    print(f"达到最大重试次数 ({max_retries})，放弃重试")
                    break

            except Exception as e:
                print(f"语音合成出错: {e}")
                import traceback
                print(f"错误堆栈: {traceback.format_exc()}")
                last_error = str(e)
                retry_count += 1
                if retry_count < max_retries:
                    print(f"第 {retry_count} 次重试...")
                    await asyncio.sleep(1)  # 等待1秒后重试
                else:
                    print(f"达到最大重试次数 ({max_retries})，放弃重试")
                    break

        print(f"语音合成最终失败: {last_error}")
        return None


if __name__ == '__main__':
    service = VoiceService()
    service.start()