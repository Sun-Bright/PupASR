import os
import torchaudio
from glob import glob

INPUT_ROOT = "record2/recordings"
SAMPLE_RATE = 16000

def process_wav(path):
    waveform, sr = torchaudio.load(path)
    # 转为单声道
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # 重采样
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)
    # 保存（覆盖）
    output_path = path.replace("recordings", "recordings_processed")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchaudio.save(output_path, waveform, SAMPLE_RATE)
    print(f"[处理] {path}")

# 批量处理所有 .wav
for wav_path in glob(f"{INPUT_ROOT}/user_*/**/*.wav", recursive=True):
    try:
        process_wav(wav_path)
    except Exception as e:
        print(f"[跳过] {wav_path} 错误: {str(e)}")
