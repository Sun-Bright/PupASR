#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import soundfile as sf
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import numpy as np
import sys

# 加载 CER 评估类（和训练脚本一样，保持风格）
sys.path.append("/mnt/data/hf_cache/metrics/cer/default")
from cer import CER

TEST_INDEX = "./data/MDT2021S003/prepared_bigdata/test.txt"
MODEL_DIR  = "./saved_chinese_model_wbbb_unfreeze/checkpoint-5500"

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_test_data(test_index_file):
    audio_paths = []
    transcripts = []
    with open(test_index_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            audio_paths.append(parts[0])
            transcripts.append(parts[1])
    return audio_paths, transcripts

def speech_file_to_array(path, target_sample_rate=16000):
    speech, sr = sf.read(path)
    if sr != target_sample_rate:
        import torchaudio
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        # 如果是多通道音频，取第一通道
        if len(speech.shape) > 1:
            speech = speech[:, 0]
        speech = resampler(torch.tensor(speech)).numpy()
    return speech

def main():
    print(f"[设备] 使用设备: {device}")

    processor = Wav2Vec2Processor.from_pretrained(MODEL_DIR)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    cer_metric = CER()

    audio_paths, transcripts = load_test_data(TEST_INDEX)
    print(f"[数据] 载入 {len(audio_paths)} 条测试样本")

    pred_texts = []
    with torch.no_grad():
        for i in tqdm(range(len(audio_paths)), desc="测试进度"):
            audio = speech_file_to_array(audio_paths[i])
            input_values = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True).input_values.to(device)
            logits = model(input_values).logits
            pred_ids = torch.argmax(logits, dim=-1)
            pred_str = processor.batch_decode(pred_ids)[0]
            pred_texts.append(pred_str)

    # 完全复用训练的 CER 评估逻辑
    cer_score = cer_metric._compute(predictions=pred_texts, references=transcripts)
    print(f"[结果] 测试集 CER: {cer_score:.4f}")

    print("\n[样例输出]")
    for i in range(min(5, len(pred_texts))):
        print(f"[样例{i}] 预测: {pred_texts[i]}")
        print(f"[样例{i}] 正确: {transcripts[i]}")

if __name__ == "__main__":
    main()

