#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MagicData-RAMC 数据集预处理脚本（官方划分版，已移除所有标点）

主要功能：
1. 加载原始音频与对应转写文本，过滤特殊标记。
2. 文本清洗：数字转汉字、删除英文、移除所有标点、合并空格。
3. 按时长筛选：保留时长在 min_duration～max_duration 秒之间的片段。
4. 样本数量控制：可选抽样至 target_num 条。
5. 按官方 15∶1∶2 (train/dev/test) 比例划分数据。
6. 音频处理：重采样到 16kHz，可选静音裁剪、RMS 归一化。
7. 支持快速抽样模式 sample_ratio（二次抽样）。
8. 输出 train.txt、dev.txt、test.txt，格式：wav_path\tcleaned_text。

使用示例：
python prepare_magicdata.py \
    --input_dir data/MDT2021S003 \
    --output_dir data/MDT2021S003/prepared_bigdata \
    --min_duration 2.0 --max_duration 10.0 \
    --target_num 120000 \
    --dev_ratio 0.0556 --test_ratio 0.1111 \
    --sample_ratio 1.0 \
    --silence_thresh 0.01 --do_norm
"""
import os
import re
import random
import argparse
import soundfile as sf
import numpy as np

# ---------------------------------------------
# 数字转换：阿拉伯数字 -> 中文数字
# ---------------------------------------------
digits_map = ['零','一','二','三','四','五','六','七','八','九']
def arabic_to_chinese(num):
    try:
        num = int(num)
    except:
        return num
    if num == 0:
        return digits_map[0]
    s = str(num)
    length = len(s)
    result = ""
    prev_zero = False
    for i, ch in enumerate(s):
        idx = length - 1 - i
        digit = int(ch)
        unit = ['','十','百','千','万','十万','百万','千万','亿','十亿'][idx] if idx < 10 else ''
        if digit != 0:
            ch_char = '两' if digit == 2 and idx >= 2 else digits_map[digit]
            if prev_zero:
                result += '零'
            result += ch_char + unit
            prev_zero = False
        else:
            prev_zero = True
    if 10 <= num < 20:
        result = result.lstrip('一')
    return result

# ---------------------------------------------
# 文本清洗：移除所有标点
# ---------------------------------------------
def clean_text(text):
    # 1. 去除特殊标记
    for m in ['[+]','[*]','[LAUGHTER]','[SONANT]','[MUSIC]']:
        text = text.replace(m, '')
    # 2. 数字转汉字
    text = re.sub(r'\d+', lambda m: arabic_to_chinese(m.group()), text)
    # 3. 删除英文单词
    text = re.sub(r'[A-Za-z]+', '', text)
    # 4. 合并空格并去首尾空格
    text = re.sub(r'\s+', ' ', text).strip()
    # 5. 移除所有非中文字符（包括中英文标点）
    text = re.sub(r'[^一-鿿\s]', '', text)
    # 6. 再次合并空格
    return re.sub(r'\s+', ' ', text).strip()

# ---------------------------------------------
# 静音裁剪
# ---------------------------------------------
def trim_silence(audio, thresh):
    if thresh <= 0:
        return audio
    abs_audio = np.abs(audio)
    non_silence = np.where(abs_audio > thresh)[0]
    if not len(non_silence):
        return audio
    return audio[non_silence[0]: non_silence[-1] + 1]

# ---------------------------------------------
# 主流程
# ---------------------------------------------
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # 收集文件列表
    aud_exts = ['.wav', '.WAV']
    txt_exts = ['.txt', '.TXT', '.trn', '.TRN']
    audio_files = {}
    transcripts = {}
    for root, _, files in os.walk(args.input_dir):
        for fname in files:
            base, ext = os.path.splitext(fname)
            path = os.path.join(root, fname)
            if ext in aud_exts:
                audio_files[base] = path
            elif ext in txt_exts and base.upper() not in ['UTTERANCEINFO', 'SPKINFO']:
                transcripts[base] = path

    # 收集并过滤片段
    utts = []
    for base, txt_path in transcripts.items():
        if base not in audio_files:
            continue
        wav_path = audio_files[base]
        with open(txt_path, 'r', encoding='utf-8') as rf:
            for line in rf:
                parts = line.strip().split(None, 3)
                if len(parts) < 4:
                    continue
                try:
                    start, end = map(float, parts[0].strip('[]').split(','))
                except:
                    continue
                duration = end - start
                if duration < args.min_duration or duration > args.max_duration:
                    continue
                text = clean_text(parts[3])
                if text:
                    utts.append((wav_path, start, end, text))

    # 总样本数控制
    random.seed(1234)
    if args.target_num > 0 and len(utts) > args.target_num:
        utts = random.sample(utts, args.target_num)

    # 二次抽样
    if 0 < args.sample_ratio < 1:
        uds = int(len(utts) * args.sample_ratio)
        utts = random.sample(utts, max(1, uds))

    # 官方划分：train/dev/test
    random.shuffle(utts)
    n_dev  = max(1, int(len(utts) * args.dev_ratio))
    n_test = max(1, int(len(utts) * args.test_ratio))
    dev_utts  = utts[:n_dev]
    test_utts = utts[n_dev:n_dev + n_test]
    train_utts = utts[n_dev + n_test:]

    # 按音频文件分组
    groups = {}
    for item in train_utts + dev_utts + test_utts:
        groups.setdefault(item[0], []).append(item[1:])

    # 打开输出文件
    train_f = open(os.path.join(args.output_dir, 'train.txt'), 'w', encoding='utf-8')
    dev_f   = open(os.path.join(args.output_dir, 'dev.txt'),   'w', encoding='utf-8')
    test_f  = open(os.path.join(args.output_dir, 'test.txt'),  'w', encoding='utf-8')
    counters = {}

    # 处理并保存片段
    for wav_path, segs in groups.items():
        data, sr = sf.read(wav_path)
        if data.ndim > 1:
            data = data[:, 0]
        if sr != 16000:
            import librosa
            data = librosa.resample(data, sr, 16000)
            sr = 16000
        else:
            if data.dtype not in [np.float32, np.float64]:
                data = data / np.max(np.abs(data))
        base = os.path.splitext(os.path.basename(wav_path))[0]
        counters.setdefault(base, 0)

        for start, end, text in segs:
            counters[base] += 1
            s = int(round(start * sr))
            e = int(round(end   * sr))
            segment = data[s:e]
            if args.silence_thresh > 0:
                segment = trim_silence(segment, args.silence_thresh)
            if segment.size == 0:
                continue
            if args.do_norm:
                rms = np.sqrt(np.mean(segment**2))
                if rms > 0:
                    segment = segment * (0.1 / rms)
                    segment = np.clip(segment, -1.0, 1.0)
            wav_name = f"{base}_{counters[base]:04d}.wav"
            out_path = os.path.join(args.output_dir, wav_name)
            sf.write(out_path, (segment * 32767).astype(np.int16), sr)
            line = f"{out_path}\t{text}\n"
            if (wav_path, start, end, text) in train_utts:
                train_f.write(line)
            elif (wav_path, start, end, text) in dev_utts:
                dev_f.write(line)
            else:
                test_f.write(line)

    train_f.close()
    dev_f.close()
    test_f.close()
    print(f"完成：train {len(train_utts)}条，dev {len(dev_utts)}条，test {len(test_utts)}条，保存在 {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MagicData-RAMC 预处理脚本（官方划分版，移除标点）")
    parser.add_argument('--input_dir',    type=str, required=True,  help='原始数据目录')
    parser.add_argument('--output_dir',   type=str, default='prepared', help='输出目录')
    parser.add_argument('--min_duration', type=float, default=2.0,    help='最短时长（秒）')
    parser.add_argument('--max_duration', type=float, default=6.0,    help='最长时长（秒）')
    parser.add_argument('--target_num',   type=int,   default=0,      help='目标样本总数，0表示不限制')
    parser.add_argument('--dev_ratio',    type=float, default=1/18,   help='验证集比例 (官方 1/18)')
    parser.add_argument('--test_ratio',   type=float, default=2/18,   help='测试集比例 (官方 2/18)')
    parser.add_argument('--sample_ratio', type=float, default=1.0,    help='快速抽样比例')
    parser.add_argument('--silence_thresh', type=float, default=0.0,  help='静音裁剪阈值')
    parser.add_argument('--do_norm',      action='store_true',       help='RMS 归一化')
    args = parser.parse_args()
    main(args)
