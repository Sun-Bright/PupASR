import os
import random
from glob import glob

# === 配置 ===
USER_IDS = [1, 2, 3, 4, 5, 6]
TXT_DIR = "record2/txt"
RECORD_DIR = "record2/recordings_processed"
OUTPUT_DIR = os.path.join(TXT_DIR, "split")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 第一步：生成每个用户的 train_user_X.txt ===
print("=== 生成每个用户的训练对齐文本 ===")
for user_id in USER_IDS:
    text_file = os.path.join(TXT_DIR, f"recording_plan_user_{user_id}.txt")
    output_file = os.path.join(TXT_DIR, f"train_user_{user_id}.txt")
    wav_dir = os.path.join(RECORD_DIR, f"user_{user_id}")

    if not os.path.exists(text_file):
        print(f"[跳过] 文本文件不存在: {text_file}")
        continue

    with open(text_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    with open(output_file, 'w', encoding='utf-8') as out:
        for idx, line in enumerate(lines, 1):
            wav_name = f"{idx:04d}.wav"
            wav_path = os.path.join(wav_dir, wav_name).replace("\\", "/")
            out.write(f"{wav_path}\t{line}\n")

    print(f"[完成] 生成: {output_file}（共 {len(lines)} 条）")

# === 第二步：合并所有用户数据，打乱并划分为 train/dev ===
print("\n=== 合并并划分训练集与验证集 ===")
all_entries = []
for path in sorted(glob(os.path.join(TXT_DIR, "train_user_*.txt"))):
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
        all_entries.extend(lines)

print(f"[收集] 总样本数: {len(all_entries)}")

# 打乱数据
random.shuffle(all_entries)

# 划分比例
train_size = int(len(all_entries) * 0.8)
train_entries = all_entries[:train_size]
dev_entries = all_entries[train_size:]

# 保存划分结果
train_out = os.path.join(OUTPUT_DIR, "train.txt")
dev_out = os.path.join(OUTPUT_DIR, "dev.txt")

with open(train_out, "w", encoding="utf-8") as f:
    f.write("\n".join(train_entries))
with open(dev_out, "w", encoding="utf-8") as f:
    f.write("\n".join(dev_entries))

print(f"[完成] 训练集: {train_out}（{len(train_entries)} 条）")
print(f"[完成] 验证集: {dev_out}（{len(dev_entries)} 条）")
