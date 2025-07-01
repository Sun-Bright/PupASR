import random

# 要合并的语料文件
input_files = ["corpus.txt", "hotwords.txt"]
output_file = "corpus_merged.txt"

# 收集所有行（包括空格、标点、重复行，除掉空行）
all_lines = []
for file_path in input_files:
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip("\n\r")  # 去掉换行符，但不 strip 内容
            if line.strip():  # 忽略纯空白行（不含任何可见字符）
                all_lines.append(line)

# 原始统计
print(f"📄 corpus.txt 行数: {sum(1 for _ in open('corpus.txt', 'r', encoding='utf-8'))}")
print(f"📄 hotwords.txt 行数: {sum(1 for _ in open('hotwords.txt', 'r', encoding='utf-8'))}")
print(f"🧮 合并后（有效行数）: {len(all_lines)}")

# 打乱顺序
random.shuffle(all_lines)

# 写入合并后文件
with open(output_file, "w", encoding="utf-8") as fout:
    fout.write("\n".join(all_lines) + "\n")

print(f"✅ 已生成打乱合并语料: {output_file}")
