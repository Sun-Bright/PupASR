import json
import re

input_files = [
    "LCCC-base_train.json",
    "LCCC-base_valid.json",
    "LCCC-base_test.json"
]

output_file = "corpus.txt"

# 仅保留中文汉字（去除所有非中文字符）
def clean_text(text):
    text = re.sub(r"[^\u4e00-\u9fa5]", "", text)
    return text.strip()

seen = set()
with open(output_file, "w", encoding="utf-8") as out_f:
    for file in input_files:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            for dialog in data:
                for sentence in dialog:
                    cleaned = clean_text(sentence)
                    if cleaned and cleaned not in seen and 4 <= len(cleaned) <= 50:
                        seen.add(cleaned)
                        out_f.write(cleaned + "\n")
