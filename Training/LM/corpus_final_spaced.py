# prepare_for_kenlm.py
input_path = "corpus_merged.txt"
output_path = "corpus_final_spaced.txt"

with open(input_path, "r", encoding="utf-8") as fin, \
     open(output_path, "w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if line:
            spaced = " ".join(list(line))
            fout.write(spaced + "\n")

print(f"✅ 已生成分字语料：{output_path}")
