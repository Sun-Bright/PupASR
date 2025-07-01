from transformers import Wav2Vec2Processor

# 加载你想查看的模型，比如 wbbbbb 中文模型
processor = Wav2Vec2Processor.from_pretrained("/root/projects/wav2vec2-live/my_model/final_stage2_output/checkpoint-463")

# 获取字符表字典
vocab_dict = processor.tokenizer.get_vocab()

# 按照 token 顺序排列（可选）
sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])

# 打印数量
print(f"字符表共 {len(vocab_dict)} 个字符")

# 保存为 unigrams.txt（每行一个字符）
with open("unigrams_wbbb.txt", "w", encoding="utf-8") as f:
    for token, _ in sorted_vocab:
        f.write(token + "\n")
