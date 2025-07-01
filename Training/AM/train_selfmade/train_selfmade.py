import os
import torch
import torchaudio
from datasets import Dataset, Audio
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback,
)
from jiwer import cer
import numpy as np
from pathlib import Path
from transformers.trainer_utils import EvalPrediction

# === 配置路径 ===
PRETRAINED_MODEL = "/root/projects/wav2vec2-live/my_chinese_model/saved_chinese_model_wbbb_unfreeze/checkpoint-5500"
TRAIN_TXT = "./record/txt/split/train.txt"
DEV_TXT = "./record/txt/split/dev.txt"
OUTPUT_DIR = "./my_model/finetune_cpu_output"
SAMPLE_RATE = 16000
USE_FP16 = torch.cuda.is_available()

# === 加载模型和处理器 ===
processor = Wav2Vec2Processor.from_pretrained(PRETRAINED_MODEL)
model = Wav2Vec2ForCTC.from_pretrained(PRETRAINED_MODEL)
model.freeze_feature_encoder()

# === 加载数据 ===
def load_txt_data(txt_path):
    paths, texts = [], []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            if "\t" in line:
                path, text = line.strip().split("\t")
                paths.append(path)
                texts.append(text)
    return Dataset.from_dict({"audio": paths, "text": texts})

train_dataset = load_txt_data(TRAIN_TXT)
dev_dataset = load_txt_data(DEV_TXT)
train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
dev_dataset = dev_dataset.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))

# === 特征处理 ===
def prepare(example):
    audio = example["audio"]
    example["input_values"] = processor(audio["array"], sampling_rate=SAMPLE_RATE).input_values[0]
    with processor.as_target_processor():
        example["labels"] = processor(example["text"]).input_ids
    return example

train_dataset = train_dataset.map(prepare, remove_columns=["audio", "text"])
dev_dataset = dev_dataset.map(prepare, remove_columns=["audio", "text"])

# === Collator ===
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(label_features, padding=self.padding, return_tensors="pt")
        batch["labels"] = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        return batch

# === 每条样本 CER 输出 ===
def compute_metrics(pred: EvalPrediction):
    pred_ids = np.argmax(pred.predictions, axis=-1)
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, group_tokens=False)

    min_len = min(len(pred_str), len(label_str))
    pred_str = pred_str[:min_len]
    label_str = label_str[:min_len]

    cer_list = []
    print("\n" + "="*30 + " 验证集逐条输出 " + "="*30)
    for i, (ref, hyp) in enumerate(zip(label_str, pred_str)):
        try:
            cer_i = cer(ref, hyp)
            cer_list.append(cer_i)
            print(f"[样本 {i:03}]\n  REF: {ref}\n  HYP: {hyp}\n  CER: {cer_i:.4f}\n")
        except Exception as e:
            print(f"[❌ 样本 {i} 跳过] {e}")
            continue

    cer_avg = np.mean(cer_list) if cer_list else 1.0
    print(f"[✅ 验证完成] 样本数: {len(cer_list)}，平均 CER: {cer_avg:.4f}")
    print("="*80 + "\n")
    return {"cer": cer_avg}

# === 训练参数 ===
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=3,
    num_train_epochs=100,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    logging_steps=25,
    dataloader_num_workers=4,
    remove_unused_columns=False,
    group_by_length=False,
    fp16=USE_FP16,
    report_to=["tensorboard"],
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False,
)

# === 日志记录回调 ===
class EvalLoggerCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        log_path = os.path.join(args.output_dir, "eval_log.txt")
        with open(log_path, "a", encoding="utf-8") as f:
            log_line = {
                "epoch": metrics.get("epoch"),
                "eval_loss": metrics.get("eval_loss"),
                "eval_cer": metrics.get("eval_cer"),  # ✅ 修复字段名
                "eval_runtime": metrics.get("eval_runtime"),
            }
            f.write(str(log_line) + "\n")
            print(f"[📄 已记录评估结果] {log_line}")

# === checkpoint 检查 ===
last_checkpoint = None
checkpoint_dir = Path(OUTPUT_DIR)
if checkpoint_dir.exists():
    checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"), key=lambda x: int(x.name.split("-")[-1]))
    if checkpoints:
        last_checkpoint = str(checkpoints[-1])
        print(f"[恢复训练] 检测到 checkpoint: {last_checkpoint}")
    else:
        print("[训练开始] 没有检测到 checkpoint，将从头开始训练。")
else:
    print("[训练开始] 输出目录不存在，将从头开始训练。")

# === Trainer 实例 ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    tokenizer=processor,
    data_collator=DataCollatorCTCWithPadding(processor=processor),
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=3),
        EvalLoggerCallback()
    ]
)

# === 启动训练 ===
state_path = os.path.join(last_checkpoint or "", "trainer_state.json")
if last_checkpoint and os.path.isfile(state_path):
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    print(f"[⚠️ 忽略无效 checkpoint] {last_checkpoint}")
    trainer.train()