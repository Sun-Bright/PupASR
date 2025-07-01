# ======= train_wbbb_stable_freeze.py（稳定训练 + 数据增强 + EarlyStopping）=======

import os
import warnings
import torch
import gc
import psutil
import numpy as np
import random
import torchaudio
import sys
from datasets import load_from_disk
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    Wav2Vec2Config,
    TrainingArguments,
    Trainer,
    TrainerCallback, TrainerState, TrainerControl,
    EarlyStoppingCallback
)
from dataclasses import dataclass
from typing import Dict, List, Union, Optional

sys.path.append("/mnt/data/hf_cache/metrics/cer/default")
from cer import CER

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["USE_TF"] = "0"
warnings.filterwarnings("ignore")

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"[] 内存使用: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    if torch.cuda.is_available():
        print(
            f"[] GPU 内存: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB / {torch.cuda.memory_reserved() / 1024 / 1024:.2f} MB")

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[float], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features, label_features = [], []
        for f in features:
            if "input_values" not in f or "labels" not in f:
                continue
            input_val = f["input_values"]
            if isinstance(input_val, np.ndarray):
                input_val = input_val.tolist()
            input_features.append({"input_values": input_val})
            label_features.append({"input_ids": f["labels"]})

        if not input_features:
            return {"input_values": [], "labels": []}

        batch = self.processor.pad(input_features, padding=self.padding, max_length=self.max_length,
                                   return_tensors="pt")
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(label_features, padding=self.padding, max_length=self.max_length,
                                              return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["input_ids"] == self.processor.tokenizer.pad_token_id, -100)
        batch["labels"] = labels
        return batch

cer_metric = CER()

def compute_metrics(pred):
    pred_ids = torch.argmax(torch.tensor(pred.predictions), dim=-1)
    pred_str = processor.batch_decode(pred_ids)

    # 转换 label_ids -> 过滤 -100
    label_str = []
    for label in pred.label_ids:
        label = [l for l in label if l != -100]
        label_str.append(processor.tokenizer.decode(label, skip_special_tokens=True))

    cer_score = cer_metric._compute(predictions=pred_str, references=label_str)
    print(f"[评估] CER: {cer_score:.4f}")
    print("[样例输出]")
    for i in range(min(3, len(pred_str))):
        print(f"[样例{i}]  预测: {pred_str[i]}")
        print(f"[样例{i}]  正确: {label_str[i]}")
    return {"cer": cer_score}

class TxtLoggerCallback(TrainerCallback):
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is not None:
            try:
                with open(LOG_TXT, mode="a", encoding="utf-8") as f:
                    f.write(f"[步骤 {state.global_step}] {logs}\n")
                    if 'eval_cer' in logs:
                        f.write(f"[CER 记录] 步骤 {state.global_step} cer={logs['eval_cer']:.4f}\n")
            except Exception as e:
                print(f"[] 写入日志失败: {str(e)}")

class MemoryManagementCallback(TrainerCallback):
    def __init__(self, log_memory=True):
        self.log_memory = log_memory

    def on_step_end(self, args, state, control, logs=None, **kwargs):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self.log_memory and state.global_step % 500 == 0:
            print(f"[] 步骤 {state.global_step} 内存状态:")
            print_memory_usage()

        if logs and "grad_norm" in logs:
            grad_val = logs["grad_norm"]
            if isinstance(grad_val, float) and (np.isnan(grad_val) or grad_val > 100):
                print(f"[] 第 {state.global_step} 步 grad_norm 异常（{grad_val}），终止训练")
                control.should_training_stop = True

def augment_waveform(waveform, sample_rate):
    if random.random() < 0.2:
        gain_db = random.uniform(-6, 6)
        waveform = waveform * (10 ** (gain_db / 20))

    if random.random() < 0.2:
        speed_factor = random.uniform(0.9, 1.1)
        effects = [['speed', str(speed_factor)], ['rate', str(sample_rate)]]
        waveform, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, sample_rate, effects)

    if random.random() < 0.2:
        noise = torch.randn_like(waveform) * 0.005
        waveform = waveform + noise

    return waveform

class AudioDatasetWrapper:
    def __init__(self, dataset, sample_rate=16000, apply_augment=False):
        self.dataset = dataset
        self.sample_rate = sample_rate
        self.apply_augment = apply_augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            item = self.dataset[idx]
            if "input_values" not in item or "labels" not in item:
                return {"input_values": np.zeros(1408), "labels": [0]}
            input_tensor = torch.tensor(item["input_values"]).unsqueeze(0)
            if self.apply_augment:
                input_tensor = augment_waveform(input_tensor, self.sample_rate)
            return {"input_values": input_tensor.squeeze().numpy(), "labels": item["labels"]}
        except:
            return {"input_values": np.zeros(1408), "labels": [0]}

# ====== 主函数入口 ======
def main():
    global processor, LOG_TXT, BEST_MODEL_DIR

    MODEL_PATH = "/mnt/data/wav2vec2-wbbb-offline"
    DATASET_PATH = "/mnt/data/preprocessed_magicdata_wbbb_fixed"
    OUTPUT_DIR = "./saved_chinese_model_wbbb_stable"
    LOG_TXT = os.path.join(OUTPUT_DIR, "training_log.txt")
    BEST_MODEL_DIR = os.path.join(OUTPUT_DIR, "best_model")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)

    processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH, local_files_only=True)
    config = Wav2Vec2Config.from_pretrained(MODEL_PATH, local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH, config=config, local_files_only=True, low_cpu_mem_usage=True)
    model.freeze_feature_encoder()
    model.to("cuda")

    dataset = load_from_disk(DATASET_PATH)
    dataset["validation"] = dataset["validation"].select(range(1408))
    data_collator = DataCollatorCTCWithPadding(processor=processor)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=6,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        learning_rate=5e-6,
        warmup_ratio=0.1,
        num_train_epochs=10,
        logging_steps=25,
        fp16=True,
        fp16_full_eval=True,
        eval_accumulation_steps=32,
        report_to=[],
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        group_by_length=False,
        save_safetensors=True,
        max_grad_norm=1.0,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        ddp_find_unused_parameters=True,
        ddp_bucket_cap_mb=100,
        seed=42,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=AudioDatasetWrapper(dataset["train"], apply_augment=True),
        eval_dataset=AudioDatasetWrapper(dataset["validation"], apply_augment=False),
        tokenizer=processor,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            TxtLoggerCallback(),
            MemoryManagementCallback(log_memory=True),
            EarlyStoppingCallback(early_stopping_patience=3)
        ],
    )

    resume_from_checkpoint = any(dirname.startswith("checkpoint-") for dirname in os.listdir(OUTPUT_DIR))
    if resume_from_checkpoint:
        print("[] 检测到 checkpoint，禁用 optimizer 恢复以解决参数组不匹配错误。")
        trainer._load_optimizer_and_scheduler = lambda *args, **kwargs: None

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    print("[] 训练完成，正在保存最佳模型...")
    if trainer.state.best_model_checkpoint is not None:
        print(f"[] 最佳模型保存在: {trainer.state.best_model_checkpoint}")
        trainer.save_model(trainer.state.best_model_checkpoint)
    else:
        print(f"[] 未找到最佳模型，保存最后状态到: {BEST_MODEL_DIR}")
        trainer.save_model(BEST_MODEL_DIR)

if __name__ == "__main__":
    main()