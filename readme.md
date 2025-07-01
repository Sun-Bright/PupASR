# 机器狗狗中文语音识别系统

本项目旨在构建一个高效、准确的中文语音识别系统，部署用于语音交互机器狗，在语音指令识别和低延迟交互场景表现较好。系统基于 Wav2Vec 2.0 模型进行开发，并通过多轮微调优化，以满足实际应用场景的需求。

## 项目结构

```plaintext
PupASR/
├── Deployment/ # 语音识别服务代码
│   ├── client.py # 客户端示例代码
│   ├── test_client.py # 测试客户端
│   ├── unigrams.py # 词汇表生成脚本
│   ├── unigrams_wbbb.txt # 词汇表文件
│   ├── voice_service.py # 语音交互整体代码
│   ├── wbbb_api.py # 语音识别接口封装
│   └── wbbb_gpu_api.py # GPU加速版接口
├── Training/
│   ├── AM/ # 声学模型相关代码
│   │   ├── train_magicdata/ # MagicData-RAMC数据集训练脚本
│   │   │   ├── prepare_magicdata.py # 数据准备
│   │   │   ├── preprocess_wbbb.py # 数据预处理
│   │   │   ├── train_wbbb.py # 训练脚本
│   │   │   ├── train_wbbb_unfreeze.py # 微调脚本
│   │   │   └── test.py # 测试脚本
│   │   ├── train_selfmade/ # 自制数据训练脚本
│   │   │   ├── preprocess_audio_selfmade.py # 数据预处理
│   │   │   ├── prepared_selfmade.py # 准备数据
│   │   │   ├── record.py # 录音脚本
│   │   │   ├── train_selfmade.py # 训练脚本
│   │   │   └── train_selfmade2.py # 训练脚本2
│   ├── LM/ # 语言模型相关代码
│   │   ├── corpus_final_spaced.py # 语料处理
│   │   ├── process_lccc.py # LCCC数据处理
│   │   └── shuffle.py # 数据打乱脚本
├── .gitignore # Git忽略文件，包含 .idea/
├── README.md # 项目说明文档
└── ...
```




## 核心技术

### 声学模型

- **基础模型**：`wbbbbb/wav2vec2-large-chinese-zh-cn`，具备强大的音频特征提取能力和高准确率，特别适合中文端到端识别任务。
- **数据集微调**：
  - **MagicData-RAMC 数据集**：经过系统性预处理，包括音频裁剪、归一化、文本清洗等操作，最终划分为训练集、验证集和测试集。
  - **自录语音数据集**：由3女3男大学生录制，涵盖日常交流语句、命令指令等，进一步提升模型对真实语音场景的适应能力。
- **训练策略**：
  - **冻结特征提取层**：在微调初期固定底层卷积编码器，稳定训练过程，快速适配中文语料。
  - **解冻特征提取层**：在最优 checkpoint 基础上解冻全部参数，进一步挖掘语音-文本对齐能力，提升模型对细粒度发音差异的感知能力。

### 语言模型

- **KenLM 语言模型**：采用分字级 5-gram 模型，集成至 CTC 解码流程中，提升解码准确率和热词命中率。
- **语料来源**：
  - LCCC 开放中文对话语料（清洗处理后保留规范句子）。
  - 自定义热词相关语句（如“小智前进”、“请坐下”等）。

## 部署与应用

- **后端服务**：基于 PyTorch 和 FastAPI 构建，支持实时语音流处理的中文语音识别 API。
- **接口类型**：
  - HTTP 接口：支持标准音频文件上传识别。
  - WebSocket 接口：支持音频流式传输与实时反馈，适用于语音聊天机器人等交互式应用场景。
- **模型融合**：声学模型在 GPU 上进行高效推理，语言模型与热词处理在 CPU 上运行，兼顾识别准确率与响应速度。

## 环境依赖


以下是运行本项目所需的主要环境依赖。请确保您的开发环境中已安装以下库和工具。

- **Python**: 3.8.10
- **PyTorch**: 2.4.1
- **Transformers**: 4.46.3
- **FastAPI**: 0.115.12
- **Uvicorn**: 0.33.0
- **pyctcdecode**: 0.6.0
- **KenLM**: 0.3.0
- **websockets**: 13.1
- **aiohttp**: 3.10.11
- **numpy**: 1.24.4
- **scipy**: 1.10.1
- **soundfile**: 0.13.1
- **tokenizers**: 0.20.3
- **huggingface-hub**: 0.30.2
- **starlette**: 0.44.0

