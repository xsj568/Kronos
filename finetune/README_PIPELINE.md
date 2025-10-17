# Kronos 训练流水线使用指南

本文档介绍了 Kronos 金融模型的完整训练流水线，包括数据处理、分词模型训练、预测模型训练和预测功能。流水线支持 GPU 和 CPU 训练模式，以及不同的数据源。

## 目录结构

```
finetune/
├── config.py                  # 全局配置
├── dataset.py                 # 数据集定义
├── main.py                    # 完整训练流水线的类实现
├── qlib_data_preprocess.py    # Qlib 数据预处理
├── sina_data_processor.py     # 新浪数据处理器
├── train_complete_pipeline.py # 完整训练流水线脚本
├── train_predictor.py         # 预测模型训练（GPU 版本）
├── train_predictor_cpu.py     # 预测模型训练（CPU 版本）
├── train_tokenizer.py         # 分词模型训练（GPU 版本）
├── train_tokenizer_cpu.py     # 分词模型训练（CPU 版本）
└── utils/
    ├── __init__.py
    └── training_pipeline_utils.py  # 训练工具函数
```

## 快速开始

### 1. 安装依赖

确保已安装所有必要的依赖：

```bash
pip install -r requirements.txt
```

### 2. 配置

在 `config.py` 中设置全局配置，包括：

- 数据路径和参数
- 模型路径
- 训练超参数
- 保存路径等

重要配置项：

- `qlib_data_path`: Qlib 数据路径
- `pretrained_tokenizer_path`: 预训练分词模型路径
- `pretrained_predictor_path`: 预训练预测模型路径
- `finetuned_tokenizer_path`: 微调后的分词模型保存路径
- `finetuned_predictor_path`: 微调后的预测模型保存路径

### 3. 运行完整流水线

#### 使用 GPU 训练

```bash
# 使用 torchrun 启动分布式训练
torchrun --standalone --nproc_per_node=NUM_GPUS train_complete_pipeline.py --data-source qlib
```

#### 使用 CPU 训练

```bash
# 使用 CPU 训练
python train_complete_pipeline.py --cpu --data-source qlib
```

#### 命令行参数

- `--cpu`: 使用 CPU 训练
- `--data-source {qlib,sina}`: 选择数据源
- `--skip-data-process`: 跳过数据处理步骤
- `--skip-tokenizer`: 跳过分词模型训练
- `--skip-predictor`: 跳过预测模型训练
- `--fast-mode`: 快速模式，减少训练轮数和批次
- `--config-path PATH`: 指定配置文件路径

### 4. 单独运行各个步骤

#### 数据处理

```python
from sina_data_processor import DataProcessorFactory
from config import Config

config = Config()
processor = DataProcessorFactory.create_processor('qlib', config)
processor.run_pipeline()
```

#### 分词模型训练

GPU 版本:
```bash
torchrun --standalone --nproc_per_node=NUM_GPUS train_tokenizer.py
```

CPU 版本:
```bash
python train_tokenizer_cpu.py
```

#### 预测模型训练

GPU 版本:
```bash
torchrun --standalone --nproc_per_node=NUM_GPUS train_predictor.py
```

CPU 版本:
```bash
python train_predictor_cpu.py
```

## 流水线详细说明

### 1. 数据处理

流水线支持两种数据源：

1. **Qlib 数据源**：使用 Qlib 提供的金融数据
2. **新浪数据源**：从新浪财经获取股票数据

数据处理步骤：

1. 加载原始数据
2. 处理和标准化特征
3. 划分训练集、验证集和测试集
4. 保存处理后的数据集

### 2. 分词模型训练

分词模型负责将原始金融时间序列转换为离散的 token 序列。训练步骤：

1. 加载预训练的分词模型
2. 使用处理后的数据集进行微调
3. 保存最佳模型检查点

### 3. 预测模型训练

预测模型基于分词模型的输出进行预测。训练步骤：

1. 加载预训练的预测模型和微调后的分词模型
2. 使用分词模型对输入数据进行编码
3. 训练预测模型
4. 保存最佳模型检查点

### 4. 预测功能

使用训练好的模型对最新数据进行预测：

1. 加载微调后的分词模型和预测模型
2. 对输入数据进行预处理
3. 使用模型进行预测
4. 返回预测结果

## 高级使用

### 自定义配置

可以通过 JSON 文件提供自定义配置：

```bash
python train_complete_pipeline.py --config-path my_config.json
```

配置文件示例：

```json
{
  "epochs": 20,
  "batch_size": 64,
  "tokenizer_learning_rate": 1e-4,
  "predictor_learning_rate": 2e-5
}
```

### 使用不同的数据源

切换到新浪数据源：

```bash
python train_complete_pipeline.py --data-source sina
```

### 快速测试

使用快速模式测试流水线：

```bash
python train_complete_pipeline.py --fast-mode
```

## 模型架构分析

Kronos 模型由两个主要组件组成：

1. **分词模型 (KronosTokenizer)**：
   - 将连续的金融时间序列转换为离散的 token
   - 基于 VQ-VAE 架构
   - 包含编码器和解码器

2. **预测模型 (Kronos)**：
   - 基于 Transformer 架构
   - 接收分词模型生成的 token 序列
   - 预测未来的价格走势

详细的模型架构分析可以参考 `MODEL_ARCHITECTURE_ANALYSIS.md`。

## 故障排除

### 常见问题

1. **内存不足错误**：
   - 减小 `batch_size`
   - 使用梯度累积 (`accumulation_steps > 1`)

2. **CUDA 相关错误**：
   - 确保 CUDA 版本与 PyTorch 兼容
   - 检查 GPU 内存使用情况

3. **数据加载错误**：
   - 确保数据路径正确
   - 检查数据格式是否符合要求

### 日志

训练过程中的日志会显示在控制台，并且可以通过 Comet.ml（如果启用）查看更详细的指标和图表。

## 参考资料

- [Kronos 模型文档](https://github.com/yourname/kronos)
- [PyTorch 分布式训练指南](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Qlib 文档](https://qlib.readthedocs.io/)