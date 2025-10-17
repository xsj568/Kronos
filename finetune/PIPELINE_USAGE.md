# Kronos 训练流水线使用指南

本文档提供了 Kronos 训练流水线的详细使用说明，包括安装、配置、运行和常见问题解决方法。

## 安装与环境设置

### 系统要求

- Python 3.8+
- CUDA 11.3+ (用于 GPU 训练)
- 8GB+ RAM (CPU 训练)
- 16GB+ GPU 内存 (GPU 训练)

### 安装步骤

1. **克隆仓库**

```bash
git clone https://github.com/yourusername/Kronos.git
cd Kronos
```

2. **安装依赖**

```bash
pip install -r requirements.txt
```

3. **设置数据目录**

对于 Qlib 数据：

```bash
mkdir -p ~/.qlib/qlib_data/cn_data
# 下载并解压 Qlib 数据到上述目录
```

## 配置说明

### 配置文件

主要配置文件位于 `finetune/config.py`，包含以下关键配置项：

#### 数据配置

```python
# 数据路径与特征
self.qlib_data_path = "~/.qlib/qlib_data/cn_data"  # Qlib 数据路径
self.instrument = 'csi300'  # 使用的股票池
self.feature_list = ['open', 'high', 'low', 'close', 'vol', 'amt']  # 使用的特征
self.time_feature_list = ['minute', 'hour', 'weekday', 'day', 'month']  # 时间特征

# 时间范围
self.dataset_begin_time = "2011-01-01"  # 数据开始时间
self.dataset_end_time = '2025-06-05'  # 数据结束时间
```

#### 训练配置

```python
# 训练超参数
self.epochs = 30  # 训练轮数
self.batch_size = 50  # 每个 GPU 的批次大小
self.tokenizer_learning_rate = 2e-4  # 分词模型学习率
self.predictor_learning_rate = 4e-5  # 预测模型学习率
```

#### 模型路径配置

```python
# 预训练模型路径
self.pretrained_tokenizer_path = "path/to/your/Kronos-Tokenizer-base"
self.pretrained_predictor_path = "path/to/your/Kronos-small"

# 微调后模型保存路径
self.save_path = "./outputs/models"
self.tokenizer_save_folder_name = 'finetune_tokenizer_demo'
self.predictor_save_folder_name = 'finetune_predictor_demo'
```

### 自定义配置

可以通过 JSON 文件提供自定义配置：

```json
{
  "epochs": 20,
  "batch_size": 64,
  "tokenizer_learning_rate": 1e-4,
  "predictor_learning_rate": 2e-5,
  "pretrained_tokenizer_path": "/path/to/custom/tokenizer",
  "pretrained_predictor_path": "/path/to/custom/predictor"
}
```

使用自定义配置：

```bash
python train_complete_pipeline.py --config-path my_config.json
```

## 运行流水线

### 完整流水线

#### GPU 训练

```bash
# 单 GPU 训练
torchrun --standalone --nproc_per_node=1 train_complete_pipeline.py

# 多 GPU 训练
torchrun --standalone --nproc_per_node=4 train_complete_pipeline.py
```

#### CPU 训练

```bash
python train_complete_pipeline.py --cpu
```

### 命令行参数

```
--cpu                   使用 CPU 训练
--data-source {qlib,sina}  选择数据源
--skip-data-process     跳过数据处理步骤
--skip-tokenizer        跳过分词模型训练
--skip-predictor        跳过预测模型训练
--fast-mode             快速模式，减少训练轮数和批次
--config-path PATH      指定配置文件路径
```

### 单独运行各个步骤

#### 1. 数据处理

```bash
# 使用 Qlib 数据
python -c "from sina_data_processor import DataProcessorFactory; from config import Config; processor = DataProcessorFactory.create_processor('qlib', Config()); processor.run_pipeline()"

# 使用新浪数据
python -c "from sina_data_processor import DataProcessorFactory; from config import Config; processor = DataProcessorFactory.create_processor('sina', Config()); processor.run_pipeline()"
```

#### 2. 分词模型训练

```bash
# GPU 训练
torchrun --standalone --nproc_per_node=NUM_GPUS train_tokenizer.py

# CPU 训练
python train_tokenizer_cpu.py
```

#### 3. 预测模型训练

```bash
# GPU 训练
torchrun --standalone --nproc_per_node=NUM_GPUS train_predictor.py

# CPU 训练
python train_predictor_cpu.py
```

## 高级用法

### 自定义数据源

要添加新的数据源，需要：

1. 在 `sina_data_processor.py` 中创建新的数据处理器类，继承 `BaseDataProcessor`
2. 实现 `download_data()` 和 `process_raw_data()` 方法
3. 在 `DataProcessorFactory` 中注册新的数据处理器

示例：

```python
class CustomDataProcessor(BaseDataProcessor):
    def __init__(self, config):
        super().__init__(config)
        # 自定义初始化
        
    def download_data(self):
        # 实现数据下载逻辑
        pass
        
    def process_raw_data(self):
        # 实现数据处理逻辑
        pass

# 在工厂中注册
@staticmethod
def create_processor(data_source_type: str, config, **kwargs):
    if data_source_type.lower() == 'custom':
        return CustomDataProcessor(config)
    # 其他数据源...
```

### 自定义训练循环

要自定义训练循环，可以修改 `main.py` 中的 `train_tokenizer` 和 `train_predictor` 方法：

```python
def train_tokenizer(self):
    # 自定义训练逻辑
    pass

def train_predictor(self):
    # 自定义训练逻辑
    pass
```

### 分布式训练配置

高级分布式训练配置：

```bash
# 指定 GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_complete_pipeline.py

# 多节点训练
torchrun --nnodes=2 --node_rank=0 --master_addr="192.168.1.1" --master_port=29500 --nproc_per_node=4 train_complete_pipeline.py
```

## 输出与结果

### 目录结构

训练后的输出目录结构：

```
outputs/
├── models/
│   ├── finetune_tokenizer_demo/
│   │   ├── checkpoints/
│   │   │   └── best_model/
│   │   └── summary.json
│   └── finetune_predictor_demo/
│       ├── checkpoints/
│       │   └── best_model/
│       └── summary.json
├── backtest_results/
│   └── finetune_backtest_demo/
│       └── predictions.pkl
└── pipeline_config.json
```

### 日志与指标

训练过程中的日志包含以下指标：

- 训练损失
- 验证损失
- 学习率
- 训练时间
- 最佳模型性能

## 故障排除

### 常见错误与解决方案

#### 1. CUDA 内存不足

错误信息：`CUDA out of memory`

解决方案：
- 减小 `batch_size`
- 增加 `accumulation_steps`
- 使用更小的模型

#### 2. 数据加载错误

错误信息：`FileNotFoundError: [Errno 2] No such file or directory`

解决方案：
- 检查数据路径是否正确
- 确保数据已下载并解压

#### 3. 分布式训练错误

错误信息：`RuntimeError: NCCL error in: ../torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:XXX`

解决方案：
- 检查 CUDA 版本与 PyTorch 兼容性
- 确保所有 GPU 可见且工作正常
- 尝试使用较新版本的 NCCL

#### 4. 模型加载错误

错误信息：`Error(s) in loading state_dict for KronosTokenizer`

解决方案：
- 确保预训练模型路径正确
- 检查模型版本兼容性

### 性能优化建议

1. **GPU 训练优化**：
   - 使用混合精度训练 (`torch.cuda.amp`)
   - 优化数据加载 (增加 `num_workers`, 使用 `pin_memory=True`)
   - 使用梯度累积处理大批次

2. **CPU 训练优化**：
   - 减小模型大小
   - 使用 `fast_mode` 快速测试
   - 考虑使用 Intel MKL 优化版本的 PyTorch

## 附录

### 环境变量

可以通过环境变量控制某些行为：

```bash
# 设置 CUDA 可见设备
export CUDA_VISIBLE_DEVICES=0,1

# 设置 PyTorch 线程数
export OMP_NUM_THREADS=4

# 启用 PyTorch 调试
export TORCH_DEBUG=1
```

### 依赖版本兼容性

| 依赖项 | 最低版本 | 推荐版本 | 说明 |
|--------|----------|----------|------|
| PyTorch | 1.8.0 | 1.13.0 | 需要支持 CUDA 11.3+ |
| CUDA | 11.0 | 11.6 | 与 PyTorch 版本匹配 |
| Qlib | 0.8.0 | 0.9.0 | 用于数据处理 |

### 参考资源

- [PyTorch 分布式训练文档](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Qlib 文档](https://qlib.readthedocs.io/)
- [Kronos 模型论文](https://arxiv.org/abs/xxxx.xxxxx)