# DDP 多GPU训练使用指南

## 概述

系统现已支持两种多GPU训练模式：

1. **DDP (DistributedDataParallel)** - **推荐，性能最佳**
2. **DataParallel** - 备选方案（兼容性）

## 性能对比

| 模式 | 相对速度 | GPU利用率 | 启动方式 |
|------|---------|----------|---------|
| DDP | **最快（基准）** | 均衡 | torchrun |
| DataParallel | ~60-70% | GPU0负载重 | python |
| 单GPU | ~25-30% | 单卡 | python |

## 快速开始

### 1. 自动模式（推荐）

脚本会自动检测GPU并选择最佳训练方式：

```bash
# 使用定时任务脚本
bash daily_train.sh
```

**配置参数**（在 `daily_train.sh` 中修改）：
```bash
NUM_GPUS=0    # 0=使用所有可用GPU（自动DDP）
              # 1=单GPU
              # 2+=指定GPU数量
```

### 2. 手动启动

#### 多GPU训练（DDP模式）

```bash
# 使用所有GPU
torchrun --standalone --nproc_per_node=gpu main.py

# 使用2个GPU
torchrun --standalone --nproc_per_node=2 main.py

# 使用4个GPU + 其他参数
torchrun --standalone --nproc_per_node=4 main.py \
    --data-source sina \
    --model-version base \
    --top-k-stocks 1000
```

#### 单GPU训练

```bash
python main.py --data-source sina --model-version base
```

#### CPU训练

```bash
python main.py --cpu --data-source sina --model-version base
```

## 工作原理

### 自动选择逻辑

```
检测GPU数量
    ↓
  > 1个GPU?
    ↓
  是 → 检测torchrun环境变量?
         ↓
       存在 → 使用DDP（最快）✓
         ↓
       不存在 → 使用DataParallel（兼容）
    ↓
  否 → 单GPU或CPU训练
```

### 关键环境变量

DDP需要这些环境变量（由torchrun自动设置）：
- `RANK` - 全局进程编号
- `WORLD_SIZE` - 总进程数
- `LOCAL_RANK` - 本地GPU编号

## 配置说明

### daily_train.sh 配置

```bash
NUM_GPUS=0              # GPU数量（0=自动检测所有）
NUM_WORKERS=16          # DataLoader工作进程数
TORCH_THREADS=28        # PyTorch计算线程数（CPU模式）
TOP_K_STOCKS=1000       # 股票数量
```

### 命令行参数

```bash
--num-gpus N            # 指定GPU数量（用于脚本）
--min-gpu-memory 5.0    # 最小GPU空闲内存（GB）
--no-multi-gpu          # 禁用多GPU，只用最佳单GPU
--cpu                   # 强制使用CPU
```

## 性能优化建议

### GPU数量选择

| GPU数量 | 推荐场景 | 预计训练速度 |
|--------|---------|------------|
| 1个 | 小规模测试 | 基准 |
| 2个 | 中等规模 | ~1.8x |
| 4个 | 大规模训练 | ~3.5x |
| 8个 | 超大规模 | ~6-7x |

### 内存优化

```bash
# 如果遇到GPU内存不足
--min-gpu-memory 8.0    # 只使用空闲内存>8GB的GPU
--top-k-stocks 500      # 减少股票数量
--batch-size 32         # 减小批次大小（需修改config）
```

## 故障排查

### 问题1：torchrun命令不存在

**解决**：
```bash
# 检查PyTorch版本
python -c "import torch; print(torch.__version__)"

# 如果版本 < 1.9，升级PyTorch
pip install --upgrade torch
```

### 问题2：DDP初始化失败

**日志**：`RuntimeError: NCCL error`

**解决**：
```bash
# 设置NCCL环境变量
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # 禁用InfiniBand

# 或者回退到DataParallel
python main.py  # 不使用torchrun
```

### 问题3：GPU内存不足

**日志**：`CUDA out of memory`

**解决**：
```bash
# 方案1：减少GPU数量
NUM_GPUS=2  # 在daily_train.sh中设置

# 方案2：减少股票数量
--top-k-stocks 500

# 方案3：只使用空闲内存充足的GPU
--min-gpu-memory 10.0
```

### 问题4：多进程死锁

**日志**：训练卡住不动

**解决**：
```bash
# 检查是否有残留进程
ps aux | grep "main.py"

# 清理残留进程
pkill -9 -f "main.py"

# 重新启动
bash daily_train.sh
```

## 监控训练

### 实时查看日志

```bash
# 查看最新训练日志
tail -f logs/training_$(date +%Y%m%d)_sina_base_k1000.log

# 查看GPU使用情况
watch -n 1 nvidia-smi
```

### 性能指标

关注日志中的这些指标：
- **每轮训练时间**：应该比单GPU快2-4倍（取决于GPU数量）
- **GPU利用率**：所有GPU应该接近100%
- **内存使用**：每个GPU应该均衡

## 最佳实践

1. ✅ **优先使用DDP**：性能最佳
2. ✅ **使用定时任务脚本**：自动选择最佳模式
3. ✅ **监控GPU内存**：确保不会OOM
4. ✅ **定期清理进程**：避免资源泄漏
5. ✅ **保存检查点**：防止训练中断

## 示例场景

### 场景1：日常训练（推荐）

```bash
# 使用定时任务脚本，自动选择最佳模式
bash daily_train.sh
```

### 场景2：快速测试

```bash
# 100支股票，单GPU
python main.py --top-k-stocks 100 --no-multi-gpu
```

### 场景3：完整训练

```bash
# 1000支股票，使用所有GPU
torchrun --standalone --nproc_per_node=gpu main.py \
    --top-k-stocks 1000 \
    --early-stopping-patience 3
```

### 场景4：指定GPU

```bash
# 只使用GPU 0和1
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 main.py
```

## 更新日志

- **2025-11-05**：
  - ✨ 新增DDP支持，性能提升2-4倍
  - ✨ 自动检测并选择最佳训练模式
  - ✨ daily_train.sh自动支持DDP
  - 📝 股票数量从3000减少到1000

---

**提示**：如有问题，请查看日志文件中的详细错误信息。

