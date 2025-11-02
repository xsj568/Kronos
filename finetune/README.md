# Kronos 训练使用指南

## 🚀 快速开始

```bash
cd /root/zouning/Kronos/finetune

# 激活环境
conda activate kronos

# CPU优化训练（使用本地模型，3000支股票）
./run_cpu_optimized.sh

# 快速测试（10支股票）
python main.py --top-k-stocks 10
```

## ⚙️ 默认配置（无需参数）

```bash
# 直接运行，使用所有默认配置
python main.py
```

**默认参数：**
- 数据源：sina
- 模型：base (Kronos-base模型，默认从本地加载)
- 股票数量：3000
- CPU优化：16 workers, 28 threads
- 设备：CPU

## 📊 核心功能

### 1. 股票选择（TopK）

**智能隔离：** 不同股票数量自动使用不同的缓存和输出目录，测试和正式训练互不干扰。

**使用缓存（快速）：**
```bash
python main.py --top-k-stocks 3000
# 使用: selected_stocks_3000.json
# 输出: outputs/sina/base_k3000/
```

**重新选择：**
```bash
python main.py --top-k-stocks 3000 --no-stock-cache
```

**测试配置（10支股票）：**
```bash
python main.py --top-k-stocks 10
# 使用: selected_stocks_10.json  
# 输出: outputs/sina/base_k10/
# ✓ 不会影响3000支股票的训练结果
```

**参数说明：**
- `--top-k-stocks 3000` - 股票数量（默认3000）
- `--stock-selection-days 365` - 评估天数（默认365天）
- `--no-stock-cache` - 强制重新选择

**文件命名规则：**
- 缓存文件：`selected_stocks_{数量}.json`
- 输出目录：`outputs/sina/{模型}_{k数量}/`

### 2. CPU多核优化

**服务器配置：** 56物理核心 / 112逻辑CPU

**快速启动：**
```bash
bash run_cpu_optimized.sh
```

**自定义参数：**
```bash
python main.py \
    --cpu \
    --num-workers 16 \
    --torch-threads 28
```

### 3. 定时任务

**一键设置（推荐）：**
```bash
bash setup_daily_training.sh
```

**手动测试：**
```bash
bash daily_train.sh
```

**特性：**
- ✓ 自动激活conda环境
- ✓ 智能进程检测（避免重复运行）
- ✓ 3次重试机制
- ✓ 自动清理30天前的日志
- ✓ 完整的输出验证

**管理命令：**
```bash
crontab -l                                    # 查看定时任务
tail -f logs/training_$(date +%Y%m%d)*.log   # 查看当天日志
ls -lht logs/training_*.log | head           # 列出最近日志
```

---

## 📝 常用参数

```bash
python main.py \
    --data-source sina \              # 数据源（sina/qlib）
    --model-version base \            # 模型版本（默认base，从本地加载）
    --top-k-stocks 3000 \            # 股票数量
    --num-workers 16 \               # DataLoader进程数
    --torch-threads 28 \             # PyTorch线程数
    --early-stopping-patience 3      # 提前停止
```

---

## ❓ 常见问题

**查看训练进度：**
```bash
# 日志文件格式：training_YYYYMMDD_数据源_模型版本_k股票数.log
# 示例：training_20251101_sina_base_k3000.log
tail -f logs/training_$(date +%Y%m%d)*.log

# 或者列出所有日志文件
ls -lht logs/training_*.log | head

# 监控CPU使用率
htop  # CPU使用率应接近2800%
```

**更新股票池：**
```bash
rm selected_stocks.json  # 删除缓存
```

**停止训练：**
```bash
pkill -f "python main.py"
```

**内存不足：**
```bash
# 减少worker和股票数
python main.py --num-workers 8 --top-k-stocks 1000
```

---

## 📁 核心文件

- `main.py` - 统一训练入口
- `daily_train.sh` - 每日训练脚本
- `setup_daily_training.sh` - 设置定时任务
- `run_cpu_optimized.sh` - CPU优化训练
- `selected_stocks.json` - 股票缓存（自动生成）
- `logs/` - 训练日志目录

---

## 🎯 推荐配置

| 场景 | 命令 |
|------|------|
| 默认训练 | `python main.py` |
| 快速测试 | `python main.py --top-k-stocks 10` |
| CPU优化 | `./run_cpu_optimized.sh` |
| 最大性能 | `./run_cpu_optimized_with_compile.sh` |
| 定时任务 | `./setup_daily_training.sh` |
