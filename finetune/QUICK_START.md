# Kronos 快速开始指南

## 📦 功能特性

### 智能文件隔离
不同股票数量自动使用**独立的缓存和输出目录**，测试和正式训练互不干扰。

```
测试(10支)   → selected_stocks_10.json   → outputs/sina/local_base_k10/
正式(3000支) → selected_stocks_3000.json → outputs/sina/local_base_k3000/
```

## 🚀 快速使用

### 1. 快速测试（10支股票）
```bash
conda activate kronos
cd /root/zouning/Kronos/finetune

# 第一次运行：自动选择10支最活跃股票
python main.py --top-k-stocks 10

# 后续运行：使用缓存，秒速启动
python main.py --top-k-stocks 10
```

### 2. 正式训练（3000支股票）
```bash
# 第一次运行：自动选择3000支最活跃股票
python main.py --top-k-stocks 3000

# 后续运行：使用缓存
python main.py --top-k-stocks 3000
```

### 3. 更新股票池
```bash
# 强制重新选择股票（不使用缓存）
python main.py --top-k-stocks 3000 --no-stock-cache
```

### 4. CPU优化训练
```bash
# 启用多核优化
python main.py \
  --top-k-stocks 3000 \
  --num-workers 16 \
  --torch-threads 28

# 或使用脚本
bash run_cpu_optimized.sh
```

## 📁 文件结构

```
finetune/
├── selected_stocks_10.json      # 10支股票缓存
├── selected_stocks_3000.json    # 3000支股票缓存
├── outputs/
│   └── sina/
│       ├── local_base_k10/      # 10支股票的输出
│       └── local_base_k3000/    # 3000支股票的输出
└── data/
    └── processed_datasets/       # 处理后的训练数据
```

## ⚡ 参数说明

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--top-k-stocks` | 3000 | 选择TopK活跃股票数量 |
| `--stock-selection-days` | 365 | 评估活跃度的天数 |
| `--no-stock-cache` | False | 强制重新选择股票 |
| `--num-workers` | 0 | DataLoader工作进程数 |
| `--torch-threads` | 0 | PyTorch计算线程数 |

## 💡 最佳实践

1. **测试阶段**：使用 `--top-k-stocks 10` 快速验证流程
2. **正式训练**：使用 `--top-k-stocks 3000` 完整训练
3. **CPU优化**：根据服务器配置设置 `--num-workers` 和 `--torch-threads`
4. **缓存管理**：股票池一旦生成，不会自动更新，除非使用 `--no-stock-cache`

## 🔍 故障排查

### 问题：训练数据不足
```bash
# 检查缓存的股票列表
cat selected_stocks_10.json

# 强制重新选择股票
python main.py --top-k-stocks 10 --no-stock-cache
```

### 问题：测试和正式训练混淆
```
✓ 不会发生！不同stock数量自动隔离：
  - 10支  → selected_stocks_10.json + outputs/.../k10/
  - 3000支 → selected_stocks_3000.json + outputs/.../k3000/
```

## 📅 定时任务

```bash
# 设置每天凌晨2点自动训练
bash setup_daily_training.sh
```

## 📖 更多文档

- `README.md` - 完整文档
- `daily_train.sh` - 定时任务脚本
- `setup_daily_training.sh` - 定时任务配置
