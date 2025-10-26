# Kronos 每日训练使用指南

## 🚀 快速开始

### 方式1：设置定时任务（推荐）

```bash
cd /root/Kronos/finetune
./setup_cron_optimized.sh
```

按提示操作：
- 选择 `1` 添加/更新定时任务
- 输入执行时间（例如：凌晨2点 → 输入 `2` 和 `0`）
- 定时任务将自动每天运行

### 方式2：手动执行训练

```bash
cd /root/Kronos/finetune
./daily_train_optimized.sh
```

或直接运行Python脚本：

```bash
cd /root/Kronos/finetune
python main.py --data-source sina --model-version base
```

## 📋 训练流程说明

完整流程包括6个步骤：

1. **数据抓取** - 从新浪财经抓取10000支美股数据
2. **基于base模型微调** - 加载预训练模型进行训练
3. **模型评估** - 在测试集上评估base/历史最佳/当前模型
4. **选择最佳模型** - 自动选择损失最小的模型
5. **更新历史模型** - 保存最佳模型到历史目录
6. **预测并保存** - 预测下一个工作日涨跌幅，追加到主Excel文件

## 📂 输出文件位置

训练完成后，输出文件在以下位置：

```
finetune/
├── outputs/sina/base/
│   ├── predictions_master.xlsx      # 【主预测文件】每天增量更新
│   ├── predictions_master.csv       # CSV备份
│   ├── predictions/
│   │   └── prediction_YYYYMMDD_HHMMSS.xlsx  # 详细预测（10天）
│   ├── finetune_tokenizer/
│   │   ├── best_model/              # 当前最佳tokenizer
│   │   └── training_summary.json
│   └── finetune_predictor/
│       ├── best_model/              # 当前最佳predictor
│       └── training_summary.json
│
├── model_history/sina/base/
│   ├── best_tokenizer/              # 历史最佳tokenizer
│   └── best_predictor/              # 历史最佳predictor
│
└── ../logs/
    ├── daily_train_YYYYMMDD.log     # 每日训练日志
    ├── training_YYYYMMDD.log        # 详细训练日志
    └── cron.log                     # 定时任务日志
```

## 📊 查看结果

### 查看主预测文件（推荐）
```bash
# Excel文件路径
finetune/outputs/sina/base/predictions_master.xlsx
```

这个文件包含：
- **预测历史**：所有历史预测记录
- **日期摘要**：按日期统计的预测概览
- **涨跌幅排行榜**：按收盘价涨跌幅排序
- **各特征预测**：开盘价、最高价、最低价、收盘价、成交量预测

### 查看训练日志
```bash
# 查看最新训练日志
tail -f ../logs/daily_train_$(date +%Y%m%d).log

# 查看详细训练日志
tail -f ../logs/training_$(date +%Y%m%d).log
```

### 查看定时任务
```bash
# 查看当前定时任务
crontab -l

# 查看定时任务日志
tail -f ../logs/cron.log
```

## ⚙️ 配置参数

### 修改训练参数

编辑 `daily_train_optimized.sh`：

```bash
# 训练配置
DATA_SOURCE="sina"              # 数据源：sina（新浪美股）或qlib
MODEL_VERSION="base"            # 模型版本：mini/small/base
EARLY_STOPPING_PATIENCE=3       # 提前停止耐心值
USE_GPU="--cpu"                 # GPU使用：空字符串表示使用GPU，"--cpu"表示使用CPU
```

### 修改股票数量

编辑 `optimized_config.py` 第122行：

```python
self.max_sina_symbols = 10000  # 修改为想要的股票数量
```

## 🔧 管理定时任务

### 添加定时任务
```bash
cd /root/Kronos/finetune
./setup_cron_optimized.sh
# 选择 1
```

### 删除定时任务
```bash
cd /root/Kronos/finetune
./setup_cron_optimized.sh
# 选择 2
```

### 查看定时任务
```bash
cd /root/Kronos/finetune
./setup_cron_optimized.sh
# 选择 3
```

## 🐛 故障排查

### 训练失败
1. 查看日志：`tail -100 ../logs/daily_train_$(date +%Y%m%d).log`
2. 检查Python环境：`python --version`
3. 检查依赖包：`pip list | grep -E "torch|pandas|numpy"`

### 定时任务不执行
1. 检查crontab：`crontab -l`
2. 查看cron日志：`tail -50 ../logs/cron.log`
3. 检查系统cron服务：`systemctl status cron` 或 `service cron status`

### 预测文件未生成
1. 检查模型是否训练成功
2. 查看训练日志中的预测部分
3. 确认输出目录权限正确

## 📝 重要提示

1. **首次运行**：首次运行会下载base模型，需要网络连接
2. **磁盘空间**：确保有足够磁盘空间（建议至少50GB）
3. **训练时间**：完整训练可能需要数小时，取决于硬件配置
4. **增量更新**：predictions_master.xlsx 会自动追加，不会覆盖历史数据
5. **模型更新**：每次训练都会自动选择最佳模型并更新

## 🎯 核心逻辑

```
每日训练流程：
1. 抓取10000支美股最新数据
2. 基于base模型微调
3. 评估三个模型（base/历史最佳/当前训练）
4. 选择测试集损失最小的模型
5. 更新为新的历史最佳模型
6. 使用最佳模型预测下一个工作日
7. 结果追加到 predictions_master.xlsx
```

## 📞 获取帮助

遇到问题时：
1. 查看日志文件
2. 检查配置参数
3. 确认环境依赖
4. 查看代码注释

---

**最后更新**: 2025-10-26

