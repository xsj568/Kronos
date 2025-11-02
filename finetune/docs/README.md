# Kronos 微调训练文档

## 📚 文档导航

- **[快速开始](QUICK_START.md)** - 快速上手指南（⭐推荐首先阅读）
- **[项目结构](PROJECT_STRUCTURE.md)** - 项目整理记录
- **[模型配置](LOCAL_MODEL_CONFIG.md)** - 本地模型配置详解
- **[输出文件](结果文件位置.md)** - 训练结果和输出文件说明

---

## 🚀 一分钟快速开始

```bash
# 1. 激活环境
conda activate kronos

# 2. 快速测试（10支股票）
python main.py --top-k-stocks 10

# 3. 正式训练（3000支股票）
python main.py --top-k-stocks 3000

# 4. 定时任务（每天自动训练）
bash setup_cron.sh
```

详细使用请查看 **[快速开始指南](QUICK_START.md)**

---

## 📊 训练流程

完整的训练流程包括：

1. **数据获取** - 从新浪财经获取股票数据
2. **股票筛选** - 自动选择最活跃的Top-K股票
3. **Tokenizer训练** - 训练分词模型（编码价格序列）
4. **Predictor训练** - 训练预测模型（预测未来价格）
5. **生成预测** - 对所有股票生成未来预测

**训练输出位置**：`outputs/sina/base_k3000/predictions_master.xlsx`

详细说明请查看 **[输出文件说明](结果文件位置.md)**

---

## ⚙️ 主要参数

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--top-k-stocks` | 3000 | 股票数量 |
| `--model-version` | base | 模型版本（mini/small/base） |
| `--data-source` | sina | 数据源（sina/yfinance） |
| `--num-workers` | 16 | CPU核心数 |
| `--cpu` | - | 强制使用CPU |

完整参数说明请查看 **[快速开始指南](QUICK_START.md)**

---

## 🐛 常见问题

**训练失败？**
```bash
# 查看日志
tail -f logs/training_*.log

# 减少股票数量
python main.py --top-k-stocks 100
```

**找不到结果？**
```bash
# 检查输出目录
ls -lh outputs/sina/base_k3000/predictions_master.xlsx
```

更多故障排查请查看 **[快速开始指南](QUICK_START.md)**

---

**版本**: 2.1  
**更新时间**: 2025-11-02
