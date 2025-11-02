# Kronos 项目结构

## 📁 目录结构

```
finetune/
├── configs/              # 配置文件
│   ├── custom_predictor_config.json
│   ├── custom_tokenizer_config.json
│   └── kronos_models_config.json
├── data/                 # 数据文件
│   └── stock_code_US.csv
├── docs/                 # 文档
├── logs/                 # 日志
├── outputs/              # 训练输出
│   └── sina/
│       ├── base_k10/     # 10支股票结果
│       └── base_k3000/   # 3000支股票结果
├── utils/                # 工具模块
├── main.py               # 主程序
├── setup_cron.sh         # 定时任务设置
└── daily_train.sh        # 训练脚本
```

## 🔧 核心文件

### Python文件
- `main.py` - 主程序入口
- `optimized_config.py` - 配置管理
- `common_data_processor.py` - 数据处理
- `train_tokenizer.py` - 分词器训练
- `train_predictor.py` - 预测器训练

### Shell脚本
- `setup_cron.sh` - 配置定时任务
- `daily_train.sh` - 每日训练脚本

### 配置文件
- `configs/custom_tokenizer_config.json` - 分词器配置
- `configs/custom_predictor_config.json` - 预测器配置
- `configs/kronos_models_config.json` - 模型路径配置

## 📝 文档文件

- `README.md` - 主文档（导航）
- `QUICK_START.md` - 快速开始指南
- `LOCAL_MODEL_CONFIG.md` - 模型配置详解
- `结果文件位置.md` - 输出文件说明
- `PROJECT_STRUCTURE.md` - 本文档

---

**最后更新**: 2025-11-02
