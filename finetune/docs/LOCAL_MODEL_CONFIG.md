# 本地模型配置说明

## 📁 目录结构

```
finetune/Kronos_models/
├── Kronos-mini/
│   ├── Kronos-mini/           (预测模型)
│   └── Kronos-Tokenizer-2k/   (分词模型)
├── Kronos-small/
│   ├── Kronos-small/          (预测模型)
│   └── Kronos-Tokenizer-base/ (分词模型)
└── Kronos-base/
    ├── Kronos-base/           (预测模型)
    └── Kronos-Tokenizer-base/ (分词模型)
```

## 🔧 配置路径

### 在 `optimized_config.py` 中已配置的本地模型路径：

| 模型版本 | Tokenizer 路径 | Predictor 路径 |
|---------|---------------|---------------|
| `local_mini` | `Kronos_models/Kronos-mini/Kronos-Tokenizer-2k` | `Kronos_models/Kronos-mini/Kronos-mini` |
| `local_small` | `Kronos_models/Kronos-small/Kronos-Tokenizer-base` | `Kronos_models/Kronos-small/Kronos-small` |
| `local_base` | `Kronos_models/Kronos-base/Kronos-Tokenizer-base` | `Kronos_models/Kronos-base/Kronos-base` |

## 🚀 使用方法

### 方法 1: 命令行参数（推荐）

#### 使用本地 mini 模型
```bash
python main.py \
  --model-version local_mini \
  --data-source sina \
  --model-source local
```

#### 使用本地 base 模型（默认）
```bash
python main.py \
  --model-version base \
  --data-source sina \
  --model-source local
```

#### 使用本地 small 模型
```bash
python main.py \
  --model-version local_small \
  --data-source sina \
  --model-source local
```

### 方法 2: 自动检测（智能模式）

当使用 `local_*` 模型版本时，系统会自动设置 `model_source='local'`：

```bash
# model_source 会自动设置为 'local'
python main.py --model-version local_base --data-source sina
```

### 方法 3: 配置文件

创建 `local_config.json`：

```json
{
  "model_version": "local_base",
  "model_source": "local",
  "data_source": "sina",
  "epochs": 30,
  "batch_size": 50,
  "early_stopping_patience": 6
}
```

运行：
```bash
python main.py --config-path local_config.json
```

### 方法 4: 交互式选择

```bash
python main.py
```

然后在交互界面选择：
- 数据源: `a` (sina)
- 模型版本: `f` (local_base) 或 `d` (local_mini) 或 `e` (local_small)

## 📊 模型版本对比

| 版本 | 参数量 | 训练速度 | 预测精度 | 推荐场景 |
|------|--------|---------|---------|---------|
| `local_mini` | 最小 | 最快 | 较低 | 快速测试、资源受限 |
| `local_small` | 中等 | 中等 | 中等 | 平衡性能和速度 |
| `local_base` | 最大 | 较慢 | 最高 | 生产环境、追求精度 |

## ⚙️ 完整命令示例

### GPU 训练（推荐）
```bash
python main.py \
  --model-version local_base \
  --data-source sina \
  --model-source local \
  --early-stopping-patience 6
```

### CPU 训练
```bash
python main.py \
  --model-version local_mini \
  --data-source sina \
  --model-source local \
  --cpu \
  --early-stopping-patience 8
```

### 强制重新下载数据
```bash
python main.py \
  --model-version local_base \
  --data-source sina \
  --force-download
```

## 🔍 路径解析流程

1. **配置初始化**: `OptimizedConfig.__init__()` 读取 `model_version` 参数
2. **路径设置**: `_init_model_config()` 从 `model_versions` 字典获取对应路径
3. **自动检测**: 如果 `model_version` 以 `local_` 开头，自动设置 `model_source='local'`
4. **模型加载**: `load_tokenizer()` 和 `load_predictor()` 使用 `model_source='local'` 加载模型
5. **本地验证**: `model_loader.py` 中的 `_load_from_local()` 验证路径存在性

## ⚠️ 注意事项

1. **路径相对性**: 所有路径都是相对于 `finetune/` 目录的相对路径
2. **模型文件**: 确保本地模型目录中包含必要的文件：
   - `config.json`
   - `model.safetensors`
   - `configuration.json`
3. **权限**: 确保对模型文件有读取权限
4. **存储空间**: 确保有足够的磁盘空间存储训练结果

## 🐛 故障排查

### 问题 1: 找不到本地模型
```
FileNotFoundError: 本地模型路径不存在
```

**解决方案**:
```bash
# 检查路径是否存在
ls -la finetune/Kronos_models/Kronos-base/Kronos-base/

# 确认模型文件
ls -la finetune/Kronos_models/Kronos-base/Kronos-base/model.safetensors
```

### 问题 2: 模型加载失败
```
Error loading model from local
```

**解决方案**:
1. 检查 `config.json` 是否存在且格式正确
2. 检查 `model.safetensors` 文件是否完整
3. 尝试使用较小的模型版本（如 `local_mini`）

### 问题 3: 路径配置错误
```
ValueError: 不支持的模型版本
```

**解决方案**:
使用正确的版本名称：`local_mini`, `local_small`, `local_base`

## 📝 配置参数总结

| 参数 | 说明 | 可选值 | 默认值 |
|------|------|-------|--------|
| `--model-version` | 模型版本 | `mini`, `small`, `base`, `local_mini`, `local_small`, `local_base`, `customer` | `base` |
| `--model-source` | 模型来源 | `auto`, `huggingface`, `modelscope`, `local` | `auto` |
| `--data-source` | 数据源 | `qlib`, `sina` | `sina` |
| `--cpu` | 使用CPU训练 | - | `False` |
| `--early-stopping-patience` | 提前终止耐心值 | 整数 | `8` |
| `--force-download` | 强制重新下载数据 | - | `False` |

## 🎯 推荐配置

### 日常训练（快速迭代）
```bash
python main.py --model-version local_mini --data-source sina
```

### 生产环境（追求精度）
```bash
python main.py --model-version local_base --data-source sina --early-stopping-patience 6
```

### 测试环境（CPU）
```bash
python main.py --model-version local_mini --data-source sina --cpu
```

