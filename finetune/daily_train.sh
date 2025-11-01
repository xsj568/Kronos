#!/bin/bash

# Kronos每日自动训练脚本
# 用于cron定时任务

set -e  # 遇到错误立即退出

echo "=========================================="
echo "Kronos 每日自动训练"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# 切换到脚本所在目录
cd "$(dirname "$0")"

# 设置环境变量
export OMP_NUM_THREADS=28
export MKL_NUM_THREADS=28
export OPENBLAS_NUM_THREADS=28
export VECLIB_MAXIMUM_THREADS=28
export NUMEXPR_NUM_THREADS=28
export CUDA_VISIBLE_DEVICES=""

# 配置参数
DATA_SOURCE="sina"
MODEL_VERSION="base"
TOP_K_STOCKS=3000
NUM_WORKERS=16
TORCH_THREADS=28
EARLY_STOPPING_PATIENCE=3
MAX_RETRIES=3

# 日志目录
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

# 训练日志文件（会由Python程序自动创建，包含数据源、模型版本和股票数量）
# 实际文件名格式：training_20251101_sina_base_k3000.log
TRAIN_LOG="$LOG_DIR/training_$(date +%Y%m%d)_${DATA_SOURCE}_${MODEL_VERSION}_k${TOP_K_STOCKS}.log"

echo "配置参数："
echo "  - 数据源: $DATA_SOURCE"
echo "  - 模型版本: $MODEL_VERSION"
echo "  - 股票数量: $TOP_K_STOCKS"
echo "  - Worker数: $NUM_WORKERS"
echo "  - 线程数: $TORCH_THREADS"
echo "  - 最大重试: $MAX_RETRIES"
echo ""

# 检查上一个训练进程（定时任务自动处理）
if pgrep -f "python main.py" > /dev/null; then
    echo "警告：检测到正在运行的训练进程"
    echo "进程列表："
    ps aux | grep "python main.py" | grep -v grep
    echo ""
    
    # 如果是非交互式环境（如cron），自动跳过
    if [ ! -t 0 ]; then
        echo "非交互式环境，跳过本次训练"
        exit 0
    fi
    
    # 交互式环境，询问用户
    read -p "是否停止旧进程并继续？(y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pkill -f "python main.py"
        sleep 5
        echo "已停止旧进程"
    else
        echo "取消训练"
        exit 1
    fi
fi

# 训练函数
train_model() {
    echo "开始训练..."
    python main.py \
        --cpu \
        --data-source "$DATA_SOURCE" \
        --model-version "$MODEL_VERSION" \
        --top-k-stocks "$TOP_K_STOCKS" \
        --num-workers "$NUM_WORKERS" \
        --torch-threads "$TORCH_THREADS" \
        --early-stopping-patience "$EARLY_STOPPING_PATIENCE" \
        2>&1 | tee -a "$TRAIN_LOG"
    
    return ${PIPESTATUS[0]}
}

# 重试机制
attempt=1
success=false

while [ $attempt -le $MAX_RETRIES ]; do
    echo ""
    echo "=========================================="
    echo "第 $attempt/$MAX_RETRIES 次尝试"
    echo "=========================================="
    
    if train_model; then
        success=true
        echo ""
        echo "✓ 训练成功完成"
        break
    else
        exit_code=$?
        echo ""
        echo "✗ 训练失败 (退出码: $exit_code)"
        
        if [ $attempt -lt $MAX_RETRIES ]; then
            wait_time=$((attempt * 60))
            echo "等待 $wait_time 秒后重试..."
            sleep $wait_time
        fi
    fi
    
    attempt=$((attempt + 1))
done

# 训练结束
echo ""
echo "=========================================="
echo "训练结束"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"

if [ "$success" = true ]; then
    echo "状态: ✓ 成功"
    
    # 检查输出文件（考虑stock数量后缀）
    echo ""
    echo "输出文件检查："
    
    OUTPUT_DIR="./outputs/$DATA_SOURCE/${MODEL_VERSION}_k${TOP_K_STOCKS}"
    
    if [ -d "$OUTPUT_DIR" ]; then
        echo "  ✓ 输出目录存在: $OUTPUT_DIR"
        ls -lh "$OUTPUT_DIR/" | head -10
    else
        echo "  ✗ 输出目录不存在: $OUTPUT_DIR"
    fi
    
    # 检查模型文件
    if [ -d "$OUTPUT_DIR/finetune_tokenizer/best_model" ]; then
        echo "  ✓ 分词模型已保存"
    fi
    
    if [ -d "$OUTPUT_DIR/finetune_predictor/best_model" ]; then
        echo "  ✓ 预测模型已保存"
    fi
    
    # 显示预测文件
    if [ -f "$OUTPUT_DIR/predictions_master.xlsx" ]; then
        echo "  ✓ 预测文件已生成"
        ls -lh "$OUTPUT_DIR/predictions_master.xlsx"
    fi
    
    exit_code=0
else
    echo "状态: ✗ 失败（已重试 $MAX_RETRIES 次）"
    exit_code=1
fi

echo "=========================================="

# 清理旧日志（保留最近30天）
echo ""
echo "清理旧日志文件..."
find "$LOG_DIR" -name "training_*.log" -type f -mtime +30 -delete 2>/dev/null || true
find "$LOG_DIR" -name "cron_*.log" -type f -mtime +30 -delete 2>/dev/null || true
echo "  保留的最近日志文件："
ls -lht "$LOG_DIR"/training_*.log 2>/dev/null | head -5 || echo "  （无日志文件）"
echo "✓ 日志清理完成"

exit $exit_code

