#!/bin/bash
################################################################################
# Kronos 每日自动训练脚本
# 
# 功能说明：
#   执行完整的Kronos模型训练流程，包括数据获取、模型训练、预测生成
#   支持自动重试机制、进程管理、日志记录等功能
#   适用于cron定时任务或手动执行
#
# 使用方法：
#   1. 定时任务自动运行（推荐）:
#      先使用 setup_cron.sh 配置定时任务，cron会自动调用此脚本
#   
#   2. 手动运行:
#      bash daily_train.sh
#
# 主要功能：
#   - 自动激活conda环境（kronos）
#   - 自动杀掉正在运行的旧训练进程（避免重复训练）
#   - 支持失败自动重试（默认最多3次）
#   - 完整的日志记录（训练日志 + cron日志）
#   - 训练完成后自动清理旧日志（保留30天）
#   - 输出文件检查和验证
#
# 训练配置（可在脚本中修改）：
#   DATA_SOURCE="sina"              # 数据源: sina, yfinance
#   MODEL_VERSION="base"            # 模型版本: mini, small, base
#   TOP_K_STOCKS=3000              # 训练股票数量
#   NUM_WORKERS=16                 # DataLoader工作进程数
#   TORCH_THREADS=28               # PyTorch计算线程数
#   EARLY_STOPPING_PATIENCE=3      # 提前停止的耐心值
#   MAX_RETRIES=3                  # 最大重试次数
#
# 环境变量（自动设置）：
#   OMP_NUM_THREADS=28             # OpenMP线程数
#   MKL_NUM_THREADS=28             # Intel MKL线程数
#   CUDA_VISIBLE_DEVICES=""        # 可选：禁用GPU，使用CPU（默认使用GPU，会自动降级）
#   CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1  # 解决OpenSSL兼容性
#
# 输出文件：
#   训练日志: logs/training_YYYYMMDD_sina_base_k3000.log
#   模型文件: outputs/sina/base_k3000/finetune_tokenizer/best_model/
#            outputs/sina/base_k3000/finetune_predictor/best_model/
#   预测文件: outputs/sina/base_k3000/predictions_master.xlsx
#
# 日志管理：
#   - 训练日志会自动按日期命名
#   - 每次训练结束后自动清理30天前的旧日志
#   - 查看实时日志: tail -f logs/training_$(date +%Y%m%d)*.log
#   - 查看历史日志: ls -lht logs/training_*.log
#
# 进程管理：
#   - 脚本启动时会自动检测并终止正在运行的训练进程
#   - 使用 pkill -9 强制终止，确保资源释放
#   - 避免多个训练任务同时运行导致资源竞争
#
# 重试机制：
#   - 训练失败后会自动重试，重试间隔递增（60秒、120秒、180秒）
#   - 达到最大重试次数后脚本退出，返回错误码
#   - 每次重试都会记录在日志中
#
# 退出码：
#   0 - 训练成功完成
#   1 - 训练失败（已达到最大重试次数）
#
# 注意事项：
#   1. 确保conda环境"kronos"已正确安装所有依赖
#   2. 确保qlib数据已正确初始化（qlib_bin目录）
#   3. 训练过程可能需要数小时，建议在凌晨执行
#   4. 确保磁盘空间充足（模型文件和日志会占用较多空间）
#   5. 可以通过修改脚本中的配置参数来调整训练设置
#
# 故障排查：
#   - 如果训练失败，查看日志文件中的详细错误信息
#   - 如果conda环境激活失败，检查CONDA_PATH路径是否正确
#   - 如果数据获取失败，检查网络连接和数据源配置
#   - 如果内存不足，可以减少TOP_K_STOCKS或NUM_WORKERS
#
# 作者: Kronos Team
# 版本: 2.0
# 更新时间: 2025-11-02
################################################################################

# 设置错误处理：遇到错误不立即退出，而是记录并继续
set +e  # 不立即退出，允许错误处理
set -o pipefail  # 管道命令失败时返回错误码

# 切换到脚本所在目录（必须在其他操作之前）
cd "$(dirname "$0")" || exit 1

# 解决OpenSSL 3.0 legacy provider错误
export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1

# 设置环境变量（确保PATH包含必要路径）
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH"
export HOME="/root"
export USER="root"
export SHELL="/bin/bash"

# 配置参数
DATA_SOURCE="sina"
MODEL_VERSION="base"
TOP_K_STOCKS=1000
NUM_WORKERS=16
TORCH_THREADS=28
EARLY_STOPPING_PATIENCE=3
MAX_RETRIES=3
NUM_GPUS=1  # GPU数量：0=使用所有可用GPU, 1=单GPU(推荐), 2+=指定GPU数量（需要torchrun，可能有DDP卡顿问题）

# 日志目录
LOG_DIR="./logs"
mkdir -p "$LOG_DIR" || exit 1

# 生成脚本启动时的日期（统一用于所有日志文件，即使跨天也不变）
START_DATE=$(date +%Y%m%d)

# Cron执行日志文件（记录整个脚本的执行过程，包括错误）
# 使用启动日期，即使脚本跨天运行也使用同一个日志文件
CRON_LOG="$LOG_DIR/cron_${START_DATE}_${DATA_SOURCE}_${MODEL_VERSION}_k${TOP_K_STOCKS}.log"

# 将所有输出（包括错误）重定向到日志文件
exec >> "$CRON_LOG" 2>&1

# 记录脚本启动信息
echo "=========================================="
echo "Kronos 每日自动训练 - Cron执行日志"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "脚本路径: $0"
echo "工作目录: $(pwd)"
echo "用户: $USER"
echo "PATH: $PATH"
echo "=========================================="

# 初始化conda环境（cron非交互式shell必需）
CONDA_PATH="/root/autodl-tmp/miniconda3"
if [ -f "$CONDA_PATH/etc/profile.d/conda.sh" ]; then
    echo "正在激活conda环境..."
    source "$CONDA_PATH/etc/profile.d/conda.sh" || {
        echo "错误: 无法加载conda.sh"
        exit 1
    }
    conda activate kronos || {
        echo "错误: 无法激活kronos环境"
        exit 1
    }
    echo "✓ Conda环境已激活: $CONDA_DEFAULT_ENV"
else
    echo "警告: conda.sh不存在，尝试备用方案..."
    # 备用方案：使用conda hook
    if [ -f "$CONDA_PATH/bin/conda" ]; then
        eval "$($CONDA_PATH/bin/conda shell.bash hook)" || {
            echo "错误: conda hook初始化失败"
            exit 1
        }
        conda activate kronos || {
            echo "错误: 无法激活kronos环境（备用方案）"
            exit 1
        }
        echo "✓ Conda环境已激活（备用方案）: $CONDA_DEFAULT_ENV"
    else
        echo "错误: conda路径不存在: $CONDA_PATH"
        exit 1
    fi
fi

# 验证Python环境
PYTHON_PATH=$(which python || which python3)
if [ -z "$PYTHON_PATH" ]; then
    echo "错误: 无法找到python命令"
    exit 1
fi
echo "✓ Python路径: $PYTHON_PATH"
echo "✓ Python版本: $($PYTHON_PATH --version 2>&1)"

# 设置环境变量
export OMP_NUM_THREADS=28
export MKL_NUM_THREADS=28
export OPENBLAS_NUM_THREADS=28
export VECLIB_MAXIMUM_THREADS=28
export NUMEXPR_NUM_THREADS=28
# CUDA_VISIBLE_DEVICES 不设置，允许使用GPU（如果需要禁用GPU，可以设置 export CUDA_VISIBLE_DEVICES=""）

# 训练日志文件（会由Python程序自动创建，包含数据源、模型版本和股票数量）
# 实际文件名格式：training_20251120_sina_base_k3000.log（使用启动日期，即使跨天也不变）
TRAIN_LOG="$LOG_DIR/training_${START_DATE}_${DATA_SOURCE}_${MODEL_VERSION}_k${TOP_K_STOCKS}.log"

echo "配置参数："
echo "  - 数据源: $DATA_SOURCE"
echo "  - 模型版本: $MODEL_VERSION"
echo "  - 股票数量: $TOP_K_STOCKS"
echo "  - Worker数: $NUM_WORKERS"
echo "  - 线程数: $TORCH_THREADS"
echo "  - 最大重试: $MAX_RETRIES"
echo "  - 启动日期: $START_DATE（日志文件将使用此日期，即使跨天也不变）"
echo ""

# 检查并自动清理旧训练进程
LOCK_FILE="training.lock"
if [ -f "$LOCK_FILE" ]; then
    echo "检测到训练锁文件，正在清理旧进程..."
    
    # 调用 kill_training.sh 强制终止旧进程
    if [ -f "./kill_training.sh" ]; then
        bash ./kill_training.sh --force
        if [ $? -eq 0 ]; then
            echo "✓ 旧进程已清理"
        else
            echo "✗ 清理旧进程失败"
            exit 1
        fi
    else
        echo "✗ 错误: kill_training.sh 不存在"
        exit 1
    fi
    
    echo ""
fi

# 训练函数
train_model() {
    echo ""
    echo "开始训练..."
    echo "训练日志将写入: $TRAIN_LOG"
    
    # 检测可用GPU数量
    if command -v nvidia-smi &> /dev/null; then
        AVAILABLE_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
        echo "检测到 $AVAILABLE_GPUS 个GPU"
    else
        AVAILABLE_GPUS=0
        echo "未检测到nvidia-smi命令，将使用CPU训练"
    fi
    
    # 决定使用的GPU数量
    if [ "$NUM_GPUS" -eq 0 ]; then
        # 使用所有可用GPU
        ACTUAL_NUM_GPUS=$AVAILABLE_GPUS
    else
        # 使用指定数量的GPU
        ACTUAL_NUM_GPUS=$NUM_GPUS
    fi
    
    # 构建训练命令
    if [ "$ACTUAL_NUM_GPUS" -gt 1 ]; then
        # 多GPU训练：使用torchrun启动DDP
        echo "使用DDP（DistributedDataParallel）多GPU训练模式"
        echo "GPU数量: $ACTUAL_NUM_GPUS"
        echo "执行命令: torchrun --standalone --nproc_per_node=$ACTUAL_NUM_GPUS main.py ..."
        
        torchrun --standalone --nproc_per_node="$ACTUAL_NUM_GPUS" main.py \
            --data-source "$DATA_SOURCE" \
            --model-version "$MODEL_VERSION" \
            --top-k-stocks "$TOP_K_STOCKS" \
            --num-workers "$NUM_WORKERS" \
            --torch-threads "$TORCH_THREADS" \
            --early-stopping-patience "$EARLY_STOPPING_PATIENCE" \
            --min-gpu-memory 3.0 \
            --log-interval 1 \
            --start-timestamp "$START_DATE" 2>&1
    else
        # 单GPU或CPU训练：直接使用python
        if [ "$ACTUAL_NUM_GPUS" -eq 1 ]; then
            echo "使用单GPU训练模式"
        else
            echo "使用CPU训练模式"
        fi
        echo "执行命令: python main.py ..."
        
        "$PYTHON_PATH" main.py \
            --data-source "$DATA_SOURCE" \
            --model-version "$MODEL_VERSION" \
            --top-k-stocks "$TOP_K_STOCKS" \
            --num-workers "$NUM_WORKERS" \
            --torch-threads "$TORCH_THREADS" \
            --early-stopping-patience "$EARLY_STOPPING_PATIENCE" \
            --min-gpu-memory 3.0 \
            --log-interval 1 \
            --start-timestamp "$START_DATE" 2>&1
    fi
    
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "训练失败，退出码: $exit_code"
        echo "详细错误信息请查看: $TRAIN_LOG"
    fi
    
    return $exit_code
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
echo ""
echo "=========================================="
echo "训练结束"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Cron日志文件: $CRON_LOG"

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
echo "  保留的最近日志文件："
ls -lht "$LOG_DIR"/training_*.log 2>/dev/null | head -5 || echo "  （无日志文件）"
echo "✓ 日志清理完成"

exit $exit_code

