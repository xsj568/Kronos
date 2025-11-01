#!/bin/bash

# Kronos CPU多核优化训练脚本
# 针对112核CPU服务器优化

echo "=========================================="
echo "Kronos CPU多核优化训练"
echo "=========================================="
echo "服务器配置："
echo "- CPU核心数: $(nproc)"
echo "- 物理核心: 56 (28核 x 2线程)"
echo "- 逻辑CPU: 112"
echo ""

# 设置环境变量优化
export OMP_NUM_THREADS=28
export MKL_NUM_THREADS=28
export OPENBLAS_NUM_THREADS=28
export VECLIB_MAXIMUM_THREADS=28
export NUMEXPR_NUM_THREADS=28

# 禁用CUDA（确保使用CPU）
export CUDA_VISIBLE_DEVICES=""

echo "环境变量设置："
echo "- OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "- MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo ""

# 训练参数
DATA_SOURCE="sina"
MODEL_VERSION="base"
EARLY_STOPPING_PATIENCE=3

# CPU优化参数
NUM_WORKERS=16          # DataLoader工作进程数
TORCH_THREADS=28        # PyTorch计算线程数（物理核心数）

echo "训练参数："
echo "- 数据源: $DATA_SOURCE"
echo "- 模型版本: $MODEL_VERSION"
echo "- 提前停止耐心值: $EARLY_STOPPING_PATIENCE"
echo "- DataLoader Workers: $NUM_WORKERS"
echo "- PyTorch Threads: $TORCH_THREADS"
echo "=========================================="
echo ""

# 运行训练（不启用torch.compile，首次测试）
python main.py \
    --cpu \
    --data-source $DATA_SOURCE \
    --model-version $MODEL_VERSION \
    --early-stopping-patience $EARLY_STOPPING_PATIENCE \
    --num-workers $NUM_WORKERS \
    --torch-threads $TORCH_THREADS

echo ""
echo "=========================================="
echo "训练完成"
echo "=========================================="

