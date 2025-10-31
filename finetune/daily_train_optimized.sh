#!/bin/bash
# Kronos 每日训练优化脚本
# 功能：
# 1. 从sina抓取10000个美股数据
# 2. 基于base模型进行微调
# 3. 在历史模型和当前模型中选择最佳模型
# 4. 更新历史最佳模型
# 5. 预测下一个工作日的涨跌幅
# 6. 增量更新预测结果到主Excel文件

set -e  # 遇到错误立即退出
set -u  # 使用未定义变量时报错

# ========== 配置区域 ==========
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 日志配置 - 输出到当前目录
LOG_DIR="."  # 当前脚本目录
TODAY=$(date '+%Y%m%d')
LOG_FILE="${LOG_DIR}/daily_train_${TODAY}.log"
# mkdir -p "$LOG_DIR"  # 当前目录已存在，无需创建

# Python环境配置 - 使用kronos conda环境
PYTHON_CMD="/root/autodl-tmp/miniconda3/envs/kronos/bin/python"
# 如果需要激活conda环境，也可以使用以下方式
# source /root/autodl-tmp/miniconda3/bin/activate kronos

# 训练配置
DATA_SOURCE="sina"           # 数据源：sina（新浪美股）或qlib
MODEL_VERSION="base"         # 模型版本：mini/small/base
EARLY_STOPPING_PATIENCE=3    # 提前停止耐心值（更小的值可以加快训练）
USE_GPU="--cpu"              # 是否使用GPU，如果有GPU则设置为空字符串 ""

# 输出目录
OUTPUT_DIR="./outputs/${DATA_SOURCE}/${MODEL_VERSION}"
MASTER_EXCEL="${OUTPUT_DIR}/predictions_master.xlsx"

# ========== 日志函数 ==========
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $*" | tee -a "$LOG_FILE"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*" | tee -a "$LOG_FILE" >&2
}

log_success() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SUCCESS] $*" | tee -a "$LOG_FILE"
}

# ========== 预检查 ==========
pre_check() {
    log_info "=========================================="
    log_info "开始每日训练流程"
    log_info "=========================================="
    log_info "配置信息："
    log_info "  - 数据源: ${DATA_SOURCE}"
    log_info "  - 模型版本: ${MODEL_VERSION}"
    log_info "  - 提前停止耐心值: ${EARLY_STOPPING_PATIENCE}"
    log_info "  - GPU使用: ${USE_GPU:-GPU}"
    log_info "  - 输出目录: ${OUTPUT_DIR}"
    log_info "  - 日志文件: ${LOG_FILE}"
    
    # 检查Python环境
    if ! command -v $PYTHON_CMD &> /dev/null; then
        log_error "Python未找到，请确保Python已安装"
        exit 1
    fi
    
    log_info "Python版本: $($PYTHON_CMD --version)"
    
    # 检查必要的Python包
    log_info "检查必要的Python包..."
    $PYTHON_CMD -c "import torch; import pandas; import numpy" 2>/dev/null
    if [ $? -ne 0 ]; then
        log_error "缺少必要的Python包，请运行: pip install -r requirements.txt"
        exit 1
    fi
    
    log_info "✓ 预检查通过"
}

# ========== 主训练流程 ==========
run_training() {
    log_info "=========================================="
    log_info "启动训练流程"
    log_info "=========================================="
    
    # 构建训练命令
    TRAIN_CMD="$PYTHON_CMD main.py \
        --data-source ${DATA_SOURCE} \
        --model-version ${MODEL_VERSION} \
        --early-stopping-patience ${EARLY_STOPPING_PATIENCE} \
        ${USE_GPU}"
    
    log_info "执行命令: ${TRAIN_CMD}"
    
    # 执行训练
    if $TRAIN_CMD 2>&1 | tee -a "$LOG_FILE"; then
        log_success "✓ 训练流程完成"
        return 0
    else
        log_error "✗ 训练流程失败"
        return 1
    fi
}

# ========== 训练后处理 ==========
post_training() {
    log_info "=========================================="
    log_info "训练后处理"
    log_info "=========================================="
    
    # 检查主预测Excel文件是否存在
    if [ -f "$MASTER_EXCEL" ]; then
        log_success "✓ 主预测Excel文件已生成: ${MASTER_EXCEL}"
        
        # 统计预测记录数
        RECORD_COUNT=$($PYTHON_CMD -c "
import pandas as pd
try:
    df = pd.read_excel('${MASTER_EXCEL}', sheet_name='预测历史')
    print(len(df))
except:
    print(0)
" 2>/dev/null)
        
        log_info "主Excel文件总记录数: ${RECORD_COUNT:-0}"
    else
        log_error "✗ 主预测Excel文件未找到: ${MASTER_EXCEL}"
    fi
    
    # 列出最新的预测文件
    LATEST_PREDICTION=$(find "${OUTPUT_DIR}/predictions" -name "prediction_*.xlsx" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    if [ -n "$LATEST_PREDICTION" ]; then
        log_info "最新详细预测文件: ${LATEST_PREDICTION}"
    fi
    
    # 检查模型历史目录
    MODEL_HISTORY_DIR="../model_history/${DATA_SOURCE}/${MODEL_VERSION}"
    if [ -d "$MODEL_HISTORY_DIR" ]; then
        if [ -d "${MODEL_HISTORY_DIR}/best_tokenizer" ] && [ -d "${MODEL_HISTORY_DIR}/best_predictor" ]; then
            log_success "✓ 历史最佳模型已更新: ${MODEL_HISTORY_DIR}"
        else
            log_error "✗ 历史最佳模型目录不完整"
        fi
    else
        log_error "✗ 历史最佳模型目录未找到: ${MODEL_HISTORY_DIR}"
    fi
}

# ========== 清理临时文件 ==========
cleanup_temp_files() {
    log_info "=========================================="
    log_info "清理临时文件"
    log_info "=========================================="
    
    # 清理临时模型检查点（保留最佳模型）
    TEMP_CHECKPOINTS="${OUTPUT_DIR}/finetune_tokenizer/checkpoints/current_model_epoch_*"
    if ls $TEMP_CHECKPOINTS 1> /dev/null 2>&1; then
        rm -rf $TEMP_CHECKPOINTS
        log_info "✓ 已清理临时tokenizer检查点"
    fi
    
    TEMP_CHECKPOINTS="${OUTPUT_DIR}/finetune_predictor/checkpoints/current_model_epoch_*"
    if ls $TEMP_CHECKPOINTS 1> /dev/null 2>&1; then
        rm -rf $TEMP_CHECKPOINTS
        log_info "✓ 已清理临时predictor检查点"
    fi
    
    # 保留最近7天的日志
    find "$LOG_DIR" -name "daily_train_*.log" -type f -mtime +7 -delete 2>/dev/null
    log_info "✓ 已清理旧日志文件（保留最近7天）"
}

# ========== 发送通知（可选）==========
send_notification() {
    local status=$1
    local message=$2
    
    log_info "=========================================="
    log_info "发送训练结果通知"
    log_info "=========================================="
    
    # 这里可以添加邮件、钉钉、企业微信等通知
    # 示例：发送邮件
    # if command -v mail &> /dev/null; then
    #     echo "$message" | mail -s "Kronos训练结果：${status}" your_email@example.com
    #     log_info "✓ 邮件通知已发送"
    # fi
    
    log_info "通知内容: ${message}"
}

# ========== 生成摘要报告 ==========
generate_summary() {
    log_info "=========================================="
    log_info "生成训练摘要报告"
    log_info "=========================================="
    
    # 读取训练摘要
    TOKENIZER_SUMMARY="${OUTPUT_DIR}/finetune_tokenizer/training_summary.json"
    PREDICTOR_SUMMARY="${OUTPUT_DIR}/finetune_predictor/training_summary.json"
    
    if [ -f "$TOKENIZER_SUMMARY" ]; then
        log_info "Tokenizer训练摘要:"
        $PYTHON_CMD -c "
import json
try:
    with open('${TOKENIZER_SUMMARY}', 'r') as f:
        data = json.load(f)
    print(f\"  - 训练时长: {data.get('total_time', 'N/A')}\")
    print(f\"  - 最佳验证损失: {data.get('best_val_loss', 'N/A'):.4f}\")
    print(f\"  - 最佳测试损失: {data.get('best_test_loss', 'N/A'):.4f}\")
    print(f\"  - 训练轮数: {data.get('epochs', 'N/A')}\")
except Exception as e:
    print(f\"  - 读取摘要失败: {e}\")
" | tee -a "$LOG_FILE"
    fi
    
    if [ -f "$PREDICTOR_SUMMARY" ]; then
        log_info "Predictor训练摘要:"
        $PYTHON_CMD -c "
import json
try:
    with open('${PREDICTOR_SUMMARY}', 'r') as f:
        data = json.load(f)
    print(f\"  - 训练时长: {data.get('total_time', 'N/A')}\")
    print(f\"  - 最佳验证损失: {data.get('best_val_loss', 'N/A'):.4f}\")
    print(f\"  - 最佳测试损失: {data.get('best_test_loss', 'N/A'):.4f}\")
    print(f\"  - 训练轮数: {data.get('epochs', 'N/A')}\")
except Exception as e:
    print(f\"  - 读取摘要失败: {e}\")
" | tee -a "$LOG_FILE"
    fi
}

# ========== 主函数 ==========
main() {
    local start_time=$(date +%s)
    
    # 预检查
    pre_check
    
    # 运行训练
    if run_training; then
        # 训练后处理
        post_training
        
        # 生成摘要
        generate_summary
        
        # 清理临时文件
        cleanup_temp_files
        
        # 计算总耗时
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local hours=$((duration / 3600))
        local minutes=$(((duration % 3600) / 60))
        local seconds=$((duration % 60))
        
        log_success "=========================================="
        log_success "每日训练流程完成"
        log_success "总耗时: ${hours}小时 ${minutes}分钟 ${seconds}秒"
        log_success "=========================================="
        
        # 发送成功通知
        send_notification "SUCCESS" "训练成功完成，耗时 ${hours}h ${minutes}m ${seconds}s"
        
        exit 0
    else
        log_error "=========================================="
        log_error "每日训练流程失败"
        log_error "请查看日志文件: ${LOG_FILE}"
        log_error "=========================================="
        
        # 发送失败通知
        send_notification "FAILED" "训练失败，请查看日志: ${LOG_FILE}"
        
        exit 1
    fi
}

# 执行主函数
main "$@"

