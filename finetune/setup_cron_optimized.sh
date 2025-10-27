#!/bin/bash
# Kronos 定时任务设置脚本（优化版本）
# 用于设置每日自动训练任务

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAILY_TRAIN_SCRIPT="${SCRIPT_DIR}/daily_train_optimized.sh"
LOG_DIR="${SCRIPT_DIR}"  # 日志输出到脚本当前目录

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=================================================="
echo "    Kronos 每日训练定时任务设置（优化版本）"
echo "=================================================="
echo ""

# 检查训练脚本是否存在
if [ ! -f "$DAILY_TRAIN_SCRIPT" ]; then
    echo -e "${RED}错误: 找不到训练脚本 $DAILY_TRAIN_SCRIPT${NC}"
    exit 1
fi

# 确保训练脚本有执行权限
chmod +x "$DAILY_TRAIN_SCRIPT"
echo -e "${GREEN}✓${NC} 训练脚本: $DAILY_TRAIN_SCRIPT"

# 创建日志目录
mkdir -p "$LOG_DIR"
echo -e "${GREEN}✓${NC} 日志目录: $LOG_DIR"

# 显示当前crontab
echo ""
echo "当前的定时任务："
echo "----------------------------"
crontab -l 2>/dev/null || echo "（无定时任务）"
echo "----------------------------"
echo ""

# 询问用户
echo "请选择操作："
echo "1) 添加/更新每日训练定时任务"
echo "2) 删除Kronos相关定时任务"
echo "3) 查看当前定时任务"
echo "4) 退出"
echo ""
read -p "请输入选项 [1-4]: " choice

case $choice in
    1)
        echo ""
        echo "设置每日训练时间"
        echo "----------------------------"
        echo "建议在凌晨进行训练（例如：02:00）"
        echo ""
        read -p "请输入小时 (0-23) [默认: 2]: " hour
        read -p "请输入分钟 (0-59) [默认: 0]: " minute
        
        hour=${hour:-2}
        minute=${minute:-0}
        
        # 验证输入
        if ! [[ "$hour" =~ ^[0-9]+$ ]] || [ "$hour" -lt 0 ] || [ "$hour" -gt 23 ]; then
            echo -e "${RED}错误: 小时必须是0-23之间的数字${NC}"
            exit 1
        fi
        
        if ! [[ "$minute" =~ ^[0-9]+$ ]] || [ "$minute" -lt 0 ] || [ "$minute" -gt 59 ]; then
            echo -e "${RED}错误: 分钟必须是0-59之间的数字${NC}"
            exit 1
        fi
        
        # 创建新的crontab
        cron_entry="$minute $hour * * * cd $SCRIPT_DIR && $DAILY_TRAIN_SCRIPT >> $LOG_DIR/cron.log 2>&1"
        
        # 删除旧的Kronos任务
        (crontab -l 2>/dev/null | grep -v "daily_train" | grep -v "Kronos" || true) | crontab -
        
        # 添加新任务
        (crontab -l 2>/dev/null; echo "# Kronos 每日训练任务 (优化版本)"; echo "$cron_entry") | crontab -
        
        echo ""
        echo -e "${GREEN}✓ 定时任务已设置${NC}"
        echo ""
        echo "任务详情:"
        echo "  执行时间: 每天 $hour:$(printf "%02d" $minute)"
        echo "  脚本路径: $DAILY_TRAIN_SCRIPT"
        echo "  日志文件: $LOG_DIR/cron.log"
        echo ""
        echo "更新后的定时任务："
        echo "----------------------------"
        crontab -l
        echo "----------------------------"
        ;;
        
    2)
        echo ""
        echo "正在删除Kronos相关定时任务..."
        (crontab -l 2>/dev/null | grep -v "daily_train" | grep -v "Kronos" || true) | crontab -
        echo -e "${GREEN}✓ Kronos定时任务已删除${NC}"
        echo ""
        echo "当前的定时任务："
        echo "----------------------------"
        crontab -l 2>/dev/null || echo "（无定时任务）"
        echo "----------------------------"
        ;;
        
    3)
        echo ""
        echo "当前的定时任务："
        echo "----------------------------"
        crontab -l 2>/dev/null || echo "（无定时任务）"
        echo "----------------------------"
        ;;
        
    4)
        echo "退出"
        exit 0
        ;;
        
    *)
        echo -e "${RED}无效选项${NC}"
        exit 1
        ;;
esac

echo ""
echo "=================================================="
echo "提示："
echo "  - 手动运行训练: $DAILY_TRAIN_SCRIPT"
echo "  - 查看定时任务: crontab -l"
echo "  - 编辑定时任务: crontab -e"
echo "  - 查看训练日志: tail -f $LOG_DIR/daily_train_*.log"
echo "  - 查看cron日志: tail -f $LOG_DIR/cron.log"
echo "=================================================="

