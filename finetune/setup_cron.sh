#!/bin/bash
################################################################################
# Kronos 定时任务设置脚本
# 
# 功能说明：
#   用于配置和管理Kronos模型的每日自动训练定时任务（crontab）
#   自动设置cron任务，在指定时间运行daily_train.sh进行模型训练
#
# 使用方法：
#   bash setup_cron.sh
#   
# 功能选项：
#   1) 添加/更新每日训练定时任务 - 设置新的定时任务或更新现有任务
#   2) 删除Kronos相关定时任务    - 移除所有Kronos定时任务
#   3) 查看当前定时任务          - 显示当前所有crontab任务
#   4) 退出                      - 退出脚本
#
# 时间设置：
#   脚本会提示输入任务执行时间：
#   - 小时 (0-23): 例如 0 表示凌晨零点，5 表示早上5点
#   - 分钟 (0-59): 例如 0 表示整点，5 表示5分
#   
# 默认时间（针对美股数据）：
#   每天早上05:05运行（上海时间）
#   - 美股收盘时间：美东时间 4:00 PM
#   - 收盘后5分钟：美东时间 4:05 PM
#   - 转换到上海时间：次日 5:05 AM（冬令时）或 4:05 AM（夏令时）
#   - 为确保数据可用，使用 5:05 AM 作为默认启动时间
#   
# 示例：
#   每天凌晨00:00运行: 小时=0, 分钟=0
#   每天早上05:05运行: 小时=5, 分钟=5（推荐，美股数据）
#   每天上午09:00运行: 小时=9, 分钟=0
#
# 日志文件：
#   - 训练日志: logs/training_YYYYMMDD_sina_base_k3000.log
#   （由 daily_train.sh 内部管理，无需cron额外重定向）
#
# 管理命令：
#   - 查看定时任务: crontab -l
#   - 编辑定时任务: crontab -e
#   - 查看训练日志: tail -f logs/training_$(date +%Y%m%d)*.log
#   - 手动运行:    bash daily_train.sh
#
# 注意事项：
#   1. 脚本会自动配置conda环境，无需手动激活
#   2. 如果已存在Kronos定时任务，会先删除旧任务再添加新任务
#   3. 默认时间已优化为美股收盘后启动（上海时间早上5:05）
#   4. 如需抓取其他市场数据，可根据实际需求调整启动时间
#   5. 定时任务会在后台运行，不会阻塞其他操作
#
# 作者: Kronos Team
# 版本: 2.0
# 更新时间: 2025-11-02
################################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DAILY_TRAIN_SCRIPT="${SCRIPT_DIR}/daily_train.sh"
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
    printf "${RED}错误: 找不到训练脚本 $DAILY_TRAIN_SCRIPT${NC}\n"
    exit 1
fi

# 确保训练脚本有执行权限
chmod +x "$DAILY_TRAIN_SCRIPT"
printf "${GREEN}✓${NC} 训练脚本: $DAILY_TRAIN_SCRIPT\n"

# 创建日志目录
mkdir -p "$LOG_DIR"
printf "${GREEN}✓${NC} 日志目录: $LOG_DIR\n"

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
read -p "请输入选项 [1-4，默认: 1]: " choice

# 设置默认值
choice=${choice:-1}

case $choice in
    1)
        echo ""
        echo "设置每日训练时间"
        echo "----------------------------"
        echo "默认时间：05:05（美股收盘后5分钟，美东时间4:05 PM对应上海时间次日5:05 AM）"
        echo "说明：建议在美股收盘后执行，以抓取最新的美股数据"
        echo ""
        read -p "请输入小时 (0-23) [默认: 5]: " hour
        read -p "请输入分钟 (0-59) [默认: 5]: " minute
        
        hour=${hour:-5}
        minute=${minute:-5}
        
        # 验证输入（兼容sh和bash）
        case "$hour" in
            ''|*[!0-9]*) 
                printf "${RED}错误: 小时必须是数字${NC}\n"
                exit 1
                ;;
        esac
        
        if [ "$hour" -lt 0 ] || [ "$hour" -gt 23 ]; then
            printf "${RED}错误: 小时必须是0-23之间${NC}\n"
            exit 1
        fi
        
        case "$minute" in
            ''|*[!0-9]*) 
                printf "${RED}错误: 分钟必须是数字${NC}\n"
                exit 1
                ;;
        esac
        
        if [ "$minute" -lt 0 ] || [ "$minute" -gt 59 ]; then
            printf "${RED}错误: 分钟必须是0-59之间${NC}\n"
            exit 1
        fi
        
        # 创建新的crontab（日志由脚本内部处理，避免重复）
        cron_entry="$minute $hour * * * cd $SCRIPT_DIR && $DAILY_TRAIN_SCRIPT"
        
        # 删除旧的Kronos任务
        (crontab -l 2>/dev/null | grep -v "daily_train" | grep -v "Kronos" || true) | crontab -
        
        # 添加新任务
        (crontab -l 2>/dev/null; echo "# Kronos 每日训练任务"; echo "# 日志由 daily_train.sh 内部处理，无需cron重定向"; echo "$cron_entry") | crontab -
        
        echo ""
        printf "${GREEN}✓ 定时任务已设置${NC}\n"
        echo ""
        echo "任务详情:"
        echo "  执行时间: 每天 $hour:$(printf "%02d" $minute)"
        echo "  脚本路径: $DAILY_TRAIN_SCRIPT"
        echo "  日志文件: logs/training_YYYYMMDD_sina_base_k3000.log（由脚本自动管理）"
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
        printf "${GREEN}✓ Kronos定时任务已删除${NC}\n"
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
        printf "${RED}无效选项${NC}\n"
        exit 1
        ;;
esac

echo ""
echo "=================================================="
echo "提示："
echo "  - 手动运行训练: $DAILY_TRAIN_SCRIPT"
echo "  - 查看定时任务: crontab -l"
echo "  - 编辑定时任务: crontab -e"
echo "  - 查看训练日志: tail -f $LOG_DIR/training_\$(date +%Y%m%d)*.log"
echo "  - 列出所有日志: ls -lht $LOG_DIR/training_*.log | head"
echo "=================================================="

