#!/bin/bash

# 设置Kronos每日自动训练定时任务

echo "=========================================="
echo "Kronos 每日训练定时任务设置"
echo "=========================================="
echo ""

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DAILY_TRAIN_SCRIPT="$SCRIPT_DIR/daily_train.sh"
LOG_DIR="$SCRIPT_DIR/logs"

echo "工作目录: $SCRIPT_DIR"
echo "训练脚本: $DAILY_TRAIN_SCRIPT"
echo "日志目录: $LOG_DIR"
echo ""

# 检查脚本是否存在
if [ ! -f "$DAILY_TRAIN_SCRIPT" ]; then
    echo "错误：找不到训练脚本 $DAILY_TRAIN_SCRIPT"
    exit 1
fi

# 添加执行权限
chmod +x "$DAILY_TRAIN_SCRIPT"
echo "✓ 已设置训练脚本执行权限"

# 创建日志目录
mkdir -p "$LOG_DIR"
echo "✓ 已创建日志目录"

# 定时任务设置选项
echo ""
echo "请选择定时任务执行时间："
echo "1) 每天凌晨 2:00 (推荐)"
echo "2) 每天凌晨 3:00"
echo "3) 每天上午 9:00"
echo "4) 每6小时执行一次"
echo "5) 每周一凌晨 2:00 (周末不训练)"
echo "6) 自定义时间"
echo "7) 不设置定时任务，只测试运行"
echo ""
read -p "请输入选项 (1-7): " choice

case $choice in
    1)
        CRON_TIME="0 2 * * *"
        DESC="每天凌晨2:00"
        ;;
    2)
        CRON_TIME="0 3 * * *"
        DESC="每天凌晨3:00"
        ;;
    3)
        CRON_TIME="0 9 * * *"
        DESC="每天上午9:00"
        ;;
    4)
        CRON_TIME="0 */6 * * *"
        DESC="每6小时一次"
        ;;
    5)
        CRON_TIME="0 2 * * 1"
        DESC="每周一凌晨2:00"
        ;;
    6)
        echo ""
        echo "cron时间格式: 分 时 日 月 周"
        echo "例如: 30 14 * * * (每天14:30)"
        read -p "请输入cron时间表达式: " CRON_TIME
        DESC="自定义时间: $CRON_TIME"
        ;;
    7)
        echo ""
        echo "跳过定时任务设置，准备测试运行..."
        TEST_ONLY=true
        ;;
    *)
        echo "无效的选择"
        exit 1
        ;;
esac

if [ "$TEST_ONLY" != "true" ]; then
    # 检测conda环境
    CONDA_ENV=""
    if [ -n "$CONDA_DEFAULT_ENV" ]; then
        CONDA_ENV="$CONDA_DEFAULT_ENV"
        echo "检测到conda环境: $CONDA_ENV"
    fi
    
    # 构建cron命令
    if [ -n "$CONDA_ENV" ]; then
        # 如果有conda环境，先激活环境
        CRON_CMD="$CRON_TIME source ~/.bashrc && conda activate $CONDA_ENV && cd $SCRIPT_DIR && $DAILY_TRAIN_SCRIPT >> $LOG_DIR/cron_\$(date +\\%Y\\%m\\%d).log 2>&1"
    else
        # 没有conda环境，直接执行
        CRON_CMD="$CRON_TIME cd $SCRIPT_DIR && $DAILY_TRAIN_SCRIPT >> $LOG_DIR/cron_\$(date +\\%Y\\%m\\%d).log 2>&1"
    fi
    
    echo ""
    echo "=========================================="
    echo "定时任务配置"
    echo "=========================================="
    echo "执行时间: $DESC"
    echo "工作目录: $SCRIPT_DIR"
    echo "训练脚本: $DAILY_TRAIN_SCRIPT"
    echo "日志文件: $LOG_DIR/cron_YYYYMMDD.log"
    echo ""
    echo "Cron命令:"
    echo "$CRON_CMD"
    echo ""
    
    read -p "确认添加定时任务？(y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "已取消"
        exit 0
    fi
    
    # 检查是否已存在相同的定时任务
    if crontab -l 2>/dev/null | grep -q "$DAILY_TRAIN_SCRIPT"; then
        echo ""
        echo "警告：检测到已存在的定时任务"
        crontab -l | grep "$DAILY_TRAIN_SCRIPT"
        echo ""
        read -p "是否删除旧任务并添加新任务？(y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            # 删除旧任务
            crontab -l 2>/dev/null | grep -v "$DAILY_TRAIN_SCRIPT" | crontab -
            echo "✓ 已删除旧任务"
        else
            echo "已取消"
            exit 0
        fi
    fi
    
    # 添加新的定时任务
    (crontab -l 2>/dev/null; echo "$CRON_CMD") | crontab -
    
    echo ""
    echo "=========================================="
    echo "✓ 定时任务设置成功"
    echo "=========================================="
    echo ""
    echo "当前所有定时任务："
    crontab -l
    echo ""
fi

# 测试运行
echo ""
echo "=========================================="
echo "测试运行"
echo "=========================================="
read -p "是否立即测试运行训练脚本？(y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "开始测试运行..."
    echo "注意：这将启动完整的训练流程"
    echo "如果只想快速测试，建议Ctrl+C取消后手动运行："
    echo "  python main.py --cpu --top-k-stocks 10"
    echo ""
    sleep 3
    
    # 执行训练脚本
    "$DAILY_TRAIN_SCRIPT"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ 测试运行成功"
    else
        echo ""
        echo "✗ 测试运行失败，请检查日志"
        echo "日志位置: $LOG_DIR/training_$(date +%Y%m%d).log"
    fi
fi

echo ""
echo "=========================================="
echo "设置完成"
echo "=========================================="
echo ""
echo "管理命令："
echo "  查看定时任务: crontab -l"
echo "  编辑定时任务: crontab -e"
echo "  删除定时任务: crontab -r"
echo "  查看训练日志: tail -f $LOG_DIR/training_\$(date +%Y%m%d)*.log"
echo "  查看cron日志: tail -f $LOG_DIR/cron_\$(date +%Y%m%d).log"
echo "  列出所有日志: ls -lht $LOG_DIR/training_*.log | head"
echo "  手动运行训练: $DAILY_TRAIN_SCRIPT"
echo ""

