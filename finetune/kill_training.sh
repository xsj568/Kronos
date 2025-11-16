#!/bin/bash
# 杀掉正在运行的训练服务（只杀掉main.py相关的训练进程，避免误杀其他GPU进程）

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

LOCK_FILE="training.lock"

echo "=================================="
echo "终止训练服务"
echo "=================================="
echo ""

# 查找所有main.py训练进程（通过命令行参数识别，避免误杀其他进程）
TRAINING_PIDS=$(ps aux | grep "python.*main.py" | grep -v grep | awk '{print $2}')

if [ -z "$TRAINING_PIDS" ]; then
    echo -e "${YELLOW}⚠ 未找到训练进程（main.py）${NC}"
    
    # 检查锁文件
    if [ -f "$LOCK_FILE" ]; then
        echo "清理残留的锁文件..."
        rm -f "$LOCK_FILE"
        echo -e "${GREEN}✓ 锁文件已清理${NC}"
    fi
    
    exit 0
fi

# 显示找到的训练进程
echo "找到以下训练进程:"
for PID in $TRAINING_PIDS; do
    ps -p $PID -o pid,user,%cpu,%mem,cmd --no-headers 2>/dev/null | head -1
done

# 确认终止
if [ "$1" != "-f" ] && [ "$1" != "--force" ]; then
    echo ""
    read -p "确认终止这些训练进程? (y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "已取消"
        exit 0
    fi
fi

# 终止所有训练进程
echo ""
echo "正在终止训练进程..."

KILLED_COUNT=0
for PID in $TRAINING_PIDS; do
    # 验证进程是否存在且确实是main.py
    if ps -p $PID > /dev/null 2>&1; then
        CMD=$(ps -p $PID -o cmd= 2>/dev/null)
        if echo "$CMD" | grep -q "main.py"; then
            echo "终止进程 $PID..."
            
            # 优雅终止
            kill -15 $PID 2>/dev/null
            sleep 1
            
            # 检查是否需要强制终止
            if ps -p $PID > /dev/null 2>&1; then
                echo "  进程未响应，强制终止..."
                kill -9 $PID 2>/dev/null
                sleep 1
            fi
            
            # 验证
            if ! ps -p $PID > /dev/null 2>&1; then
                echo -e "  ${GREEN}✓ 进程 $PID 已终止${NC}"
                KILLED_COUNT=$((KILLED_COUNT + 1))
                
                # 清理子进程
                CHILD_PIDS=$(pgrep -P $PID 2>/dev/null)
                if [ -n "$CHILD_PIDS" ]; then
                    echo "  清理 $(echo $CHILD_PIDS | wc -w) 个子进程..."
                    kill -9 $CHILD_PIDS 2>/dev/null
                fi
            else
                echo -e "  ${RED}✗ 无法终止进程 $PID${NC}"
            fi
        fi
    fi
done

if [ $KILLED_COUNT -gt 0 ]; then
    echo -e "${GREEN}✓ 已终止 $KILLED_COUNT 个训练进程${NC}"
else
    echo -e "${YELLOW}⚠ 未能终止任何进程${NC}"
fi

# 清理锁文件
if [ -f "$LOCK_FILE" ]; then
    rm -f "$LOCK_FILE"
    echo -e "${GREEN}✓ 锁文件已清理${NC}"
fi

# 清理GPU缓存（只清理kronos环境的）
if command -v python &> /dev/null; then
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null && echo -e "${GREEN}✓ GPU缓存已清理${NC}"
fi

echo ""
echo "=================================="
echo -e "${GREEN}完成${NC}"
echo "=================================="

