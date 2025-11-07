#!/bin/bash
# 杀掉正在运行的训练服务（通过PID文件，避免误杀）

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

LOCK_FILE="training.lock"

echo "=================================="
echo "终止训练服务"
echo "=================================="
echo ""

# 检查PID文件
if [ ! -f "$LOCK_FILE" ]; then
    echo -e "${RED}✗ 未找到PID文件: $LOCK_FILE${NC}"
    echo "  没有正在运行的训练服务，或锁文件已被删除"
    exit 0
fi

# 读取PID
MAIN_PID=$(cat "$LOCK_FILE" 2>/dev/null)
if [ -z "$MAIN_PID" ]; then
    echo -e "${RED}✗ PID文件为空${NC}"
    rm -f "$LOCK_FILE"
    exit 1
fi

echo "从PID文件读取到: $MAIN_PID"

# 验证进程是否存在
if ! ps -p $MAIN_PID > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠ 进程 $MAIN_PID 不存在（可能已退出）${NC}"
    rm -f "$LOCK_FILE"
    echo "已清理PID文件"
    exit 0
fi

# 显示进程信息
echo ""
echo "进程信息:"
ps -p $MAIN_PID -o pid,user,%cpu,%mem,cmd --no-headers

# 确认终止
if [ "$1" != "-f" ] && [ "$1" != "--force" ]; then
    echo ""
    read -p "确认终止此进程? (y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "已取消"
        exit 0
    fi
fi

# 终止进程
echo ""
echo "正在终止进程..."

# 优雅终止
kill -15 $MAIN_PID 2>/dev/null
sleep 2

# 检查是否需要强制终止
if ps -p $MAIN_PID > /dev/null 2>&1; then
    echo "进程未响应，强制终止..."
    kill -9 $MAIN_PID 2>/dev/null
    sleep 1
fi

# 验证
if ps -p $MAIN_PID > /dev/null 2>&1; then
    echo -e "${RED}✗ 无法终止进程 $MAIN_PID${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 进程已终止${NC}"

# 清理子进程
CHILD_PIDS=$(pgrep -P $MAIN_PID 2>/dev/null)
if [ -n "$CHILD_PIDS" ]; then
    echo "清理 $(echo $CHILD_PIDS | wc -w) 个子进程..."
    kill -9 $CHILD_PIDS 2>/dev/null
fi

# 清理PID文件
rm -f "$LOCK_FILE"
echo -e "${GREEN}✓ PID文件已清理${NC}"

# 清理GPU缓存
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null && echo -e "${GREEN}✓ GPU缓存已清理${NC}"

echo ""
echo "=================================="
echo -e "${GREEN}完成${NC}"
echo "=================================="

