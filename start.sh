#!/bin/bash

# FundProphet 启动脚本

echo "========================================="
echo "  FundProphet 基金分析系统"
echo "========================================="

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python3 not found${NC}"
    exit 1
fi

# 检查 Node.js
if ! command -v node &> /dev/null; then
    echo -e "${RED}Error: Node.js not found${NC}"
    exit 1
fi

# 检查依赖
echo -e "${YELLOW}Checking dependencies...${NC}"

# Python 依赖
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}Error: requirements.txt not found${NC}"
    exit 1
fi

# 安装 Python 依赖
echo -e "${YELLOW}Installing Python dependencies...${NC}"
pip3 install -r requirements.txt -q

# 前端依赖
if [ -d "frontend" ]; then
    echo -e "${YELLOW}Installing frontend dependencies...${NC}"
    cd frontend
    if [ -f "package.json" ]; then
        pnpm install -q 2>/dev/null || npm install -q
    fi
    cd ..
fi

echo -e "${GREEN}Dependencies ready!${NC}"
echo ""

# 启动选项
case "${1}" in
    backend)
        echo -e "${GREEN}Starting backend server...${NC}"
        echo "Backend: http://localhost:5050"
        python3 server.py
        ;;
    frontend)
        echo -e "${GREEN}Starting frontend dev server...${NC}"
        echo "Frontend: http://localhost:5173"
        cd frontend && pnpm run dev
        ;;
    all|*)
        echo -e "${GREEN}Starting all services...${NC}"
        echo ""
        echo "Starting backend on port 5050..."
        python3 server.py &
        BACKEND_PID=$!
        
        sleep 2
        
        echo "Starting frontend on port 5173..."
        cd frontend && pnpm run dev &
        FRONTEND_PID=$!
        
        echo ""
        echo "========================================="
        echo -e "  ${GREEN}Services started successfully!${NC}"
        echo "========================================="
        echo "Backend:  http://localhost:5050"
        echo "Frontend: http://localhost:5173"
        echo ""
        echo "Press Ctrl+C to stop all services"
        
        # 等待
        wait
        ;;
esac
