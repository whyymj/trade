#!/bin/bash

# FundProphet 微服务状态查看脚本

set -e

COMPOSE_FILE="docker-compose.microservices.yml"
ENV_FILE=".env"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE} $1${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

print_service_status() {
    header "服务运行状态"
    docker compose -f "$COMPOSE_FILE" ps
}

print_container_stats() {
    header "容器资源使用"
    
    if docker compose -f "$COMPOSE_FILE" ps -q | grep -q .; then
        docker stats $(docker compose -f "$COMPOSE_FILE" ps -q) --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}\t{{.BlockIO}}"
    else
        warn "没有运行中的容器"
    fi
}

print_network_info() {
    header "网络信息"
    docker network ls | grep fundprophet || warn "未找到 fundprophet-network"
}

print_volume_info() {
    header "数据卷信息"
    docker volume ls | grep fundprophet || warn "未找到 fundprophet 数据卷"
}

print_logs_tail() {
    header "最近日志"
    
    services=(
        "stock-service"
        "fund-service"
        "news-service"
        "market-service"
        "fund-intel-service"
        "llm-service"
    )
    
    for service in "${services[@]}"; do
        if docker compose -f "$COMPOSE_FILE" ps -q "$service" | grep -q .; then
            echo -e "${BLUE}$service:${NC}"
            docker compose -f "$COMPOSE_FILE" logs --tail=3 "$service" 2>&1 | grep -v "WARNING" || true
            echo ""
        fi
    done
}

print_environment_info() {
    header "环境配置"
    
    if [ -f "$ENV_FILE" ]; then
        echo "DB_HOST: $(grep '^DB_HOST=' "$ENV_FILE" | cut -d'=' -f2)"
        echo "DB_NAME: $(grep '^DB_NAME=' "$ENV_FILE" | cut -d'=' -f2)"
        echo "LOG_LEVEL: $(grep '^LOG_LEVEL=' "$ENV_FILE" | cut -d'=' -f2)"
        echo "DEEPSEEK_API_KEY: $(grep '^DEEPSEEK_API_KEY=' "$ENV_FILE" | cut -d'=' -f2 | head -c 10)..."
        echo "MINIMAX_API_KEY: $(grep '^MINIMAX_API_KEY=' "$ENV_FILE" | cut -d'=' -f2 | head -c 10)..."
    else
        warn "环境变量文件不存在"
    fi
}

main() {
    header "FundProphet 微服务状态"
    
    print_service_status
    print_container_stats
    print_network_info
    print_volume_info
    print_environment_info
    
    echo ""
    echo -e "${YELLOW}使用其他命令：${NC}"
    echo -e "  ./start-microservices.sh  - 启动服务"
    echo -e "  ./stop-microservices.sh   - 停止服务"
    echo -e "  ./restart-microservices.sh - 重启服务"
    echo -e "  ./health-check.sh        - 健康检查"
    echo -e "  ./test-all-apis.sh       - API 测试"
}

main "$@"