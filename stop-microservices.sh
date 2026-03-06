#!/bin/bash

# FundProphet 微服务停止脚本

set -e

COMPOSE_FILE="docker-compose.microservices.yml"

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

check_running() {
    if ! docker compose -f "$COMPOSE_FILE" ps -q | grep -q .; then
        info "没有运行中的服务"
        exit 0
    fi
}

main() {
    header "FundProphet 微服务停止"
    
    check_running
    
    info "停止所有服务..."
    docker compose -f "$COMPOSE_FILE" down
    
    header "停止完成"
    info "所有服务已停止"
    info "数据卷和未使用的资源已保留"
    info "如需清理，运行 ./status-microservices.sh 查看详情"
}

main "$@"