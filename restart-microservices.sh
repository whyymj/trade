#!/bin/bash

# FundProphet 微服务重启脚本

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

main() {
    header "FundProphet 微服务重启"
    
    service=$1
    
    if [ -z "$service" ]; then
        info "重启所有服务..."
        docker compose -f "$COMPOSE_FILE" restart
    else
        info "重启 $service..."
        docker compose -f "$COMPOSE_FILE" restart "$service"
    fi
    
    sleep 5
    
    header "重启完成"
    info "服务已重启"
    
    if [ -z "$service" ]; then
        docker compose -f "$COMPOSE_FILE" ps
    else
        docker compose -f "$COMPOSE_FILE" ps "$service"
    fi
}

main "$@"