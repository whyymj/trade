#!/bin/bash

# FundProphet 微服务健康检查脚本

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

total_checks=0
passed_checks=0
failed_checks=0

info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

success() {
    echo -e "${GREEN}✓${NC} $1"
    passed_checks=$((passed_checks + 1))
}

failure() {
    echo -e "${RED}✗${NC} $1"
    failed_checks=$((failed_checks + 1))
}

header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE} $1${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

check_http_service() {
    local name=$1
    local url=$2
    local timeout=${3:-5}
    
    total_checks=$((total_checks + 1))
    
    echo -n "检查 $name ($url)... "
    if curl -s -f --max-time "$timeout" "$url" > /dev/null 2>&1; then
        local response=$(curl -s --max-time "$timeout" "$url")
        echo "$(success "健康")"
        if echo "$response" | grep -q "status"; then
            echo "  响应: $response" | head -c 100
        fi
    else
        failure "不健康"
    fi
}

check_service_health() {
    header "服务健康检查"
    
    check_http_service "Stock Service" "http://localhost:8001/health"
    check_http_service "Fund Service" "http://localhost:8002/health"
    check_http_service "News Service" "http://localhost:8003/health"
    check_http_service "Market Service" "http://localhost:8004/health"
    check_http_service "Fund Intel Service" "http://localhost:8005/health"
    check_http_service "LLM Service" "http://localhost:8006/health"
    check_http_service "Traefik Gateway" "http://localhost:8080/ping" 10
}

check_database() {
    header "数据库连接检查"
    
    total_checks=$((total_checks + 2))
    
    echo -n "检查 MySQL (localhost:3306)... "
    if nc -z localhost 3306 2>/dev/null || docker exec fundprophet-mysql mysqladmin ping -h localhost -uroot -p${DB_ROOT_PASSWORD:-root_password_change_me} --silent 2>/dev/null; then
        success "可连接"
    else
        failure "不可连接"
    fi
    
    echo -n "检查 Redis (localhost:6379)... "
    if nc -z localhost 6379 2>/dev/null || docker exec fundprophet-redis redis-cli ping 2>/dev/null | grep -q "PONG"; then
        success "可连接"
    else
        failure "不可连接"
    fi
}

check_api_endpoints() {
    header "核心 API 端点检查"
    
    total_checks=$((total_checks + 6))
    
    check_http_service "Stock API" "http://localhost:8001/api/stock/list" 10
    check_http_service "Fund API" "http://localhost:8002/api/fund/list" 10
    check_http_service "News API" "http://localhost:8003/api/news/latest" 10
    check_http_service "Market API" "http://localhost:8004/api/market/macro" 10
    check_http_service "Fund Industry API" "http://localhost:8005/api/fund-industry/stats" 10
    check_http_service "LLM API" "http://localhost:8006/api/llm/health" 10
}

check_docker_status() {
    header "Docker 容器状态"
    
    total_checks=$((total_checks + 1))
    
    unhealthy=$(docker compose -f docker-compose.microservices.yml ps --format json | jq -r 'select(.Health != "healthy" and .State == "running") | .Name' 2>/dev/null | wc -l)
    
    if [ "$unhealthy" -eq 0 ]; then
        success "所有运行中容器健康"
    else
        failure "$unhealthy 个容器不健康"
        echo "不健康的容器："
        docker compose -f docker-compose.microservices.yml ps --format json | jq -r 'select(.Health != "healthy" and .State == "running") | .Name' 2>/dev/null
    fi
}

check_resource_usage() {
    header "资源使用情况"
    
    echo "容器资源使用："
    docker compose -f docker-compose.microservices.yml ps --format json | jq -r '.[] | "\(.Name): CPU=\(.Stats.CPU|tostring), Memory=\(.Stats.Memory)"' 2>/dev/null || docker compose -f docker-compose.microservices.yml ps
}

generate_report() {
    header "健康检查报告"
    
    echo -e "${BLUE}检查汇总：${NC}"
    echo -e "  总检查数: ${BLUE}$total_checks${NC}"
    echo -e "  通过数:   ${GREEN}$passed_checks${NC}"
    echo -e "  失败数:   ${RED}$failed_checks${NC}"
    
    if [ $failed_checks -eq 0 ]; then
        echo ""
        success "所有检查通过！系统运行正常。"
        return 0
    else
        echo ""
        warn "发现 $failed_checks 个问题，请检查上述输出。"
        return 1
    fi
}

main() {
    header "FundProphet 健康检查"
    
    if [ -f .env ]; then
        source .env
    fi
    
    check_service_health
    check_database
    check_api_endpoints
    check_docker_status
    check_resource_usage
    
    generate_report
}

main "$@"