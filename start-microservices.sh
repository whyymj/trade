#!/bin/bash

# FundProphet 微服务启动脚本

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

check_env() {
    if [ ! -f "$ENV_FILE" ]; then
        warn "环境变量文件不存在，从模板创建..."
        cp .env.example "$ENV_FILE"
        warn "请编辑 $ENV_FILE 文件并填入正确的配置"
        exit 1
    fi
    
    info "检查环境变量配置..."
    source "$ENV_FILE"
    
    if [ "$DB_PASSWORD" = "fundpass_change_me" ]; then
        warn "数据库密码使用默认值，请修改！"
    fi
    
    if [ "$DEEPSEEK_API_KEY" = "your_deepseek_key_here" ]; then
        warn "DEEPSEEK_API_KEY 使用默认值，请修改！"
    fi
}

check_docker() {
    info "检查 Docker 环境..."
    if ! command -v docker &> /dev/null; then
        error "Docker 未安装"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        error "Docker 守护进程未运行"
        exit 1
    fi
    
    if ! command -v docker compose &> /dev/null && ! docker compose version &> /dev/null; then
        error "Docker Compose 未安装"
        exit 1
    fi
    
    info "Docker 环境检查通过"
}

check_network() {
    info "检查 Docker 网络..."
    if ! docker network ls | grep -q "fundprophet-network"; then
        info "创建网络 fundprophet-network"
        docker network create fundprophet-network
    fi
}

check_volumes() {
    info "检查 Docker 数据卷..."
    if ! docker volume ls | grep -q "fundprophet-redis-data"; then
        info "创建数据卷 fundprophet-redis-data"
        docker volume create fundprophet-redis-data
    fi
    
    if ! docker volume ls | grep -q "fundprophet-mysql-data"; then
        info "创建数据卷 fundprophet-mysql-data"
        docker volume create fundprophet-mysql-data
    fi
}

start_infrastructure() {
    header "启动基础设施服务"
    
    info "启动 Redis..."
    docker compose -f "$COMPOSE_FILE" up -d redis
    
    info "启动 MySQL..."
    docker compose -f "$COMPOSE_FILE" up -d mysql
    
    info "启动 Traefik..."
    docker compose -f "$COMPOSE_FILE" up -d traefik
    
    info "等待基础设施服务就绪..."
    sleep 15
}

start_services() {
    header "启动应用服务"
    
    services=(
        "stock-service"
        "fund-service"
        "news-service"
        "market-service"
        "llm-service"
        "fund-intel-service"
    )
    
    for service in "${services[@]}"; do
        info "启动 $service..."
        docker compose -f "$COMPOSE_FILE" up -d "$service"
        sleep 3
    done
}

start_scheduler() {
    info "启动 Scheduler..."
    docker compose -f "$COMPOSE_FILE" up -d scheduler
}

wait_for_health() {
    header "等待服务健康检查"
    
    services=(
        "stock-service:8001"
        "fund-service:8002"
        "news-service:8003"
        "market-service:8004"
        "fund-intel-service:8005"
        "llm-service:8006"
    )
    
    max_attempts=30
    for service in "${services[@]}"; do
        name="${service%:*}"
        port="${service#*:}"
        
        info "等待 $name 健康..."
        attempts=0
        while [ $attempts -lt $max_attempts ]; do
            if curl -s -f "http://localhost:$port/health" > /dev/null 2>&1; then
                echo -e "${GREEN}✓${NC} $name 健康检查通过"
                break
            fi
            attempts=$((attempts + 1))
            echo -n "."
            sleep 2
        done
        
        if [ $attempts -eq $max_attempts ]; then
            echo ""
            warn "$name 健康检查超时"
        else
            echo ""
        fi
    done
}

print_status() {
    header "服务状态"
    docker compose -f "$COMPOSE_FILE" ps
}

print_access_info() {
    header "访问信息"
    
    echo -e "${GREEN}API Gateway (Traefik):${NC} http://localhost:80"
    echo -e "${GREEN}Traefik Dashboard:${NC} http://localhost:8080"
    echo ""
    echo -e "${BLUE}各服务直接访问：${NC}"
    echo -e "  Stock Service:  ${GREEN}http://localhost:8001${NC}"
    echo -e "  Fund Service:   ${GREEN}http://localhost:8002${NC}"
    echo -e "  News Service:   ${GREEN}http://localhost:8003${NC}"
    echo -e "  Market Service: ${GREEN}http://localhost:8004${NC}"
    echo -e "  Fund Intel:     ${GREEN}http://localhost:8005${NC}"
    echo -e "  LLM Service:    ${GREEN}http://localhost:8006${NC}"
    echo ""
    echo -e "${BLUE}数据库：${NC}"
    echo -e "  MySQL:          ${GREEN}localhost:3306${NC}"
    echo -e "  Redis:          ${GREEN}localhost:6379${NC}"
    echo ""
    echo -e "${YELLOW}提示：${NC}运行 ./health-check.sh 进行完整健康检查"
    echo -e "${YELLOW}提示：${NC}运行 ./test-all-apis.sh 进行 API 测试"
}

main() {
    header "FundProphet 微服务启动"
    
    check_docker
    check_env
    check_network
    check_volumes
    
    start_infrastructure
    start_services
    start_scheduler
    
    wait_for_health
    print_status
    print_access_info
    
    header "启动完成"
    info "所有服务已启动完成！"
}

main "$@"