#!/bin/bash

# FundProphet 微服务部署脚本
# 用法: ./deploy-microservices.sh [up|down|logs|status|health]

set -e

COMPOSE_FILE="docker-compose.microservices.yml"
LOGGING_FILE="docker-compose.logging.yml"
ENV_FILE=".env"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印信息
info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查环境变量文件
check_env() {
    if [ ! -f "$ENV_FILE" ]; then
        warn "环境变量文件不存在，从模板创建..."
        cp .env.example "$ENV_FILE"
        warn "请编辑 $ENV_FILE 文件并填入正确的配置"
        exit 1
    fi
}

# 启动服务
start_services() {
    check_env
    info "启动微服务..."
    docker compose -f "$COMPOSE_FILE" up -d

    info "等待服务启动..."
    sleep 10

    info "启动日志收集服务..."
    docker compose -f "$LOGGING_FILE" up -d

    info "服务启动完成！"
    info "Traefik Dashboard: http://localhost:8080"
    info "Grafana: http://localhost:3000 (admin/admin)"
    info ""
    print_status
}

# 停止服务
stop_services() {
    info "停止微服务..."
    docker compose -f "$COMPOSE_FILE" down

    info "停止日志服务..."
    docker compose -f "$LOGGING_FILE" down

    info "服务已停止"
}

# 查看日志
view_logs() {
    local service=$1
    if [ -z "$service" ]; then
        info "查看所有服务日志..."
        docker compose -f "$COMPOSE_FILE" logs -f
    else
        info "查看 $service 服务日志..."
        docker compose -f "$COMPOSE_FILE" logs -f "$service"
    fi
}

# 查看状态
print_status() {
    info "服务状态:"
    echo ""
    docker compose -f "$COMPOSE_FILE" ps
    echo ""
    info "日志服务状态:"
    echo ""
    docker compose -f "$LOGGING_FILE" ps
}

# 健康检查
health_check() {
    info "健康检查..."

    services=(
        "stock-service:8001"
        "fund-service:8002"
        "news-service:8003"
        "market-service:8004"
        "fund-intel-service:8005"
        "llm-service:8006"
    )

    for service in "${services[@]}"; do
        name="${service%:*}"
        port="${service#*:}"
        if curl -s -f "http://localhost:$port/health" > /dev/null; then
            echo -e "${GREEN}✓${NC} $name ($port) - 健康"
        else
            echo -e "${RED}✗${NC} $name ($port) - 不健康"
        fi
    done
}

# 重启服务
restart_services() {
    info "重启服务..."
    docker compose -f "$COMPOSE_FILE" restart
    info "服务重启完成"
}

# 重新构建并启动
rebuild_services() {
    check_env
    info "重新构建并启动服务..."
    docker compose -f "$COMPOSE_FILE" up -d --build
    info "构建完成"
    print_status
}

# 清理
clean() {
    warn "清理未使用的资源..."
    docker system prune -f
    info "清理完成"
}

# 显示帮助
show_help() {
    cat << EOF
FundProphet 微服务部署脚本

用法: ./deploy-microservices.sh [命令]

命令:
  up          启动所有服务
  down        停止所有服务
  restart     重启服务
  rebuild     重新构建并启动
  logs [服务] 查看日志（可选指定服务名）
  status      查看服务状态
  health      健康检查
  clean       清理未使用的资源
  help        显示此帮助信息

示例:
  ./deploy-microservices.sh up              # 启动所有服务
  ./deploy-microservices.sh logs fund       # 查看基金服务日志
  ./deploy-microservices.sh health          # 健康检查

EOF
}

# 主函数
main() {
    case "${1:-help}" in
        up)
            start_services
            ;;
        down)
            stop_services
            ;;
        restart)
            restart_services
            ;;
        rebuild)
            rebuild_services
            ;;
        logs)
            view_logs "$2"
            ;;
        status)
            print_status
            ;;
        health)
            health_check
            ;;
        clean)
            clean
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            error "未知命令: $1"
            show_help
            exit 1
            ;;
    esac
}

# 运行主函数
main "$@"