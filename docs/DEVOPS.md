# FundProphet DevOps 部署文档

## 概述

本文档介绍 FundProphet 微服务架构的部署方案，包括 Docker Compose、CI/CD、日志收集和健康检查。

## 目录结构

```
trade/
├── docker-compose.yml                 # 单体应用部署（原有）
├── docker-compose.microservices.yml   # 微服务部署配置
├── docker-compose.logging.yml         # 日志收集配置
├── .env.example                       # 环境变量模板
├── .github/
│   └── workflows/
│       └── ci.yml                     # CI/CD 配置
├── logging/                           # 日志配置
│   ├── loki-config.yaml              # Loki 配置
│   ├── promtail-config.yaml          # Promtail 配置
│   ├── grafana-datasources.yml       # Grafana 数据源
│   ├── grafana-dashboards.yml        # Grafana 仪表板配置
│   └── dashboards/                   # Grafana 仪表板 JSON
│       └── logs.json                 # 日志监控仪表板
└── HEALTH_CHECKS.md                  # 健康检查端点要求
```

## 微服务架构

```
┌─────────────────────────────────────────────────────────────┐
│                        Traefik (API Gateway)                │
│                      端口: 80, 8080                        │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼────────┐  ┌────────▼────────┐  ┌─────────▼────────┐
│  Stock Service │  │  Fund Service   │  │  News Service    │
│   端口: 8001   │  │   端口: 8002    │  │    端口: 8003    │
└───────┬────────┘  └────────┬────────┘  └─────────┬────────┘
        │                     │                     │
┌───────▼────────┐  ┌────────▼────────┐  ┌─────────▼────────┐
│ Market Service │  │ Fund-Intel      │  │  LLM Service     │
│   端口: 8004   │  │   端口: 8005    │  │    端口: 8006    │
└────────────────┘  └────────┬────────┘  └─────────────────┘
                            │
                   ┌────────▼────────┐
                   │    Scheduler    │
                   │  定时任务服务   │
                   └─────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼────────┐  ┌───────▼────────┐  ┌──────▼────────┐
│   MySQL 8.0    │  │  Redis 7       │  │    Loki       │
│   端口: 3306   │  │  端口: 6379    │  │  日志聚合     │
└────────────────┘  └────────────────┘  └───────┬────────┘
                                              │
                                       ┌──────▼────────┐
                                       │   Grafana     │
                                       │  日志可视化   │
                                       │  端口: 3000   │
                                       └───────────────┘
```

## 快速开始

### 1. 环境准备

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑环境变量（根据实际情况修改）
vim .env
```

### 2. 启动微服务

```bash
# 启动所有服务
docker compose -f docker-compose.microservices.yml up -d

# 查看服务状态
docker compose -f docker-compose.microservices.yml ps

# 查看日志
docker compose -f docker-compose.microservices.yml logs -f
```

### 3. 启动日志收集

```bash
# 启动日志收集服务
docker compose -f docker-compose.logging.yml up -d

# 访问 Grafana
open http://localhost:3000
# 默认用户名: admin
# 默认密码: admin
```

### 4. 访问服务

```bash
# Traefik Dashboard
open http://localhost:8080

# 服务 API (通过 Traefik 路由)
curl http://localhost/api/fund/list
curl http://localhost/api/news/latest
curl http://localhost/api/market/macro
```

## 停止服务

```bash
# 停止微服务
docker compose -f docker-compose.microservices.yml down

# 停止日志服务
docker compose -f docker-compose.logging.yml down

# 停止并删除卷
docker compose -f docker-compose.microservices.yml down -v
```

## CI/CD 流程

### 工作流程

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌──────────┐    ┌─────────┐
│  Push   │ -> │  Lint   │ -> │  Test   │ -> │  Build   │ -> │ Deploy  │
│ to Git  │    │ 代码检查 │    │  单元测试 │    │ 镜像构建 │    │ 部署上线 │
└─────────┘    └─────────┘    └─────────┘    └──────────┘    └─────────┘
```

### 触发条件

- Push 到 `main` 或 `develop` 分支
- Pull Request 到 `main` 分支
- 手动触发

### 环境变量配置

在 GitHub Secrets 中配置：

```
REGISTRY_URL              # Docker Registry 地址
REGISTRY_USERNAME         # Registry 用户名
REGISTRY_PASSWORD         # Registry 密码
DEPLOY_HOST               # 部署服务器地址
DEPLOY_USER               # 部署服务器用户名
DEPLOY_KEY                # 部署服务器 SSH 密钥
```

### 本地测试 CI

```bash
# 安装依赖
pip install ruff pytest pytest-cov

# 运行 lint
ruff check .

# 运行测试
pytest tests/ -v --cov

# 构建镜像
docker build -t fundprophet-stock:latest ./services/stock
```

## 健康检查

### 检查服务健康状态

```bash
# 检查所有服务健康状态
docker compose -f docker-compose.microservices.yml ps

# 手动检查健康端点
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8003/health
curl http://localhost:8004/health
curl http://localhost:8005/health
curl http://localhost:8006/health
```

### 查看服务指标

```bash
curl http://localhost:8001/metrics
curl http://localhost:8002/metrics
```

## 日志查询

### 通过 Grafana 查询

1. 访问 http://localhost:3000
2. 登录后选择 "Explore"
3. 选择 Loki 数据源
4. 输入查询条件：
   ```logql
   {job="fundprophet"}
   {service_name="fundprophet-stock"}
   {job="fundprophet"} | logfmt | level="error"
   ```

### 通过 Promtail 查询

```bash
# 查看 Promtail 日志
docker logs fundprophet-promtail -f
```

### 通过 Docker 查询

```bash
# 查看特定服务日志
docker logs fundprophet-stock -f

# 查看所有服务日志
docker compose -f docker-compose.microservices.yml logs -f
```

## 监控告警

### Grafana 仪表板

预置仪表板包括：
- 日志级别分布
- 服务日志量统计
- 实时日志流

### 自定义告警

在 `logging/loki-config.yaml` 中配置告警规则。

## 故障排查

### 服务无法启动

```bash
# 查看服务日志
docker logs fundprophet-stock

# 检查健康检查
docker inspect fundprophet-stock | grep -A 10 Health

# 进入容器调试
docker exec -it fundprophet-stock /bin/bash
```

### 数据库连接失败

```bash
# 检查 MySQL 状态
docker compose -f docker-compose.microservices.yml ps mysql

# 检查数据库日志
docker logs fundprophet-mysql

# 测试连接
docker exec -it fundprophet-mysql mysql -u funduser -p trade_cache
```

### Redis 连接失败

```bash
# 检查 Redis 状态
docker compose -f docker-compose.microservices.yml ps redis

# 测试连接
docker exec -it fundprophet-redis redis-cli ping
```

### 日志未收集

```bash
# 检查 Promtail 状态
docker logs fundprophet-promtail

# 检查 Loki 状态
docker logs fundprophet-loki

# 验证 Loki API
curl http://localhost:3100/ready
```

## 生产环境部署

### 安全配置

1. 修改默认密码
2. 启用 HTTPS
3. 配置防火墙规则
4. 使用密钥管理服务

### 性能优化

1. 配置资源限制
2. 启用 Redis 持久化
3. 配置 MySQL 主从复制
4. 使用 CDN 加速

### 备份策略

1. 定期备份 MySQL 数据
2. 备份 Redis 数据
3. 备份日志数据

## 维护任务

### 日常维护

```bash
# 清理未使用的镜像
docker image prune -a

# 清理未使用的卷
docker volume prune

# 查看资源使用情况
docker stats
```

### 更新服务

```bash
# 拉取最新代码
git pull

# 重新构建并启动
docker compose -f docker-compose.microservices.yml up -d --build

# 验证更新
curl http://localhost/api/fund/list
```

## 参考文档

- [Docker Compose 文档](https://docs.docker.com/compose/)
- [Traefik 文档](https://doc.traefik.io/traefik/)
- [Loki 文档](https://grafana.com/docs/loki/latest/)
- [Grafana 文档](https://grafana.com/docs/grafana/latest/)
- [GitHub Actions 文档](https://docs.github.com/en/actions)

## 支持

如有问题，请联系 DevOps Agent 或提交 Issue。