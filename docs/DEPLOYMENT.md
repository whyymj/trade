# FundProphet 微服务部署文档

## 概述

FundProphet 采用微服务架构，包含 6 个核心服务、1 个 API 网关、1 个定时任务调度器和必要的基础设施（MySQL、Redis）。所有服务可通过 Docker Compose 一键部署。

---

## 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                       前端 (Vue3 + Nginx)                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   API Gateway (Traefik)                       │
│              端口: 80 (HTTP), 8080 (Dashboard)               │
└──────┬──────────┬──────────┬──────────┬──────────┬───────────┘
       │          │          │          │          │
       ▼          ▼          ▼          ▼          ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐
│ Stock    │ │ Fund     │ │ News     │ │ Market   │ │ Fund Intel   │
│ Service  │ │ Service  │ │ Service  │ │ Service  │ │ Service      │
│ :8001    │ │ :8002    │ │ :8003    │ │ :8004    │ │ :8005        │
└──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────────┘
                                                          │
                                                ┌─────────▼──────────┐
                                                │    LLM Service     │
                                                │       :8006        │
                                                └────────────────────┘
所有服务共享：Redis (端口 6379) + MySQL (端口 3306)
```

---

## 服务列表

| 服务 | 端口 | 职责 | 依赖 |
|------|------|------|------|
| Traefik | 80, 8080 | API 网关、路由、负载均衡 | - |
| Stock Service | 8001 | 股票数据、LSTM 预测 | MySQL, Redis |
| Fund Service | 8002 | 基金数据、净值分析 | MySQL, Redis |
| News Service | 8003 | 新闻爬取、新闻分析 | MySQL, Redis |
| Market Service | 8004 | 市场数据、宏观数据 | MySQL, Redis |
| Fund Intel Service | 8005 | 基金行业分析、新闻分类、投资建议 | MySQL, Redis, Fund, News, LLM |
| LLM Service | 8006 | AI 分析、投资建议生成 | Redis, LLM API |
| Scheduler | - | 定时任务调度 | MySQL, Redis, 各业务服务 |

---

## 前置要求

### 必需软件

- Docker 20.10+
- Docker Compose 2.0+
- 至少 4GB 可用内存
- 至少 20GB 可用磁盘空间

### 可选软件

- jq (JSON 处理，用于健康检查)
- nc (网络连接测试)

### 安装 Docker

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install docker.io docker-compose
sudo usermod -aG docker $USER
```

**macOS:**
下载并安装 [Docker Desktop](https://www.docker.com/products/docker-desktop)

**验证安装:**
```bash
docker --version
docker compose version
```

---

## 部署步骤

### 步骤 1: 克隆代码仓库

```bash
git clone <repository-url>
cd trade
```

### 步骤 2: 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑环境变量
nano .env
```

**必填环境变量:**

```bash
# 数据库配置
DB_ROOT_PASSWORD=your_secure_root_password
DB_USER=funduser
DB_PASSWORD=your_secure_password
DB_NAME=trade_cache

# LLM API 密钥（必须配置，否则 AI 功能不可用）
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
MINIMAX_API_KEY=xxxxxxxxxxxxxxxx

# 日志级别
LOG_LEVEL=INFO
```

**可选环境变量:**

```bash
# 服务间通信 URL（通常不需要修改）
FUND_SERVICE_URL=http://fund-service:8002
STOCK_SERVICE_URL=http://stock-service:8001
NEWS_SERVICE_URL=http://news-service:8003
MARKET_SERVICE_URL=http://market-service:8004
FUND_INTEL_SERVICE_URL=http://fund-intel-service:8005
LLM_SERVICE_URL=http://llm-service:8006
```

### 步骤 3: 启动服务

**方式一：使用启动脚本（推荐）**

```bash
./start-microservices.sh
```

**方式二：使用 Docker Compose**

```bash
# 启动所有服务
docker compose -f docker-compose.microservices.yml up -d

# 查看启动日志
docker compose -f docker-compose.microservices.yml logs -f
```

### 步骤 4: 验证部署

**运行健康检查:**

```bash
./health-check.sh
```

**测试所有 API:**

```bash
./test-all-apis.sh
```

**访问 Traefik Dashboard:**

打开浏览器访问 http://localhost:8080

---

## 常用操作

### 启动服务

```bash
./start-microservices.sh
```

### 停止服务

```bash
./stop-microservices.sh
```

### 重启服务

```bash
# 重启所有服务
./restart-microservices.sh

# 重启指定服务
./restart-microservices.sh stock-service
```

### 查看服务状态

```bash
./status-microservices.sh
```

### 查看日志

```bash
# 查看所有服务日志
docker compose -f docker-compose.microservices.yml logs -f

# 查看指定服务日志
docker compose -f docker-compose.microservices.yml logs -f stock-service

# 查看最近 50 行日志
docker compose -f docker-compose.microservices.yml logs --tail=50 stock-service
```

### 进入容器

```bash
# 进入 Stock Service 容器
docker exec -it fundprophet-stock bash

# 进入 MySQL 容器
docker exec -it fundprophet-mysql mysql -uroot -p

# 进入 Redis 容器
docker exec -it fundprophet-redis redis-cli
```

### 重新构建镜像

```bash
# 重新构建并启动所有服务
docker compose -f docker-compose.microservices.yml up -d --build

# 重新构建指定服务
docker compose -f docker-compose.microservices.yml up -d --build stock-service
```

### 清理资源

```bash
# 停止并删除容器、网络
docker compose -f docker-compose.microservices.yml down

# 停止并删除容器、网络、数据卷（⚠️ 会删除所有数据）
docker compose -f docker-compose.microservices.yml down -v

# 清理未使用的镜像和容器
docker system prune -a
```

---

## API 访问示例

### 通过 API 网关访问

所有 API 通过 Traefik 统一入口访问（推荐生产环境使用）：

```bash
# 股票列表
curl http://localhost/api/stock/list

# 基金列表
curl http://localhost/api/fund/list

# 最新新闻
curl http://localhost/api/news/latest

# 市场宏观数据
curl http://localhost/api/market/macro

# 基金行业分析
curl http://localhost/api/fund-industry/analyze/000001

# 投资建议
curl http://localhost/api/investment-advice/000001
```

### 直接访问服务端口（用于开发调试）

```bash
# Stock Service
curl http://localhost:8001/api/stock/list
curl http://localhost:8001/api/stock/data?symbol=000001
curl http://localhost:8001/api/lstm/predict?symbol=000001

# Fund Service
curl http://localhost:8002/api/fund/list
curl http://localhost:8002/api/fund/000001
curl http://localhost:8002/api/fund/nav/000001

# News Service
curl http://localhost:8003/api/news/latest
curl http://localhost:8003/api/news/sync

# Market Service
curl http://localhost:8004/api/market/macro
curl http://localhost:8004/api/market/sync

# Fund Intel Service
curl http://localhost:8005/api/fund-industry/stats
curl http://localhost:8005/api/news-classification/stats
curl http://localhost:8005/api/fund-news/list

# LLM Service
curl http://localhost:8006/api/llm/chat
```

---

## 数据持久化

### MySQL 数据

MySQL 数据存储在 Docker Volume `fundprophet-mysql-data` 中，容器重启后数据不会丢失。

**备份 MySQL 数据:**

```bash
docker exec fundprophet-mysql mysqldump -uroot -p${DB_ROOT_PASSWORD} trade_cache > backup_$(date +%Y%m%d).sql
```

**恢复 MySQL 数据:**

```bash
docker exec -i fundprophet-mysql mysql -uroot -p${DB_ROOT_PASSWORD} trade_cache < backup_20240101.sql
```

### Redis 数据

Redis 数据存储在 Docker Volume `fundprophet-redis-data` 中，容器重启后数据不会丢失。

**导出 Redis 数据:**

```bash
docker exec fundprophet-redis redis-cli SAVE
docker cp fundprophet-redis:/data/dump.rdb ./redis_backup.rdb
```

**恢复 Redis 数据:**

```bash
docker cp ./redis_backup.rdb fundprophet-redis:/data/dump.rdb
docker restart fundprophet-redis
```

---

## 故障排查

### 问题 1: 服务启动失败

**症状:**
```bash
docker compose ps
# 显示服务为 Exit 状态
```

**排查步骤:**

1. 查看服务日志：
```bash
docker compose logs <service-name>
```

2. 常见原因：
- 环境变量未正确配置
- 端口被占用
- 依赖服务未启动

**解决方案:**
```bash
# 检查环境变量
cat .env

# 检查端口占用
lsof -i :8001

# 重启服务
./restart-microservices.sh
```

### 问题 2: 数据库连接失败

**症状:**
```
Can't connect to MySQL server on 'mysql' (111)
```

**排查步骤:**

1. 检查 MySQL 容器状态：
```bash
docker compose ps mysql
```

2. 查看 MySQL 日志：
```bash
docker compose logs mysql
```

3. 测试 MySQL 连接：
```bash
docker exec fundprophet-mysql mysqladmin ping -h localhost -uroot -p${DB_ROOT_PASSWORD}
```

**解决方案:**
```bash
# 等待 MySQL 完全启动（首次启动需要 30-60 秒）
sleep 60

# 检查环境变量中的数据库配置
grep DB_ .env

# 重新创建 MySQL 容器
docker compose down mysql
docker compose up -d mysql
```

### 问题 3: Redis 连接失败

**症状:**
```
Error 111 connecting to redis:6379. Connection refused.
```

**排查步骤:**

1. 检查 Redis 容器状态：
```bash
docker compose ps redis
```

2. 测试 Redis 连接：
```bash
docker exec fundprophet-redis redis-cli ping
```

**解决方案:**
```bash
# 重启 Redis
docker compose restart redis

# 查看 Redis 日志
docker compose logs redis
```

### 问题 4: LLM API 调用失败

**症状:**
```
Error: DEEPSEEK_API_KEY not configured
```

**排查步骤:**

1. 检查 LLM Service 日志：
```bash
docker compose logs llm-service
```

2. 验证 API 密钥：
```bash
grep DEEPSEEK_API_KEY .env
```

3. 测试 API 连接：
```bash
curl -X POST http://localhost:8006/api/llm/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "test"}'
```

**解决方案:**
```bash
# 重新配置环境变量
nano .env

# 重启 LLM Service
docker compose restart llm-service
```

### 问题 5: 健康检查失败

**症状:**
```bash
./health-check.sh
# 显示服务不健康
```

**排查步骤:**

1. 手动测试健康检查端点：
```bash
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8003/health
curl http://localhost:8004/health
curl http://localhost:8005/health
curl http://localhost:8006/health
```

2. 查看服务日志：
```bash
docker compose logs <service-name>
```

**解决方案:**
```bash
# 重启不健康的服务
docker compose restart <service-name>

# 如果问题持续，查看详细日志
docker compose logs -f <service-name>
```

### 问题 6: 内存不足

**症状:**
```
Cannot allocate memory
OOMKilled
```

**排查步骤:**

1. 查看容器资源使用：
```bash
docker stats
```

2. 检查系统内存：
```bash
free -h
```

**解决方案:**
```bash
# 减少并发服务数量，分批启动
docker compose up -d redis mysql traefik
sleep 30
docker compose up -d llm-service news-service market-service
sleep 10
docker compose up -d stock-service fund-service fund-intel-service

# 或者增加系统交换空间
sudo dd if=/dev/zero of=/swapfile bs=1G count=4
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 问题 7: 端口冲突

**症状:**
```
Bind for 0.0.0.0:8001 failed: port is already allocated
```

**排查步骤:**

1. 查看端口占用：
```bash
lsof -i :8001
netstat -tulpn | grep 8001
```

**解决方案:**

**选项 1: 停止占用端口的进程**
```bash
# 找到进程 ID
lsof -i :8001

# 杀死进程
kill -9 <PID>
```

**选项 2: 修改服务端口**
编辑 `docker-compose.microservices.yml`，修改端口映射：
```yaml
ports:
  - "8101:8001"  # 宿主机端口改为 8101
```

### 问题 8: 数据卷权限问题

**症状:**
```
Permission denied: './data'
```

**排查步骤:**

1. 查看数据卷权限：
```bash
docker volume inspect fundprophet-mysql-data
docker volume inspect fundprophet-redis-data
```

**解决方案:**
```bash
# 删除并重新创建数据卷
docker compose down -v
docker compose up -d

# 注意：此操作会删除所有数据，请先备份
```

---

## 监控和日志

### 查看容器资源使用

```bash
# 实时监控
docker stats

# 单次查看
docker compose ps
```

### 查看服务日志

```bash
# 查看所有服务日志
docker compose -f docker-compose.microservices.yml logs -f

# 查看特定服务日志
docker compose -f docker-compose.microservices.yml logs -f stock-service

# 查看最近 100 行日志
docker compose -f docker-compose.microservices.yml logs --tail=100 fund-service
```

### 导出日志

```bash
# 导出所有日志
docker compose -f docker-compose.microservices.yml logs > all_logs.txt

# 导出特定服务日志
docker compose -f docker-compose.microservices.yml logs stock-service > stock_logs.txt
```

### 访问 Traefik Dashboard

打开浏览器访问 http://localhost:8080 可以查看：
- 所有已注册的路由
- 服务健康状况
- 请求统计

---

## 性能优化

### 1. 调整 MySQL 配置

创建 `mysql.cnf` 并挂载到容器：
```ini
[mysqld]
max_connections = 200
innodb_buffer_pool_size = 1G
query_cache_size = 128M
```

在 `docker-compose.microservices.yml` 中添加：
```yaml
mysql:
  volumes:
    - ./mysql.cnf:/etc/mysql/conf.d/custom.cnf
```

### 2. 调整 Redis 配置

在 `docker-compose.microservices.yml` 中添加 Redis 配置：
```yaml
redis:
  command: redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru
```

### 3. 启用服务副本

修改 `docker-compose.microservices.yml`，为无状态服务启用多个副本：
```yaml
deploy:
  replicas: 3
```

### 4. 使用外部 MySQL/Redis

如果已有外部数据库，修改 `.env` 中的连接信息，并在 `docker-compose.microservices.yml` 中移除 MySQL 和 Redis 服务。

---

## 生产环境部署建议

### 1. 使用 Docker Registry

```bash
# 构建并推送镜像
docker compose -f docker-compose.microservices.yml build
docker tag fundprophet-stock registry.example.com/fundprophet-stock:latest
docker push registry.example.com/fundprophet-stock:latest
```

### 2. 使用环境变量管理敏感信息

不要将 `.env` 文件提交到版本控制。使用 Docker Secrets 或外部密钥管理服务。

### 3. 启用 HTTPS

在 Traefik 中配置 SSL 证书：
```yaml
traefik:
  command:
    - --certificatesresolvers.myresolver.acme.tlschallenge=true
    - --certificatesresolvers.myresolver.acme.email=your-email@example.com
  labels:
    - "traefik.http.routers.stock.tls=true"
    - "traefik.http.routers.stock.tls.certresolver=myresolver"
```

### 4. 定期备份数据

设置定时任务备份 MySQL 和 Redis 数据：
```bash
# 每天凌晨 2 点备份
0 2 * * * /path/to/backup.sh
```

### 5. 监控告警

集成 Prometheus + Grafana 进行监控：
```bash
docker compose -f docker-compose.logging.yml up -d
```

访问 Grafana: http://localhost:3000 (admin/admin)

---

## 安全建议

1. **修改默认密码**: 确保 `.env` 中的数据库密码不是默认值
2. **限制网络访问**: 使用防火墙限制对外暴露的端口
3. **定期更新**: 定期更新 Docker 镜像和依赖包
4. **日志脱敏**: 确保日志中不包含敏感信息
5. **使用非 root 用户**: 容器内使用非 root 用户运行服务

---

## 卸载

```bash
# 停止并删除所有容器、网络
docker compose -f docker-compose.microservices.yml down

# 删除数据卷（⚠️ 会删除所有数据）
docker compose -f docker-compose.microservices.yml down -v

# 删除镜像
docker rmi $(docker images | grep fundprophet | awk '{print $3}')

# 删除未使用的资源
docker system prune -a
```

---

## 支持和联系

- 文档: `/docs` 目录
- 问题反馈: GitHub Issues
- 技术支持: [your-email@example.com]

---

## 附录

### A. 完整 API 列表

详见 `/docs/API.md`

### B. 服务依赖关系

```
Fund Intel Service
  ├─> Fund Service
  ├─> News Service
  └─> LLM Service

所有业务服务
  ├─> MySQL
  └─> Redis

Scheduler
  ├─> Stock Service
  ├─> Fund Service
  ├─> News Service
  └─> Market Service
```

### C. 端口分配

| 服务 | 内部端口 | 外部端口 | 说明 |
|------|---------|---------|------|
| Traefik | 80 | 80 | HTTP 入口 |
| Traefik Dashboard | 8080 | 8080 | 管理界面 |
| Stock Service | 8001 | 8001 | 股票 API |
| Fund Service | 8002 | 8002 | 基金 API |
| News Service | 8003 | 8003 | 新闻 API |
| Market Service | 8004 | 8004 | 市场 API |
| Fund Intel Service | 8005 | 8005 | 智能分析 API |
| LLM Service | 8006 | 8006 | LLM API |
| MySQL | 3306 | 3306 | 数据库 |
| Redis | 6379 | 6379 | 缓存 |

---

**最后更新**: 2026-03-06
**版本**: 1.0.0