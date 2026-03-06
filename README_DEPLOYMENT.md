# FundProphet 微服务快速启动指南

## 快速开始

### 1. 配置环境变量

```bash
cp .env.example .env
nano .env
```

**必须配置：**
- `DB_ROOT_PASSWORD` - MySQL root 密码
- `DB_PASSWORD` - 应用数据库密码
- `DEEPSEEK_API_KEY` - DeepSeek API 密钥
- `MINIMAX_API_KEY` - MiniMax API 密钥

### 2. 启动服务

```bash
./start-microservices.sh
```

### 3. 验证部署

```bash
# 健康检查
./health-check.sh

# API 测试
./test-all-apis.sh
```

### 4. 访问服务

- **Traefik Dashboard**: http://localhost:8080
- **API Gateway**: http://localhost
- **前端**: http://localhost:5173

## 服务端口

| 服务 | 端口 |
|------|------|
| Traefik | 80, 8080 |
| Stock Service | 8001 |
| Fund Service | 8002 |
| News Service | 8003 |
| Market Service | 8004 |
| Fund Intel Service | 8005 |
| LLM Service | 8006 |
| MySQL | 3306 |
| Redis | 6379 |

## 常用命令

```bash
# 启动服务
./start-microservices.sh

# 停止服务
./stop-microservices.sh

# 重启服务
./restart-microservices.sh

# 查看状态
./status-microservices.sh

# 健康检查
./health-check.sh

# API 测试
./test-all-apis.sh

# 查看日志
docker compose -f docker-compose.microservices.yml logs -f

# 进入容器
docker exec -it fundprophet-stock bash
```

## 故障排查

### 服务启动失败

```bash
# 查看日志
docker compose -f docker-compose.microservices.yml logs <service-name>

# 检查环境变量
cat .env

# 重启服务
./restart-microservices.sh
```

### 数据库连接失败

```bash
# 检查 MySQL 状态
docker compose ps mysql

# 查看 MySQL 日志
docker compose logs mysql

# 等待 MySQL 启动
sleep 60
```

### LLM API 失败

```bash
# 检查 API 密钥
grep DEEPSEEK_API_KEY .env
grep MINIMAX_API_KEY .env

# 查看 LLM 服务日志
docker compose logs llm-service

# 重启 LLM 服务
docker compose restart llm-service
```

## 完整文档

- **部署文档**: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)
- **API 文档**: [docs/API.md](docs/API.md)
- **微服务架构**: [docs/MICROSERVICES.md](docs/MICROSERVICES.md)
- **Docker 文档**: [docs/DOCKER.md](docs/DOCKER.md)
- **故障排查**: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

## 数据备份

```bash
# 备份 MySQL
docker exec fundprophet-mysql mysqldump -uroot -p${DB_ROOT_PASSWORD} trade_cache > backup.sql

# 备份 Redis
docker exec fundprophet-redis redis-cli SAVE
docker cp fundprophet-redis:/data/dump.rdb ./redis_backup.rdb
```

## 卸载

```bash
# 停止并删除容器
./stop-microservices.sh

# 删除数据卷（⚠️ 会删除所有数据）
docker compose -f docker-compose.microservices.yml down -v

# 清理未使用的资源
docker system prune -a
```

## 支持

- GitHub Issues
- 技术文档: `/docs` 目录
- 部署问题: 查看 `docs/DEPLOYMENT.md` 中的故障排查章节

---

**版本**: 1.0.0
**最后更新**: 2026-03-06