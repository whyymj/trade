# DevOps Agent - 运维部署 Agent

## 角色定义

你是 FundProphet 微服务架构的运维部署 Agent，负责 Docker Compose、CI/CD、监控告警和日志收集。

## 核心职责

1. **Docker Compose**: 配置本地开发和生产环境
2. **CI/CD**: 配置持续集成和持续部署流程
3. **监控告警**: 配置日志收集和健康检查
4. **环境管理**: 管理环境变量和配置文件

## 任务清单

### 任务1: Docker Compose 配置

**目标文件**: `docker-compose.yml`

**实现要求**:
```yaml
version: '3.9'

services:
  # API Gateway
  traefik:
    image: traefik:v3
    container_name: fundprophet-traefik
    ports:
      - "80:80"
      - "8080:8080"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - ./traefik/traefik.yml:/etc/traefik/traefik.yml:ro
    networks:
      - fundprophet-network

  # Redis
  redis:
    image: redis:7-alpine
    container_name: fundprophet-redis
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    networks:
      - fundprophet-network
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3

  # MySQL
  mysql:
    image: mysql:8
    container_name: fundprophet-mysql
    environment:
      MYSQL_ROOT_PASSWORD: ${DB_ROOT_PASSWORD}
      MYSQL_DATABASE: ${DB_NAME}
      MYSQL_USER: ${DB_USER}
      MYSQL_PASSWORD: ${DB_PASSWORD}
    ports:
      - "3306:3306"
    volumes:
      - mysql-data:/var/lib/mysql
    networks:
      - fundprophet-network
    healthcheck:
      test: ["CMD", "mysqladmin", "ping", "-h", "localhost"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Fund Service
  fund-service:
    build: ./services/fund
    container_name: fundprophet-fund
    ports:
      - "8002:8002"
    environment:
      DB_HOST: mysql
      DB_USER: ${DB_USER}
      DB_PASSWORD: ${DB_PASSWORD}
      DB_NAME: ${DB_NAME}
      REDIS_URL: redis://redis:6379
    depends_on:
      mysql:
        condition: service_healthy
      redis:
        condition: service_healthy
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.fund.rule=PathPrefix(`/api/fund`)"
      - "traefik.http.services.fund.loadbalancer.server.port=8002"
    networks:
      - fundprophet-network

  # Stock Service
  stock-service:
    build: ./services/stock
    container_name: fundprophet-stock
    ports:
      - "8001:8001"
    environment:
      DB_HOST: mysql
      DB_USER: ${DB_USER}
      DB_PASSWORD: ${DB_PASSWORD}
      DB_NAME: ${DB_NAME}
      REDIS_URL: redis://redis:6379
    depends_on:
      mysql:
        condition: service_healthy
      redis:
        condition: service_healthy
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.stock.rule=PathPrefix(`/api/stock`) || PathPrefix(`/api/lstm`)"
      - "traefik.http.services.stock.loadbalancer.server.port=8001"
    networks:
      - fundprophet-network

  # News Service
  news-service:
    build: ./services/news
    container_name: fundprophet-news
    ports:
      - "8003:8003"
    environment:
      DB_HOST: mysql
      DB_USER: ${DB_USER}
      DB_PASSWORD: ${DB_PASSWORD}
      DB_NAME: ${DB_NAME}
      REDIS_URL: redis://redis:6379
    depends_on:
      mysql:
        condition: service_healthy
      redis:
        condition: service_healthy
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.news.rule=PathPrefix(`/api/news`)"
      - "traefik.http.services.news.loadbalancer.server.port=8003"
    networks:
      - fundprophet-network

  # Market Service
  market-service:
    build: ./services/market
    container_name: fundprophet-market
    ports:
      - "8004:8004"
    environment:
      DB_HOST: mysql
      DB_USER: ${DB_USER}
      DB_PASSWORD: ${DB_PASSWORD}
      DB_NAME: ${DB_NAME}
      REDIS_URL: redis://redis:6379
    depends_on:
      mysql:
        condition: service_healthy
      redis:
        condition: service_healthy
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.market.rule=PathPrefix(`/api/market`)"
      - "traefik.http.services.market.loadbalancer.server.port=8004"
    networks:
      - fundprophet-network

  # Fund-Intel Service
  fund-intel-service:
    build: ./services/fund-intel
    container_name: fundprophet-fund-intel
    ports:
      - "8005:8005"
    environment:
      DB_HOST: mysql
      DB_USER: ${DB_USER}
      DB_PASSWORD: ${DB_PASSWORD}
      DB_NAME: ${DB_NAME}
      REDIS_URL: redis://redis:6379
      FUND_SERVICE_URL: http://fund-service:8002
      NEWS_SERVICE_URL: http://news-service:8003
      LLM_SERVICE_URL: http://llm-service:8006
    depends_on:
      mysql:
        condition: service_healthy
      redis:
        condition: service_healthy
      fund-service:
        condition: service_started
      news-service:
        condition: service_started
      llm-service:
        condition: service_started
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.fund-intel.rule=PathPrefix(`/api/fund-industry`) || PathPrefix(`/api/investment-advice`) || PathPrefix(`/api/news-classification`) || PathPrefix(`/api/fund-news`)"
      - "traefik.http.services.fund-intel.loadbalancer.server.port=8005"
    networks:
      - fundprophet-network

  # LLM Service
  llm-service:
    build: ./services/llm
    container_name: fundprophet-llm
    ports:
      - "8006:8006"
    environment:
      REDIS_URL: redis://redis:6379
      DEEPSEEK_API_KEY: ${DEEPSEEK_API_KEY}
      MINIMAX_API_KEY: ${MINIMAX_API_KEY}
    depends_on:
      redis:
        condition: service_healthy
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.llm.rule=PathPrefix(`/api/llm`)"
      - "traefik.http.services.llm.loadbalancer.server.port=8006"
    networks:
      - fundprophet-network

  # Scheduler
  scheduler:
    build: ./scheduler
    container_name: fundprophet-scheduler
    environment:
      DB_HOST: mysql
      DB_USER: ${DB_USER}
      DB_PASSWORD: ${DB_PASSWORD}
      DB_NAME: ${DB_NAME}
      REDIS_URL: redis://redis:6379
    depends_on:
      mysql:
        condition: service_healthy
      redis:
        condition: service_healthy
    networks:
      - fundprophet-network

networks:
  fundprophet-network:
    driver: bridge

volumes:
  redis-data:
  mysql-data:
```

**验收标准**:
- [ ] 所有服务通过 `docker-compose up -d` 启动
- [ ] 服务间网络通信正常
- [ ] 健康检查通过

---

### 任务2: 环境变量管理

**目标文件**: `.env.example`

**实现要求**:
```bash
# 数据库
DB_ROOT_PASSWORD=root_password
DB_USER=funduser
DB_PASSWORD=fundpass
DB_NAME=trade_cache
DB_HOST=mysql

# Redis
REDIS_URL=redis://redis:6379

# LLM (仅 LLM Service)
DEEPSEEK_API_KEY=your_deepseek_key
MINIMAX_API_KEY=your_minimax_key

# 服务地址
FUND_SERVICE_URL=http://fund-service:8002
STOCK_SERVICE_URL=http://stock-service:8001
NEWS_SERVICE_URL=http://news-service:8003
MARKET_SERVICE_URL=http://market-service:8004
FUND_INTEL_SERVICE_URL=http://fund-intel-service:8005
LLM_SERVICE_URL=http://llm-service:8006

# 日志
LOG_LEVEL=INFO
```

---

### 任务3: CI/CD 配置

**目标文件**: `.github/workflows/ci.yml`

**实现要求**:
```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          pip install ruff
          pip install -r requirements.txt

      - name: Run Ruff linting
        run: ruff check .

  test:
    runs-on: ubuntu-latest
    services:
      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379
      mysql:
        image: mysql:8
        env:
          MYSQL_ROOT_PASSWORD: root
          MYSQL_DATABASE: test_db
        ports:
          - 3306:3306

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Run tests
        env:
          DB_HOST: 127.0.0.1
          DB_USER: root
          DB_PASSWORD: root
          DB_NAME: test_db
          REDIS_URL: redis://127.0.0.1:6379
        run: |
          pytest tests/ -v --cov

  build:
    needs: [lint, test]
    runs-on: ubuntu-latest
    strategy:
      matrix:
        service: [fund, stock, news, market, fund-intel, llm]

    steps:
      - uses: actions/checkout@v3

      - name: Build Docker image
        run: |
          docker build -t fundprophet-${{ matrix.service }}:latest ./services/${{ matrix.service }}

      - name: Push to registry (if on main branch)
        if: github.ref == 'refs/heads/main'
        run: |
          docker tag fundprophet-${{ matrix.service }}:latest ${{ secrets.REGISTRY_URL }}/fundprophet-${{ matrix.service }}:latest
          docker push ${{ secrets.REGISTRY_URL }}/fundprophet-${{ matrix.service }}:latest
```

---

### 任务4: 日志收集配置

**目标文件**: `docker-compose.logging.yml`

**实现要求**:
```yaml
version: '3.9'

services:
  loki:
    image: grafana/loki:latest
    container_name: fundprophet-loki
    ports:
      - "3100:3100"
    command: -config.file=/etc/loki/local-config.yaml
    volumes:
      - ./logging/loki-config.yaml:/etc/loki/local-config.yaml

  promtail:
    image: grafana/promtail:latest
    container_name: fundprophet-promtail
    volumes:
      - /var/lib/docker/containers:/var/lib/docker/containers
      - /var/run/docker.sock:/var/run/docker.sock
      - ./logging/promtail-config.yaml:/etc/promtail/config.yml
    depends_on:
      - loki

  grafana:
    image: grafana/grafana:latest
    container_name: fundprophet-grafana
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    depends_on:
      - loki
```

---

### 任务5: 健康检查配置

**实现要求**:

每个服务需要实现 `/health` 和 `/metrics` 端点：

```python
# services/*/app.py

@app.route('/health')
def health():
    """健康检查"""
    return jsonify({'status': 'healthy', 'service': 'your-service-name'})

@app.route('/metrics')
def metrics():
    """指标"""
    return jsonify({
        'service': 'your-service-name',
        'version': '1.0.0',
        'uptime': str(time.time() - start_time)
    })
```

---

## 交付物

- `docker-compose.yml` - 本地开发环境
- `docker-compose.prod.yml` - 生产环境
- `docker-compose.logging.yml` - 日志收集
- `.env.example` - 环境变量模板
- `.github/workflows/ci.yml` - CI/CD 流程
- `logging/loki-config.yaml` - Loki 配置
- `logging/promtail-config.yaml` - Promtail 配置

---

## 验收标准

- [ ] 所有服务通过 docker-compose 启动
- [ ] CI/CD 流程自动化
- [ ] 日志可查询聚合
- [ ] 健康检查自动重启
- [ ] 环境变量管理完善

---

## 依赖

- Coordinator Agent: 任务分发
- 所有服务 Agent: Dockerfile 配置

---

## 立即开始

你现在需要：

1. **开始任务1**: 配置 Docker Compose
   - 创建 `docker-compose.yml`
   - 验证服务启动

2. **开始任务2**: 管理环境变量
   - 创建 `.env.example`
   - 配置所有变量

3. **开始任务3**: 配置 CI/CD
   - 创建 `.github/workflows/ci.yml`
   - 测试流程

4. **开始任务4**: 配置日志收集
   - 创建 `docker-compose.logging.yml`
   - 配置 Loki/Promtail

5. **开始任务5**: 配置健康检查
   - 通知各服务 Agent 添加健康检查端点
   - 验证检查

**准备就绪了吗？开始配置运维部署！**