# Docker 部署说明

## 一、项目架构概览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            用户浏览器                                     │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Flask 应用 (端口 5050)                                                   │
│  ├── /api/*     → 数据接口（股票列表、日线、抓取、同步、配置等）              │
│  └── /*         → 前端静态 (frontend/dist，SPA 回退)                       │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
            ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
            │ config.yaml  │  │    MySQL     │  │  akshare     │
            │ (根目录)     │  │ (行情存储)   │  │ (数据源)     │
            └──────────────┘  └──────────────┘  └──────────────┘
```

### 组件说明

| 组件 | 说明 | 技术 |
|------|------|------|
| **后端** | 统一入口 `server.py`，Flask 提供 API 与静态资源 | Python 3.10+、Flask、PyMySQL、akshare、pandas 等 |
| **前端** | 构建产物 `frontend/dist`，由后端托管 | Vue3、Vite、ECharts、Element Plus |
| **配置** | 根目录 `config.yaml`：日期范围、复权、股票列表、MySQL 连接 | YAML |
| **数据库** | MySQL：表 `stock_meta`、`stock_daily`，启动时自动建表 | MySQL 5.7+ / 8.x |
| **分析模块** | `analysis/` 包，可离线跑报告，输出到 `output/` | 独立于 Web，按需使用 |

### 数据流

- 前端页面 → 调用 `/api/*` → 后端读写 MySQL、读写在 `config.yaml`、通过 akshare 拉取行情。
- 生产部署：前端先 `pnpm run build`，后端服务 `frontend/dist`，单进程即可对外提供完整站点。

---

## 二、Docker 部署架构

```
                    docker-compose
┌──────────────────────────────────────────────────────────────────┐
│  trade-app (Flask)          │  trade-mysql (MySQL)               │
│  - 多阶段构建：先构建前端     │  - 镜像 mysql:8.0                 │
│  - 再装 Python 依赖 + 后端   │  - 初始化库 trade_cache            │
│  - 通过 MYSQL_* 环境变量连库  │  - 数据卷持久化                   │
│  - 可选挂载 config.yaml      │                                   │
└──────────────────────────────────────────────────────────────────┘
```

- **镜像**：一个应用镜像内同时包含前端静态与后端代码，无需单独前端容器。
- **数据库**：由 `docker-compose` 启动 MySQL 服务，应用通过环境变量连接（不把密码写进镜像）。
- **配置**：容器内可提供默认 `config.yaml`；MySQL 连接由环境变量覆盖。若需持久化股票列表等，可挂载宿主机 `config.yaml`。

---

## 三、使用方式

### 3.1 构建并启动（含 MySQL）

```bash
# 在项目根目录
docker compose up -d --build
```

- 首次会构建应用镜像（含前端构建），并拉取 MySQL 镜像、创建网络与数据卷。
- 应用访问：<http://localhost:5050>
- MySQL 对外端口可在 `docker-compose.yml` 中按需映射（默认仅容器内访问）。

### 3.2 环境变量（MySQL）

应用通过环境变量覆盖 MySQL 配置（优先于 `config.yaml` 中的 `mysql` 段）：

| 变量 | 说明 | 示例 |
|------|------|------|
| `MYSQL_HOST` | 主机 | `trade-mysql` 或 `127.0.0.1` |
| `MYSQL_PORT` | 端口 | `3306` |
| `MYSQL_USER` | 用户 | `root` |
| `MYSQL_PASSWORD` | 密码 | 与 compose 中一致 |
| `MYSQL_DATABASE` | 数据库名 | `trade_cache` |

在 `docker-compose.yml` 中已为 `trade-app` 配置上述变量，指向同栈中的 `trade-mysql` 服务。

### 3.3 挂载 config.yaml（可选）

若希望持久化并编辑股票列表、日期范围等，可挂载宿主机配置：

```yaml
# docker-compose.yml 中 trade-app 的 volumes 增加：
volumes:
  - ./config.yaml:/app/config.yaml:ro
```

注意：MySQL 连接仍可由环境变量覆盖，挂载的 `config.yaml` 中可保留或省略 `mysql` 段。

### 3.4 仅构建镜像、不启动 MySQL

若使用外部已有 MySQL，只启动应用：

```bash
docker compose run --rm -e MYSQL_HOST=host.docker.internal -e MYSQL_PASSWORD=你的密码 trade-app
```

或先修改 `docker-compose.yml` 中 `trade-app` 的 `MYSQL_*` 与 `depends_on`，再 `docker compose up -d trade-app`。

---

## 四、文件说明

| 文件 | 说明 |
|------|------|
| `Dockerfile` | 多阶段：Stage 1 用 Node 构建 frontend/dist；Stage 2 用 Python 复制后端与 dist，安装依赖，暴露 5050 |
| `docker-compose.yml` | 定义服务 `trade-app`、`trade-mysql`，网络与数据卷，环境变量 |
| `.dockerignore` | 排除 node_modules、__pycache__、.git、output 等，加快构建与减小上下文 |

---

## 五、与开发环境对照

| 项目 | 开发 | Docker 生产 |
|------|------|-------------|
| 后端 | `python server.py` (5050) | 容器内 `python server.py`，端口映射 5050 |
| 前端 | `pnpm run dev` (5173)，代理 /api | 已构建进镜像，由后端直接提供 |
| MySQL | 本机或远程 config.yaml | compose 内 MySQL + 环境变量 |
| config | 根目录 config.yaml | 镜像内默认 + 可选挂载 + 环境变量覆盖 DB |
