# 股票数据与展示

前后分离：后端 Flask API + 前端 Vue3（Vite），行情数据存 MySQL，数据抓取与管理仅通过前端页面完成。

## 主要功能

- **数据展示**：前端选择股票，ECharts 展示日线（开高低收、成交量等）。
- **数据管理**：通过前端添加股票、一键更新全部、全量同步、移除股票；配置日期范围、复权方式、股票列表等。
- **数据存储**：MySQL（表 `stock_meta`、`stock_daily`），支持按日增量更新。
- **分析模块**：离线分析包（时域、频域、ARIMA、形态相似度、复杂度、**LSTM 深度学习预测**等），可对本地 CSV 或导出数据生成报告；LSTM 支持未来 5 日方向与涨跌幅预测、交叉验证、SHAP 可解释性。

**完整说明**：见 [docs/PROJECT.md](docs/PROJECT.md)；接口细节见 [docs/API.md](docs/API.md)。

## 项目结构

```
trade/
├── server.py            # 后端统一启动入口
├── config.yaml          # 数据与股票配置
├── data/                # MySQL 连接与仓储（schema、stock_repo）
├── output/              # 分析报告输出
├── docs/                # API 等文档
├── server/              # 后端包
│   ├── app.py           # Flask 应用工厂
│   ├── utils.py         # 配置、拉取、DB 存储等工具
│   └── routes/          # 路由（API 等）
├── analysis/            # 股票分析模块（时域、频域、ARIMA 等）
└── frontend/            # 前端 Vue3 + Vite
    ├── src/
    └── dist/            # 构建产物（pnpm run build）
```

## 开发方式

### 本地开发（前后端分离）

- **后端**（API，默认 5050）  
  ```bash
  python server.py
  ```
  或：`make backend`

- **前端**（热更新，默认 5173，Vite 将 `/api` 代理到 5050）  
  ```bash
  cd frontend && pnpm install && pnpm run dev
  ```
  或：`make frontend`  
  浏览器访问：http://localhost:5173

需在两个终端分别启动后端与前端；一键安装依赖：`make install`（执行 `pip install -r requirements.txt` 与 `cd frontend && pnpm install`）。

### 本地生产（前后端一体）

1. 构建前端：`cd frontend && pnpm run build` 或 `make build`
2. 启动后端：`python server.py` 或 `make backend`  
   访问 http://localhost:5050，后端同时提供 API 与静态资源。

### 配置与数据

- **config.yaml**（项目根目录）：配置日期范围、复权方式、股票列表、MySQL 连接等；数据抓取与同步仅通过前端页面调用 API 完成。
- **MySQL**：需先创建数据库（如 `CREATE DATABASE trade_cache DEFAULT CHARSET utf8mb4;`），在 `config.yaml` 的 `mysql` 中填写连接信息。启动后端时会自动创建业务表。

---

## 部署方案（Docker）

使用 Docker Compose 一键启动应用与 MySQL：

```bash
docker compose up -d --build
```

- 应用访问：http://localhost:5050  
- MySQL 连接由环境变量提供（见 `docker-compose.yml` 中 `MYSQL_HOST`、`MYSQL_PASSWORD` 等），无需在镜像内写入密码。  
- 生产环境请修改 `docker-compose.yml` 中的 `MYSQL_PASSWORD` 与 MySQL 的 root 密码。  
- 可选：挂载宿主机 `config.yaml` 以持久化股票列表等（在 `docker-compose.yml` 的 `trade-app` 的 `volumes` 中取消注释）。

详细说明（架构、环境变量、挂载、仅跑应用等）：见 [docs/DOCKER.md](docs/DOCKER.md)。

---

## 环境与依赖

- **Python 3.10+**：见 `requirements.txt`（Flask、akshare、pandas、PyMySQL、PyYAML、**torch**、scikit-learn、shap 等；LSTM 模块依赖 PyTorch）。
- **MySQL**：需先创建数据库（如 `trade_cache`），在 `config.yaml` 的 `mysql` 中配置连接（Docker 部署时由环境变量覆盖）。
- **Node 18+、pnpm**：前端开发与构建，`cd frontend && pnpm install`。
