# 股票数据与展示

前后分离：后端 Flask API + 前端 Vue3（Vite），行情数据存 MySQL，数据抓取与管理仅通过前端页面完成。

## 主要功能

- **数据展示**：前端选择股票，ECharts 展示日线（开高低收、成交量等）。
- **数据管理**：通过前端添加股票、一键更新全部、全量同步、移除股票；配置日期范围、复权方式、股票列表等。
- **数据存储**：MySQL（表 `stock_meta`、`stock_daily`），支持按日增量更新。
- **分析模块**：离线分析包（时域、频域、ARIMA、形态相似度、复杂度等），可对本地 CSV 或导出数据生成报告。

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
    └── dist/            # 构建产物（npm run build）
```

## 启动方式

### 开发

- **后端**（API，默认 5050）  
  `python server.py`
- **前端**（热更新，默认 5173，代理 /api 到 5050）  
  `cd frontend && npm install && npm run dev`  
  浏览器访问 http://localhost:5173

### 生产（前后端一体）

1. 构建前端：`cd frontend && npm run build`
2. 启动后端：`python server.py`  
  访问 http://localhost:5050，后端同时提供 API 与静态资源。

### 数据抓取与管理

仅通过前端页面操作：添加股票、一键更新、全量同步等，数据写入 MySQL（config.yaml 中配置 `mysql`）。

## 环境与依赖

- **Python 3.10+**：见 `requirements.txt`（Flask、akshare、pandas、PyMySQL、PyYAML 等）。
- **MySQL**：需先创建数据库（如 `trade_cache`），在 `config.yaml` 的 `mysql` 中配置连接。
- **Node 18+**：前端开发与构建，`cd frontend && npm install`。
