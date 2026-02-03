# 股票数据与展示

前后分离：后端 Flask API + 前端 Vue3（Vite）。数据与配置在项目根目录，关注点分离。

**完整项目说明**：见 [docs/PROJECT.md](docs/PROJECT.md)（结构、技术栈、配置、API 概览、分析模块、扩展说明等）。

## 项目结构

```
trade/
├── server.py            # 后端统一启动入口
├── config.yaml          # 数据与股票配置
├── data/                # 股票 CSV 数据（由 config.output_dir 决定）
├── output/              # 分析报告输出
├── docs/                # API 等文档
├── server/              # 后端包
│   ├── app.py           # Flask 应用工厂
│   ├── utils.py         # 配置、拉取、存储等工具
│   └── routes/          # 路由（API 等）
├── analysis/            # 股票分析模块（时域、频域、ARIMA 等）
├── frontend/            # 前端 Vue3 + Vite
│   ├── src/
│   └── dist/            # 构建产物（npm run build）
└── fetch_stock_data.py  # 命令行：按 config 拉取数据
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

### 数据拉取

按 `config.yaml` 拉取股票数据到 `data/`：  
`python fetch_stock_data.py`

## 环境

- Python：见 `requirements.txt`
- 前端：Node 18+，`cd frontend && npm install`
