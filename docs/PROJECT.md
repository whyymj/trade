# 项目说明文档

## 一、项目概述

本项目为**股票数据与展示**系统：支持从网络拉取股票日线数据、本地存储与配置管理，通过 Web 前端（Vue3 + ECharts）展示价格与成交量曲线，并提供股票分析模块（时域、频域、ARIMA、形态相似度、复杂度等）用于离线分析报告生成。

- **前后分离**：后端 Flask API 提供数据接口，前端 Vue3 + Vite 独立开发与构建，生产时由后端统一提供静态资源。
- **关注点分离**：后端分为应用工厂、路由、工具；前端分为路由、状态、API、视图；配置与数据集中在项目根目录。

---

## 二、技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| 后端 | Python 3 | 运行环境 |
| 后端 | Flask | Web 框架，提供 API 与静态服务 |
| 后端 | akshare | 股票数据拉取 |
| 后端 | pandas / PyYAML | 数据处理与配置 |
| 前端 | Vue 3 | Composition API |
| 前端 | Vite | 构建与开发服务器 |
| 前端 | Vue Router | 路由 |
| 前端 | Pinia | 状态管理 |
| 前端 | ECharts | 图表展示 |
| 分析 | analysis 包 | 时域、频域、ARIMA、形态、复杂度等 |

---

## 三、项目结构

```
trade/
├── server.py              # 后端统一启动入口
├── config.yaml             # 全局配置：日期范围、复权、股票列表、数据目录
├── data/                   # 股票 CSV 数据目录（可由 config.output_dir 配置）
├── output/                 # 分析报告与图表输出目录
├── docs/                   # 文档
│   ├── PROJECT.md          # 本说明文档
│   └── API.md              # 接口与功能说明
├── server/                 # 后端包
│   ├── app.py              # Flask 应用工厂（create_app）
│   ├── utils.py            # 配置、路径、akshare 拉取、CSV 读写、config 写入
│   └── routes/
│       ├── __init__.py
│       └── api.py          # /api 下所有数据接口（list、data、add_stock、update_all、fetch_data）
├── analysis/               # 股票分析模块（可独立运行）
│   ├── __init__.py         # 导出各子模块接口
│   ├── time_domain.py      # 时域与统计、STL 分解等
│   ├── frequency_domain.py # 频域、FFT、小波等
│   ├── arima_model.py      # ARIMA 预测
│   ├── shape_similarity.py # 形态相似度、DTW
│   ├── complexity.py      # 非线性与复杂度分析
│   └── full_report.py      # 综合分析报告生成
├── frontend/               # 前端工程（Vue3 + Vite）
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── public/
│   ├── src/
│   │   ├── main.js         # 入口，挂载 Pinia、Router
│   │   ├── App.vue         # 根组件，router-view
│   │   ├── api/            # 接口封装（如 stock.js）
│   │   ├── router/        # 路由配置
│   │   ├── stores/         # Pinia（app、stock）
│   │   └── views/          # 页面（如 StockChart.vue）
│   └── dist/               # 构建产物（npm run build）
├── fetch_stock_data.py     # 命令行：按 config 拉取数据到 data/
├── Makefile                # 简化命令：backend、frontend、build、install
├── requirements.txt        # Python 依赖
└── README.md               # 快速入门与启动说明
```

---

## 四、前后端职责与分离

### 4.1 后端

- **入口**：`python server.py` 启动 Flask，默认端口 5050。
- **职责**：
  - 提供 `/api/*` 数据接口（列表、按文件/按代码取数、添加股票、一键更新）。
  - 生产模式下提供前端静态资源：服务 `frontend/dist` 下的文件，并对未知路径做 SPA 回退（返回 `index.html`），以支持 Vue Router history 模式。
- **配置与数据**：从项目根目录读取 `config.yaml`；数据目录由 `config.output_dir` 决定，默认为根目录下 `data/`。

### 4.2 前端

- **开发**：在 `frontend/` 下执行 `npm run dev`，默认端口 5173，Vite 将 `/api` 代理到后端 5050。
- **生产**：在 `frontend/` 下执行 `npm run build`，产物输出到 `frontend/dist`，由后端统一提供。
- **职责**：股票选择、代码输入、抓取/一键更新、价格与成交量 ECharts 展示；路由与状态（Pinia）便于扩展新页面。

### 4.3 配置与数据

- **config.yaml**：位于项目根目录，配置日期范围、复权方式、股票代码列表、数据输出目录等；后端与 `fetch_stock_data.py` 均依赖此文件。
- **data/**：存放股票 CSV，文件名形如 `股票名称（YYYYMMDD-YYYYMMDD）.csv`。
- **output/**：存放分析模块生成的报告与图表（如 `full_report` 输出）。

---

## 五、配置说明（config.yaml）

| 配置项 | 说明 | 示例 |
|--------|------|------|
| `start_date` | 开始日期，YYYYMMDD；空则默认近一年 | `""` 或 `"20250101"` |
| `end_date` | 结束日期，YYYYMMDD；空则默认今天 | `""` 或 `"20260128"` |
| `adjust` | 复权：`qfq` 前复权 / `hfq` 后复权 / `""` 不复权 | `qfq` |
| `stocks` | 股票代码列表（A 股 6 位、港股 5 位或 xxxxx.HK） | `["600519", "09678.HK"]` |
| `output_dir` | 数据输出目录，相对项目根或绝对路径 | `data` |

接口中的日期范围与复权方式均由此配置决定；未配置时默认近一年、前复权。

---

## 六、启动与运行

### 6.1 开发环境

1. **后端**（API，端口 5050）  
   ```bash
   python server.py
   ```
   或：`make backend`

2. **前端**（热更新，端口 5173，代理 /api 到 5050）  
   ```bash
   cd frontend && npm install && npm run dev
   ```
   或：`make frontend`  
   浏览器访问：http://localhost:5173

### 6.2 生产环境（前后端一体）

1. 构建前端：  
   ```bash
   cd frontend && npm run build
   ```
   或：`make build`

2. 启动后端：  
   ```bash
   python server.py
   ```
   访问：http://localhost:5050（后端同时提供 API 与前端静态资源）

### 6.3 数据拉取

按 `config.yaml` 中的 `stocks`、日期与复权配置，将数据拉取到 `data/`（或 `output_dir` 指定目录）：  
```bash
python fetch_stock_data.py
```

### 6.4 一键安装依赖

```bash
make install
```
会执行：`pip install -r requirements.txt` 与 `cd frontend && npm install`。

---

## 七、API 概览

所有数据接口均以 `/api` 为前缀，由后端 `server/routes/api.py` 提供。

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/list` | 获取数据目录下股票 CSV 列表（用于下拉等） |
| GET | `/api/data` | 按文件名获取单只股票数据（Query: `file=`） |
| GET | `/api/fetch_data/<stock_code>` | 按股票代码获取数据（优先本地，无则拉取） |
| POST | `/api/update_all` | 一键更新全部（按 config.stocks 拉取并覆盖） |
| POST | `/api/add_stock` | 抓取指定代码并加入 config.stocks（Body: `{"code":"600519"}`） |

请求/响应格式、错误码等详见 **docs/API.md**。

---

## 八、分析模块（analysis）

`analysis` 包提供离线股票数据分析，可独立于 Web 服务运行，结果写入 `output/` 等目录。

| 子模块 | 功能概要 |
|--------|----------|
| `time_domain` | 时域与统计特征、移动平均、最大回撤、STL 分解等 |
| `frequency_domain` | 频域分析、FFT、功率谱、小波变换、主导周期等 |
| `arima_model` | ARIMA 建模、预测、残差与指标 |
| `shape_similarity` | 形态相似度、DTW、曲线对比与模式匹配 |
| `complexity` | 非线性与复杂度（如近似熵等） |
| `full_report` | 综合分析报告生成（可指定 CSV 与输出目录） |

使用示例（在项目根目录执行）：  
```bash
python -m analysis.full_report data/白银有色（20250129-20260129）.csv
```
报告与图表将输出到指定或默认的 `output/` 子目录。各子模块的详细用法见其文件内 `if __name__ == "__main__"` 说明。

---

## 九、环境与依赖

- **Python**：建议 3.10+，依赖见 `requirements.txt`（含 akshare、pandas、Flask、matplotlib、scipy、statsmodels、PyWavelets 等）。
- **Node**：建议 18+，用于前端开发与构建；前端依赖见 `frontend/package.json`（Vue3、Vite、Vue Router、Pinia、ECharts）。

---

## 十、扩展与维护

- **新增 API**：在 `server/routes/api.py` 中为 `api_bp` 增加路由；接口说明可补充到 `docs/API.md`。
- **新增前端页面**：在 `frontend/src/router/index.js` 增加路由，在 `frontend/src/views/` 下添加页面组件；如需新状态可在 `frontend/src/stores/` 增加 Pinia store。
- **修改数据目录**：在 `config.yaml` 中修改 `output_dir`；后端与 `fetch_stock_data.py` 会据此读写数据。
- **分析报告**：可基于 `analysis` 包扩展新分析类型，并在 `full_report` 中挂接；输出目录由调用方或脚本参数指定。

---

文档版本与项目结构以当前代码为准；接口细节以 **docs/API.md** 为准。
