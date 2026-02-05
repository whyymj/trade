# 访问一直加载 / 无响应的排查指南

与日志无关时，按下面顺序排查。

---

## 快速排查清单（按顺序执行）

在**本机**或**服务器**上逐条执行，根据结果判断问题点。

```bash
# 1. 后端是否在跑、端口是否通（期望输出 200）
curl -s -o /dev/null -w "%{http_code}\n" http://118.190.155.0:5050/

# 2. 曲线页用到的接口：是否有数据（无数据会 404，异常会 500；502 多为代理超时）
curl -s -o /dev/null -w "%{http_code}\n" "http://118.190.155.0:5050/api/data?file=600519&start=2024-01-01&end=2026-02-04"

# 3. 股票列表接口（期望 200）
curl -s -o /dev/null -w "%{http_code}\n" "http://118.190.155.0:5050/api/list"
```

在**服务器**上（SSH 登录后）：

```bash
cd /home/admin/trade_analysis   # 或你的 REMOTE_PATH

# 4. 容器是否都在跑
docker compose ps

# 5. 应用最近日志（访问一次页面后再看，有无报错、超时）
docker compose logs --tail 100 trade-app

# 6. 若怀疑 MySQL：进入 MySQL 容器测连库与库内是否有数据
docker compose exec trade-mysql mysql -u root -ptrade_secret -e "USE trade_cache; SELECT COUNT(*) FROM stock_daily; SHOW TABLES;"
```

在**浏览器**里：

1. 打开 **http://118.190.155.0:5050/chart?symbol=600519**
2. 按 **F12** → **Network**，刷新
3. 看哪个请求是 **红色**（失败）或一直 **Pending**，记下该请求的 **URL** 和 **状态码**

| 现象 | 可能原因 | 下一步 |
|------|----------|--------|
| 步骤 1 非 200 / 超时 | 后端未起、端口未放行、防火墙 | 检查 `docker compose ps`、安全组 5050 |
| 步骤 2 为 404 | 该股票在库里无数据 | 首页添加该股票并「一键更新」后再试 |
| 步骤 2 为 502 | 代理超时或上游挂掉 | 看 `docker compose logs trade-app`、若有 Nginx 则加大超时 |
| 步骤 2 为 500 | 后端异常 | 看 `docker compose logs trade-app` 里的 traceback |
| Network 里 /api/data 红/502 | 同上 | 同上 + 确保该股票已同步到库 |

---

## 1. 确认是「页面不出现」还是「页面出现但数据一直转圈」

- **页面空白或一直转圈**：可能是首屏 HTML/JS 没返回，或前端加载后调用的第一个接口卡住。
- **页面有框架、但列表/图表不出现**：多半是某个 **API 请求** 一直没返回。

---

## 2. 用浏览器开发者工具看「谁」在卡住

1. 打开页面，按 **F12**（或右键 → 检查）→ 切到 **Network（网络）**。
2. 刷新页面，看请求列表：
   - 若有请求一直处于 **Pending**（一直不结束），记下该请求的 **URL** 和 **Method**。
   - 若 **第一个** 就卡住：常见是 `GET /` 或 `GET /api/xxx`（例如 `/api/list`）。

这样能区分是「静态资源/首页」卡住，还是「某个接口」卡住。

---

## 3. 确认后端是否存活、端口是否正确

在本机或服务器上执行：

```bash
# 本机
curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:5050/

# 若部署在服务器
curl -s -o /dev/null -w "%{http_code}" http://服务器IP:5050/
```

- 返回 **200**：后端在跑，且根路径可访问。
- **无输出 / 超时 / 连接被拒**：后端没起、端口错、或防火墙/安全组没放行 5050。

---

## 4. 直接测卡住的那个接口

若 Network 里看到是某个 API 一直 Pending（例如 `/api/list`），在终端测同一条：

```bash
curl -v "http://127.0.0.1:5050/api/list"
# 或
curl -v "http://服务器IP:5050/api/list"
```

- **很快返回 200 和 JSON**：后端正常，可能是前端环境/代理/跨域问题。
- **一直不返回**：问题在后端或后端依赖（见下）。
- **返回 5xx**：看后端控制台或日志里的报错。

---

## 5. 后端「卡住」的常见原因

### 5.1 数据库连不上或很慢

- 现象：请求到 `/api/list`、`/api/data` 等会查 MySQL 的接口一直不返回。
- 处理：
  - 检查 `config.yaml` 里 `mysql` 的 host/port/user/password 是否正确。
  - Docker 部署时检查 `MYSQL_*` 环境变量、以及应用容器能否访问 MySQL 容器（例如 `trade-mysql` 是否启动、网络是否正常）。
  - 在服务器上测试：`mysql -h <host> -P <port> -u <user> -p` 能否登录。

### 5.2 启动时就卡住（例如建表）

- 现象：`python server.py` 或容器启动后，没有任何请求日志，访问页面也完全无响应。
- 可能：`create_all_tables()` 连接 MySQL 超时或阻塞。
- 处理：先保证 MySQL 可连，再重启后端；看启动时是否有报错或长时间无输出。

### 5.3 某个接口内部很慢（如 akshare、大量查询）

- 现象：只有部分接口（如拉取某只股票、全量同步）一直加载。
- 处理：看后端控制台/日志里该请求是否长时间无新输出；考虑给 akshare/数据库加超时或优化查询。

---

## 6. Docker 部署时的额外检查

```bash
# 容器是否在跑
docker compose ps

# 应用容器日志（看是否有报错、请求是否进来）
docker compose logs -f trade-app

# MySQL 是否健康
docker compose ps trade-mysql
```

若访问的是服务器公网 IP，确认 **安全组/防火墙** 已放行 **5050**。

---

## 7. 前端生产环境是否请求错地址

- 当前前端使用相对路径 `API_BASE = ''`，即请求同源的 `/api/...`；后端在同一台机同一端口提供页面和 API 时，一般不会请求错。
- 若你改成了绝对地址（例如 `API_BASE = 'http://某域名'`），确认该地址在浏览器里可直接访问，且无跨域或证书问题。

---

## 8. 建议的排查顺序小结

| 步骤 | 操作 | 目的 |
|------|------|------|
| 1 | F12 → Network，刷新，看哪个请求 Pending | 确定是页面还是某个 API 卡住 |
| 2 | `curl http://host:5050/` | 确认后端存活、端口正确 |
| 3 | `curl http://host:5050/api/list`（或卡住的那个 URL） | 确认该接口是否在后端侧就卡住 |
| 4 | 看后端控制台 / `docker compose logs` | 看是否有异常、超时、连库失败 |
| 5 | 检查 MySQL 连接与 Docker 网络/环境变量 | 排除数据库导致的阻塞 |

按上述顺序做一遍，通常能定位到是「前端 / 网络 / 后端 / 数据库」哪一环导致一直加载、无返回。

---

## 九、访问 /chart?symbol=xxx 报 502

**502 Bad Gateway** 一般由**反向代理**（如 Nginx）返回，表示上游（本应用）未在约定时间内正常响应或连接异常。

常见原因与处理：

| 原因 | 处理 |
|------|------|
| **上游超时** | 曲线页会请求 `/api/data?file=股票代码`。若该股票在远端库中无数据，后端会尝试用 akshare 拉取，可能较慢或失败。**处理**：先在首页把该股票加入列表并执行「一键更新」，让数据落库后再访问曲线页。 |
| **后端崩溃/未启动** | 在服务器上执行 `docker compose logs -f trade-app` 看是否有报错；`docker compose ps` 确认 trade-app 在运行。 |
| **代理超时时间过短** | 若前面有 Nginx 等，适当调大 `proxy_read_timeout` / `proxy_connect_timeout`。 |

接口侧：无数据时已改为返回 **404**（并带提示文案），不再因未捕获异常导致进程退出；若仍出现 502，请结合代理与容器日志排查。

---

## 十、LSTM 训练/预测相关

| 现象 | 可能原因 | 处理 |
|------|----------|------|
| `/api/lstm/train` 或 `/api/lstm/predict` 返回 **503** | LSTM 模块不可用（未安装 PyTorch、scikit-learn） | 安装依赖：`pip install torch scikit-learn shap`；Docker 镜像已含，若自建环境请检查 `requirements.txt`。 |
| 训练返回 **400**「样本不足，需要至少 65 个交易日数据」 | 该股票在 start～end 范围内不足 65 个交易日 | 扩大日期范围或先对该股票执行「一键更新」再训练。 |
| 预测返回 **404**「未找到已保存的模型」 | 尚未对该环境训练过 LSTM，或模型目录被清空 | 先调用 `POST /api/lstm/train` 完成一次训练；Docker 中未挂载 `analysis_temp` 时重启容器后需重新训练。 |
| 训练/预测很慢或 OOM | PyTorch 默认占内存；数据量大或 SHAP 计算耗时 | 训练时可在 body 中设 `do_shap: false` 减少内存与时间；或缩小日期范围。 |
