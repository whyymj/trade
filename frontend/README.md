# 前端 - Vue3 + Vite

- **Vue 3**（Composition API）
- **Vite** 构建
- **Vue Router** 路由
- **Pinia** 状态管理
- **ECharts** 图表

## 开发

```bash
# 安装依赖
npm install

# 启动开发服务器（默认 http://localhost:5173，会代理 /api 到后端 5050）
npm run dev
```

## 构建

```bash
npm run build
```

产物输出到 `frontend/dist`，由项目根目录的 Flask 服务（`python server.py`）提供；访问 http://localhost:5050 即可使用。

## 目录说明

- `src/main.js` - 入口，挂载 Pinia、Router
- `src/App.vue` - 根组件，`<router-view>`
- `src/router/index.js` - 路由配置，可在此扩展新页面
- `src/stores/` - Pinia stores（`app` 全局状态/提示，`stock` 股票列表与数据）
- `src/api/` - 接口封装
- `src/views/` - 页面组件
- `public/` - 静态资源，构建时原样拷贝到 dist
