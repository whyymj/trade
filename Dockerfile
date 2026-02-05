# 多阶段构建：先构建前端，再组装后端 + 静态资源
# 阶段一：构建前端
FROM node:20-alpine AS frontend-builder
WORKDIR /build

# 先复制依赖清单，安装后再复制源码（便于利用 Docker 层缓存）
# 使用 BuildKit 缓存挂载，pnpm store 跨构建复用，避免重复下载（需 DOCKER_BUILDKIT=1）
# 发布流程默认使用国内 npm 镜像（见 docker-compose.yml build.args）
ARG PNPM_REGISTRY=https://registry.npmmirror.com
COPY frontend/package.json frontend/pnpm-lock.yaml ./
RUN --mount=type=cache,target=/root/.local/share/pnpm/store \
    corepack enable pnpm && pnpm config set registry "$PNPM_REGISTRY" && pnpm install --frozen-lockfile
RUN test -f node_modules/.bin/vite || pnpm add -D vite @vitejs/plugin-vue
COPY frontend/ ./
RUN pnpm run build

# 阶段二：应用镜像
FROM python:3.12-slim
WORKDIR /app

# 系统依赖（ca-certificates 等；若需 matplotlib 显示可加 libpng-dev）
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 安装全部 Python 依赖（含 LSTM：torch、scikit-learn、shap），便于自动部署无需再手动 pip
COPY requirements.txt ./
ARG PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
RUN --mount=type=cache,target=/root/.cache/pip,id=trade-pip-cache \
    pip install -r requirements.txt -i "$PIP_INDEX" \
    && python -c "import torch; import sklearn; import shap; print('LSTM deps OK')"

# 后端与数据层、分析模块
COPY server.py ./
COPY server/ ./server/
COPY data/ ./data/
COPY analysis/ ./analysis/
COPY cleanup_analysis_temp.py ./

# 默认配置（无敏感信息；MySQL 由环境变量 MYSQL_* 提供）
COPY config.docker.yaml ./config.yaml

# 前端构建产物
COPY --from=frontend-builder /build/dist ./frontend/dist

EXPOSE 5050
ENV PORT=5050
ENV FLASK_DEBUG=0

CMD ["python", "server.py"]
