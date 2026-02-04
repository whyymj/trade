#!/usr/bin/env bash
# 将当前项目同步到云服务器并在远端用 Docker 启动（以 admin 用户部署，不操作 root）
# 使用前确保本机可 ssh 到 SERVER，且远端已安装 Docker 与 Docker Compose
# 同步方式：若远端有 rsync 则用 rsync，否则用 tar+ssh（不依赖远端装 rsync）
set -e

SERVER="${SERVER:-admin@118.190.155.0}"
REMOTE_PATH="${REMOTE_PATH:-/home/admin/trade_analysis}"

echo "目标: $SERVER:$REMOTE_PATH"
echo "同步代码..."

if rsync -avz --delete \
  --exclude '.venv' --exclude 'venv' --exclude 'frontend/node_modules' --exclude 'frontend/dist' \
  --exclude '__pycache__' --exclude '.git' --exclude 'output' --exclude '.cursor' \
  --exclude '.DS_Store' --exclude '*.pyc' --exclude 'config.yaml' \
  . "$SERVER:$REMOTE_PATH/" 2>/dev/null; then
  echo "已使用 rsync 同步。"
else
  echo "rsync 不可用（或远端未安装），改用 tar+ssh 同步..."
  ssh "$SERVER" "mkdir -p $REMOTE_PATH && find $REMOTE_PATH -mindepth 1 -delete 2>/dev/null; true"
  # COPYFILE_DISABLE=1 避免 macOS 写入 xattr，导致 Linux 上 tar 报未知扩展头；排除 ._* 资源文件
  (COPYFILE_DISABLE=1 tar --exclude='.venv' --exclude='venv' --exclude='frontend/node_modules' --exclude='frontend/dist' \
    --exclude='__pycache__' --exclude='.git' --exclude='output' --exclude='.cursor' \
    --exclude='.DS_Store' --exclude='*.pyc' --exclude='config.yaml' --exclude='._*' \
    -czf - .) | ssh "$SERVER" "cd $REMOTE_PATH && tar xzf - --overwrite --no-same-owner 2>/dev/null; true"
  echo "已使用 tar+ssh 同步。"
fi

echo "检查远端环境（vite / 前端依赖）..."
ssh "$SERVER" "cd $REMOTE_PATH && if ! test -f frontend/node_modules/.bin/vite; then if command -v node >/dev/null 2>&1; then echo '未检测到 vite，尝试安装前端依赖…'; (cd frontend && pnpm install) || echo '前端依赖安装失败（可能权限不足），将依赖 Docker 构建时在容器内安装'; else echo '未检测到 Node.js，跳过前端依赖安装（Docker 构建时会在容器内安装）'; fi; fi"

echo "远端构建并启动..."
ssh "$SERVER" "cd $REMOTE_PATH && docker compose up -d --build"

echo "部署完成。应用: http://<服务器IP>:5050"
