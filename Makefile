# 简化启动与构建
.PHONY: backend frontend dev build install

backend:
	python server.py

frontend:
	cd frontend && pnpm run dev

build:
	cd frontend && pnpm run build

install:
	pip install -r requirements.txt
	cd frontend && pnpm install

# 开发：需在两个终端分别执行 make backend 与 make frontend
dev: install
	@echo "请在一个终端执行: make backend"
	@echo "在另一个终端执行: make frontend"
	@echo "然后访问 http://localhost:5173"
