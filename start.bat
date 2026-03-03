@echo off
REM FundProphet Windows 启动脚本

echo =========================================
echo   FundProphet 基金分析系统
echo =========================================

REM 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found
    exit /b 1
)

REM 检查依赖
echo Checking dependencies...

REM 安装 Python 依赖
echo Installing Python dependencies...
pip install -r requirements.txt -q

REM 前端依赖
if exist "frontend\package.json" (
    echo Installing frontend dependencies...
    cd frontend
    call npm install -q
    cd ..
)

echo Dependencies ready!
echo.

REM 启动选项
if "%1"=="backend" goto backend
if "%1"=="frontend" goto frontend
goto all

:backend
echo Starting backend server...
echo Backend: http://localhost:5050
python server.py
goto end

:frontend
echo Starting frontend dev server...
echo Frontend: http://localhost:5173
cd frontend
call npm run dev
goto end

:all
echo Starting all services...
echo.
echo Starting backend on port 5050...
start "Backend" python server.py

timeout /t 2 /nobreak >nul

echo Starting frontend on port 5173...
cd frontend
start "Frontend" call npm run dev
cd ..

echo.
echo =========================================
echo   Services started successfully!
echo =========================================
echo Backend:  http://localhost:5050
echo Frontend: http://localhost:5173
echo.
echo Press any key to exit
pause >nul

:end
