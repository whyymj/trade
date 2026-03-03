智能基金走势预测分析应用 - 完整实施Prompt
一、项目概述
请开发一款名为"FundProphet智能基金先知"的Web应用，该应用用于每日自动分析并预测基金走势。项目采用前后端分离架构，后端使用Python FastAPI，前端使用React配合ECharts数据可视化。

二、技术栈要求
2.1 后端技术栈
语言框架：Python 3.9+，FastAPI框架
数据处理：Pandas 1.5+，NumPy
机器学习：Scikit-learn 1.2+，XGBoost，LightGBM
深度学习：PyTorch 2.0+（如使用LSTM/Transformer）
统计模型：Statsmodels（ARIMA、GARCH）
数据库：PostgreSQL 14+（元数据），Redis 6+（缓存）
任务调度：Celery + Redis（定时任务）
数据采集：AkShare（开源财经数据库），Tushare Pro（需Token）
2.2 前端技术栈
框架：React 18+ + TypeScript
UI组件：Ant Design 5+ 或 Material-UI
图表库：ECharts 5+
状态管理：Zustand 或 Redux Toolkit
HTTP客户端：Axios
构建工具：Vite
2.3 部署要求
容器化：Docker + Docker Compose
反向代理：Nginx
云服务：支持阿里云/腾讯云部署
三、数据需求规格
3.1 核心数据源
必须获取以下数据并建立数据pipeline：

基金净值数据（每日更新）：

基金代码、基金名称
交易日期、单位净值、累计净值
日增长率、日涨跌幅
基金规模（可选）
基准指数数据（每日更新）：

沪深300指数（000300）
中证500指数（000905）
创业板指（399006）
上证指数（000001）
宏观经济数据（定期更新）：

国债收益率曲线（1年期、2年期、5年期、10年期）
SHIBOR利率
CPI、PPI月度数据
PMI月度数据
M2货币供应量
3.2 数据存储结构
PostgreSQL表结构设计：

sql
-- 基金基本信息表
CREATE TABLE fund_info (
    fund_code VARCHAR(10) PRIMARY KEY,
    fund_name VARCHAR(100),
    fund_type VARCHAR(50),
    manager VARCHAR(100),
    establishment_date DATE,
    risk_level VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 基金净值历史表
CREATE TABLE fund_nav (
    id SERIAL PRIMARY KEY,
    fund_code VARCHAR(10) REFERENCES fund_info(fund_code),
    nav_date DATE NOT NULL,
    unit_nav DECIMAL(10, 4),
    accum_nav DECIMAL(10, 4),
    daily_return DECIMAL(10, 4),
    UNIQUE(fund_code, nav_date)
);

-- 基准指数数据表
CREATE TABLE index_data (
    id SERIAL PRIMARY KEY,
    index_code VARCHAR(20),
    trade_date DATE NOT NULL,
    open_price DECIMAL(10, 2),
    high_price DECIMAL(10, 2),
    low_price DECIMAL(10, 2),
    close_price DECIMAL(10, 2),
    volume BIGINT,
    UNIQUE(index_code, trade_date)
);

-- 预测结果表
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    fund_code VARCHAR(10) REFERENCES fund_info(fund_code),
    predict_date DATE,
    pred_direction VARCHAR(10),
    pred_return DECIMAL(10, 4),
    confidence DECIMAL(5, 4),
    actual_return DECIMAL(10, 4),
    is_correct BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
3.3 数据采集频率
数据类型	更新频率	采集时间
基金净值	每日	15:30后
指数数据	每日	15:30后
宏观数据	每月/每周	定期
基金持仓	每季度	季报后
四、特征工程规格
4.1 时序特征
实现以下技术指标计算函数：

python
def calculate_ma(df, window):
    """移动平均线：MA5、MA10、MA20、MA60"""
    return df['close'].rolling(window=window).mean()

def calculate_ema(df, window):
    """指数移动平均线"""
    return df['close'].ewm(span=window).mean()

def calculate_macd(df, fast=12, slow=26, signal=9):
    """MACD指标"""
    exp1 = df['close'].ewm(span=fast).mean()
    exp2 = df['close'].ewm(span=slow).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_rsi(df, window=14):
    """RSI相对强弱指标"""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(df, window=20, num_std=2):
    """布林带"""
    ma = df['close'].rolling(window=window).mean()
    std = df['close'].rolling(window=window).std()
    upper = ma + (std * num_std)
    lower = ma - (std * num_std)
    return upper, ma, lower

def calculate_volatility(df, window=20):
    """历史波动率"""
    return df['daily_return'].rolling(window=window).std() * np.sqrt(252)
4.2 衍生特征
python
def create_lag_features(df, lags=[1, 2, 3, 5, 10]):
    """滞后特征"""
    for lag in lags:
        df[f'return_lag_{lag}'] = df['daily_return'].shift(lag)
    return df

def create_rolling_features(df, windows=[5, 10, 20]):
    """滚动统计特征"""
    for window in windows:
        df[f'roll_mean_{window}'] = df['close'].rolling(window=window).mean()
        df[f'roll_std_{window}'] = df['close'].rolling(window=window).std()
        df[f'roll_max_{window}'] = df['close'].rolling(window=window).max()
        df[f'roll_min_{window}'] = df['close'].rolling(window=window).min()
    return df

def create_date_features(df):
    """日期特征"""
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
    return df
五、模型算法规格
5.1 推荐的模型实现
模型1：XGBoost分类模型

python
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report

class FundDirectionPredictor:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

    def prepare_data(self, df):
        """准备训练数据"""
        # 特征列定义
        feature_cols = [
            'ma5', 'ma10', 'ma20', 'ma60',
            'macd', 'signal_line', 'histogram',
            'rsi', 'bb_upper', 'bb_middle', 'bb_lower',
            'volatility', 'volume_ratio',
            'return_lag_1', 'return_lag_2', 'return_lag_3',
            'roll_mean_5', 'roll_std_5',
            'day_of_week', 'month', 'is_month_end'
        ]
        # 目标变量：涨(1)/跌(-1)/震荡(0)
        df['direction'] = df['daily_return'].apply(
            lambda x: 1 if x > 0.005 else (-1 if x < -0.005 else 0)
        )
        return df[feature_cols], df['direction']

    def train(self, X_train, y_train):
        """训练模型"""
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """预测"""
        return self.model.predict(X)
模型2：LSTM深度学习模型

python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 3)  # 3分类

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

def create_sequences(data, seq_length=20):
    """创建时间序列数据"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)
5.2 模型评估指标
python
def evaluate_model(y_true, y_pred):
评估"""
    accuracy = accuracy    """模型_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    return {
        'accuracy': accuracy,
        'report': report
    }

def calculate_directional_accuracy(predictions, actuals):
    """计算方向准确率"""
    correct = sum(1 for p, a in zip(predictions, actuals) if (p > 0 and a > 0) or (p < 0 and a < 0))
    return correct / len(predictions)
5.3 模型版本管理
使用MLflow或DVC进行模型版本管理：

yaml
# model_registry.yaml
models:
  - name: fund_direction_xgb
    versions:
      - version: 1.0
        metrics:
          accuracy: 0.58
          directional_accuracy: 0.62
        created_at: 2024-01-01
      - version: 1.1
        metrics:
          accuracy: 0.61
          directional_accuracy: 0.65
        created_at: 2024-02-01
六、API接口规格
6.1 核心API端点
python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI(title="FundProphet API")

# 数据模型
class FundCode(BaseModel):
    fund_code: str

class PredictionRequest(BaseModel):
    fund_codes: List[str]
    predict_horizon: int = 1  # 预测天数

class PredictionResponse(BaseModel):
    fund_code: str
    fund_name: str
    current_nav: float
    prediction: str  # "上涨" / "下跌" / "震荡"
    confidence: float
    predicted_change: float  # 预测涨跌幅
    features_importance: dict

# API端点
@app.get("/")
async def root():
    return {"message": "FundProphet API", "version": "1.0.0"}

@app.post("/api/v1/fund/info")
async def get_fund_info(fund_code: str):
    """获取基金基本信息"""
    pass

@app.post("/api/v1/fund/history")
async def get_fund_history(fund_code: str, days: int = 365):
    """获取基金历史净值"""
    pass

@app.post("/api/v1/predict")
async def predict(request: PredictionRequest):
    """获取基金预测结果"""
    pass

@app.get("/api/v1/prediction/history/{fund_code}")
async def get_prediction_history(fund_code: str, days: int = 30):
    """获取历史预测及实际对比"""
    pass

@app.get("/api/v1/analysis/{fund_code}")
async def get_fund_analysis(fund_code: str):
    """获取基金综合分析报告"""
    pass
七、前端界面规格
7.1 页面结构
src/
├── pages/
│   ├── Dashboard.vue          # 首页/仪表盘
│   ├── FundList.vue           # 基金列表页
│   ├── FundDetail.vue         # 基金详情页
│   ├── Prediction.vue         # 预测中心
│   ├── Backtest.vue           # 回测分析
│   └── Settings.vue           # 设置页
├── components/
│   ├── FundCard.vue           # 基金卡片组件
│   ├── PredictionChart.vue    # 预测图表组件
│   ├── PerformanceChart.vue   # 业绩图表组件
│   ├── FeatureImportance.vue  # 特征重要性组件
│   └── NavBar.vue             # 导航栏
└── store/
    ├── fundStore.ts           # 基金数据状态管理
    └── predictionStore.ts     # 预测数据状态管理
7.2 核心页面设计
首页仪表盘（Dashboard）：

展示用户关注的基金列表
每只基金显示：代码、名称、最新净值、今日预测、置信度
预测信号颜色标识：红色（上涨）、绿色（下跌）、灰色（震荡）
支持添加/移除关注基金
基金详情页（FundDetail）：

历史净值折线图（支持时间范围选择）
预测走势虚线图（未来N天）
技术指标面板（MA、MACD、RSI、布林带）
特征重要性条形图
预测准确率统计
预测中心（Prediction）：

全部基金预测列表
支持排序筛选（按置信度、按涨跌幅）
批量预测功能
7.3 可视化图表要求
使用ECharts实现以下图表：

javascript
// 历史净值+预测图表示例
const chartOption = {
    title: { text: '基金历史净值及预测' },
    tooltip: { trigger: 'axis' },
    legend: { data: ['历史净值', '预测净值', '置信区间'] },
    xAxis: {
        type: 'category',
        data: dates
    },
    yAxis: {
        type: 'value',
        scale: true
    },
    series: [
        {
            name: '历史净值',
            type: 'line',
            data: historicalNav,
            smooth: true
        },
        {
            name: '预测净值',
            type: 'line',
            data: predictedNav,
            lineStyle: { type: 'dashed' },
            smooth: true
        },
        {
            name: '置信区间',
            type: 'line',
            data: confidenceUpper,
            areaStyle: { opacity: 0.1 },
            lineStyle: { opacity: 0 }
        }
    ],
    dataZoom: [
        { type: 'inside', start: 0, end: 100 },
        { type: 'slider', start: 0, end: 100 }
    ]
};
八、定时任务规格
8.1 Celery任务配置
python
from celery import Celery

app = Celery('fund_prophet')
app.conf.update(
    broker_url='redis://localhost:6379/0',
    result_backend='redis://localhost:6379/1'
)

@app.task
def daily_data_collection():
    """每日数据采集任务"""
    # 1. 采集基金净值
    collect_fund_nav()
    # 2. 采集指数数据
    collect_index_data()
    # 3. 更新宏观数据
    update_macro_data()

@app.task
def daily_feature_update():
    """每日特征更新任务"""
    # 更新所有基金的特征
    update_all_features()

@app.task
def daily_prediction():
    """每日预测任务"""
    # 对关注的基金进行预测
    predict_watched_funds()

@app.task
def model_retraining():
    """模型定期重训练"""
    # 每周/每月重新训练模型
    retrain_models()
8.2 定时配置
python
# Celery Beat 调度配置
beat_schedule = {
    'daily-data-collection': {
        'task': 'tasks.daily_data_collection',
        'schedule': crontab(hour=16, minute=0),  # 每天16:00
    },
    'daily-feature-update': {
        'task': 'tasks.daily_feature_update',
        'schedule': crontab(hour=17, minute=0),  # 每天17:00
    },
    'daily-prediction': {
        'task': 'tasks.daily_prediction',
        'schedule': crontab(hour=18, minute=0),  # 每天18:00
    },
    'weekly-model-retrain': {
        'task': 'tasks.model_retraining',
        'schedule': crontab(hour=2, minute=0, day_of_week=0),  # 每周日凌晨
    },
}
九、部署配置
9.1 Docker配置
dockerfile
# 后端 Dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# 前端 Dockerfile
FROM node:18-alpine as build
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
9.2 Docker Compose配置
yaml
version: '3.8'

services:
  postgres:
    image: postgres:14
    environment:
      POSTGRES_USER: funduser
      POSTGRES_PASSWORD: fundpass
      POSTGRES_DB: fundprophet
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:6
    ports:
      - "6379:6379"

  backend:
    build: ./backend
    ports:
      - "8000:8000"
    depends_on:
      - postgres
      - redis
    environment:
      DATABASE_URL: postgresql://funduser:fundpass@postgres:5432/fundprophet
      REDIS_URL: redis://redis:6379

  frontend:
    build: ./frontend
    ports:
      - "80:80"
    depends_on:
      - backend

volumes:
  postgres_data:
十、测试要求
10.1 单元测试
python
import pytest
from feature_engineering import calculate_ma, calculate_rsi

def test_calculate_ma():
    """测试移动平均计算"""
    import pandas as pd
    df = pd.DataFrame({'close': [1, 2, 3, 4, 5]})
    result = calculate_ma(df, 3)
    assert result.iloc[-1] == 4.0

def test_calculate_rsi():
    """测试RSI计算"""
    import pandas as pd
    df = pd.DataFrame({'close': [100, 102, 101, 103, 105]})
    result = calculate_rsi(df)
    assert result.iloc[-1] > 0 and result.iloc[-1] <= 100
10.2 API测试
python
import requests

def test_get_fund_info():
    """测试获取基金信息API"""
    response = requests.post(
        "http://localhost:8000/api/v1/fund/info",
        json={"fund_code": "001302"}
    )
    assert response.status_code == 200
    assert "fund_name" in response.json()

def test_prediction_endpoint():
    """测试预测API"""
    response = requests.post(
        "http://localhost:8000/api/v1/predict",
        json={"fund_codes": ["001302", "007040"], "predict_horizon": 1}
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
十一、项目目录结构
fund-prophet/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── endpoints/
│   │   │   │   ├── funds.py
│   │   │   │   ├── predictions.py
│   │   │   │   └── analysis.py
│   │   │   └── router.py
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   ├── database.py
│   │   │   └── security.py
│   │   ├── models/
│   │   │   ├── fund.py
│   │   │   ├── prediction.py
│   │   │   └── analysis.py
│   │   ├── schemas/
│   │   │   ├── fund.py
│   │   │   ├── prediction.py
│   │   │   └── analysis.py
│   │   ├── services/
│   │   │   ├── data_collector.py
│   │   │   ├── feature_engineering.py
│   │   │   ├── predictor.py
│   │   │   └── evaluator.py
│   │   ├── tasks/
│   │   │   ├── data_collection.py
│   │   │   ├── prediction.py
│   │   │   └── retraining.py
│   │   └── main.py
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── api/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── store/
│   │   ├── utils/
│   │   ├── App.vue
│   │   └── main.ts
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   └── Dockerfile
├── docker-compose.yml
├── README.md
└── .env.example
十二、开发实施步骤
第一阶段：基础架构（第1-2周）
1.
搭建开发环境，安装配置PostgreSQL、Redis
2.
创建数据库表结构
3.
实现数据采集模块（基金净值、指数数据）
4.
搭建FastAPI基础框架
5.
实现前端基础项目搭建
第二阶段：特征工程（第3-4周）
1.
实现技术指标计算函数（MA、MACD、RSI、布林带）
2.
实现滞后特征和滚动统计特征
3.
实现日期特征
4.
建立特征存储和管理机制
第三阶段：模型开发（第5-8周）
1.
实现XGBoost分类模型
2.
实现模型训练和评估流程
3.
实现模型推理API
4.
实现LSTM深度学习模型（可选）
5.
建立模型版本管理
第四阶段：前端开发（第9-12周）
1.
实现基金列表页面
2.
实现基金详情页面
3.
实现预测结果展示页面
4.
实现ECharts可视化图表
5.
实现响应式设计和移动端适配
第五阶段：系统集成（第13-14周）
1.
配置Celery定时任务
2.
实现每日自动预测流程
3.
前后端联调和API对接
4.
性能优化和压力测试
第六阶段：部署上线（第15-16周）
1.
Docker容器化部署
2.
云服务器部署配置
3.
监控和日志配置
4.
用户测试和Bug修复
5.
正式上线发布
十三、关键注意事项
13.1 数据质量
必须处理缺失值（使用前向填充或线性插值）
必须进行复权处理（考虑分红、配送）
必须处理异常值（使用分位数或标准差方法）
13.2 模型风险
基金预测本质上非常困难，不要过分依赖模型结果
历史业绩不代表未来表现
模型可能失效，需要定期更新和监控
建议结合人工判断和风险控制
13.3 合规要求
不得提供实际的投资建议
必须包含风险提示声明
数据使用需符合数据源许可协议
项目Prompt使用说明
将本Prompt作为完整的项目规格说明书，提供给AI开发人员或AI编程助手。AI应按照本规格逐步实现所有功能模块，并确保最终产品符合上述所有技术要求和功能规格。