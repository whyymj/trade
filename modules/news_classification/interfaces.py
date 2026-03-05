from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional, List


@dataclass
class ClassifiedNews:
    """分类后的新闻"""

    news_id: int
    title: str
    content: str
    source: str
    url: str
    published_at: Optional[datetime]
    news_date: Optional[date]
    original_category: str
    industry: str
    industry_code: str
    confidence: float
    classified_at: datetime


@dataclass
class IndustryClassification:
    """行业分类结果"""

    industry: str
    industry_code: str
    confidence: float
    reasoning: str


INDUSTRY_CATEGORIES = [
    {
        "code": "I001",
        "name": "新能源汽车",
        "keywords": [
            "新能源汽车",
            "电动车",
            "锂电池",
            "比亚迪",
            "特斯拉",
            "宁德时代",
            "动力电池",
            "充电桩",
        ],
    },
    {
        "code": "I002",
        "name": "半导体",
        "keywords": [
            "半导体",
            "芯片",
            "集成电路",
            "光刻机",
            "晶圆",
            "中芯国际",
            "华为芯片",
            "AI芯片",
        ],
    },
    {
        "code": "I003",
        "name": "医药生物",
        "keywords": [
            "医药",
            "生物",
            "疫苗",
            "创新药",
            "医疗器械",
            "中药",
            "恒瑞医药",
            "集采",
        ],
    },
    {
        "code": "I004",
        "name": "食品饮料",
        "keywords": [
            "食品",
            "饮料",
            "白酒",
            "茅台",
            "伊利",
            "海天味业",
            "乳制品",
            "消费",
        ],
    },
    {
        "code": "I005",
        "name": "房地产",
        "keywords": ["房地产", "房价", "地产", "万科", "恒大", "限购", "房贷", "土地"],
    },
    {
        "code": "I006",
        "name": "银行",
        "keywords": [
            "银行",
            "金融",
            "利率",
            "存款",
            "贷款",
            "理财",
            "央行",
            "货币政策",
        ],
    },
    {
        "code": "I007",
        "name": "保险",
        "keywords": ["保险", "保费", "险资", "寿险", "财险", "保险资金"],
    },
    {
        "code": "I008",
        "name": "证券",
        "keywords": ["证券", "券商", "A股", "IPO", "牛市", "印花税", "注册制"],
    },
    {
        "code": "I009",
        "name": "军工",
        "keywords": ["军工", "国防", "航天", "航空", "船舶", "导弹", "军费"],
    },
    {
        "code": "I010",
        "name": "新能源",
        "keywords": ["光伏", "风电", "太阳能", "绿电", "碳中和", "储能", "隆基绿能"],
    },
    {
        "code": "I011",
        "name": "有色金属",
        "keywords": ["有色", "铜", "铝", "黄金", "白银", "稀土", "锂矿"],
    },
    {
        "code": "I012",
        "name": "化工",
        "keywords": ["化工", "石化", "炼油", "化工原料", "万华化学", " PTA"],
    },
    {
        "code": "I013",
        "name": "机械设备",
        "keywords": ["机械", "设备", "工程机械", "工业机器人", "数控机床", "三一重工"],
    },
    {
        "code": "I014",
        "name": "交通运输",
        "keywords": ["航空", "航运", "港口", "铁路", "公路", "物流", "快递"],
    },
    {
        "code": "I015",
        "name": "电子",
        "keywords": ["电子", "消费电子", "手机", "苹果", "华为", "面板", "立讯精密"],
    },
    {
        "code": "I016",
        "name": "计算机",
        "keywords": [
            "计算机",
            "软件",
            "云计算",
            "AI",
            "人工智能",
            "大数据",
            "网络安全",
        ],
    },
    {
        "code": "I017",
        "name": "通信",
        "keywords": ["通信", "5G", "运营商", "中国移动", "华为", "中兴通讯"],
    },
    {
        "code": "I018",
        "name": "传媒",
        "keywords": ["传媒", "影视", "游戏", "广告", "短视频", "字节跳动", "腾讯"],
    },
    {
        "code": "I019",
        "name": "零售",
        "keywords": ["零售", "电商", "拼多多", "京东", "阿里巴巴", "百货", "免税"],
    },
    {
        "code": "I020",
        "name": "宏观",
        "keywords": [
            "GDP",
            "CPI",
            "PMI",
            "社融",
            "M2",
            "财政",
            "降准",
            "降息",
            "美联储",
            "美元指数",
        ],
    },
]


def get_industry_code(industry_name: str) -> str:
    """根据行业名称获取行业代码"""
    for cat in INDUSTRY_CATEGORIES:
        if cat["name"] == industry_name:
            return cat["code"]
    return "I000"


def get_industry_by_code(industry_code: str) -> Optional[dict]:
    """根据行业代码获取行业信息"""
    for cat in INDUSTRY_CATEGORIES:
        if cat["code"] == industry_code:
            return cat
    return None
