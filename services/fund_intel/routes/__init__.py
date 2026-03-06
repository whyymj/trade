from flask import Blueprint
from .fund_industry import fund_industry_bp
from .news_classification import news_classification_bp
from .fund_news import fund_news_bp
from .investment_advice import investment_advice_bp

__all__ = [
    "fund_industry_bp",
    "news_classification_bp",
    "fund_news_bp",
    "investment_advice_bp",
]
