from modules.news_classification.analyzer import NewsClassifier, get_classifier
from modules.news_classification.repo import ClassificationRepo, get_repo
from modules.news_classification.interfaces import (
    ClassifiedNews,
    IndustryClassification,
    INDUSTRY_CATEGORIES,
)

__all__ = [
    "NewsClassifier",
    "get_classifier",
    "ClassificationRepo",
    "get_repo",
    "ClassifiedNews",
    "IndustryClassification",
    "INDUSTRY_CATEGORIES",
]
