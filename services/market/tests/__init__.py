import sys
import os

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from services.market.data.market_crawler import MarketCrawler
from services.market.data.market_repo import MarketRepo
