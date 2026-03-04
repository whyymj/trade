#!/usr/bin/env python3
import signal
import sys


def timeout_handler(sig, frame):
    print("TIMEOUT!", file=sys.stderr)
    sys.exit(1)


signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)

print("1. Starting...", file=sys.stderr)

# 导入必要的模块
from datetime import datetime, timedelta

print("2. datetime imported", file=sys.stderr)

from data.fund_repo import get_fund_nav

print("3. get_fund_nav imported", file=sys.stderr)

import numpy as np

print("4. numpy imported", file=sys.stderr)

import pandas as pd

print("5. pandas imported", file=sys.stderr)

# 现在模拟 train_model 的开始部分
fund_code = "013428"
days = 365

end_date = datetime.now()
start_date = end_date - timedelta(days=days)

print(f"6. Getting nav for {fund_code}", file=sys.stderr)
df = get_fund_nav(fund_code, start_date=start_date.strftime("%Y-%m-%d"))
print(f"7. Got nav: {df.shape if df is not None else None}", file=sys.stderr)

signal.alarm(0)
