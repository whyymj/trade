#!/usr/bin/env python3
import signal
import sys


def timeout_handler(sig, frame):
    print("TIMEOUT!", file=sys.stderr)
    sys.exit(1)


signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)

print("1. Starting...", file=sys.stderr)

from data.fund_repo import get_fund_nav

print("2. Imported get_fund_nav", file=sys.stderr)

df = get_fund_nav("013428")
print(f"3. Got nav: {df.shape}", file=sys.stderr)

from analysis.fund_lstm import prepare_features

print("4. Imported prepare_features", file=sys.stderr)

df = prepare_features(df)
print(f"5. Prepared: {df.shape}", file=sys.stderr)

signal.alarm(0)
