#!/usr/bin/env python3
import signal
import sys


def timeout_handler(sig, frame):
    print("TIMEOUT!", file=sys.stderr)
    sys.exit(1)


signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)

print("1. Starting...", file=sys.stderr)

import analysis.fund_lstm as fl

print("2. fund_lstm imported", file=sys.stderr)

# 检查 torch 相关的全局变量
print(f"3. _TORCH_AVAILABLE: {fl._TORCH_AVAILABLE}", file=sys.stderr)
print(f"4. _torch: {fl._torch}", file=sys.stderr)
print(f"5. _device: {fl._device}", file=sys.stderr)

signal.alarm(0)
