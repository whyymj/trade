#!/usr/bin/env python3
import signal
import sys


def timeout_handler(sig, frame):
    print("TIMEOUT!", file=sys.stderr)
    sys.exit(1)


signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)

print("Starting...", file=sys.stderr)

import analysis.fund_lstm as fl

print("Imported fund_lstm", file=sys.stderr)

result = fl.train_model("013428", 365, 1)
print(f"Result: {result}", file=sys.stderr)

signal.alarm(0)
