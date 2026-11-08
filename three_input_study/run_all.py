"""Reproduce everything: python three_input_study/run_all.py  (about 5 minutes on one CPU core)."""

import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
py = sys.executable


def run(*a):
    t = time.time()
    print(">>", " ".join(a), flush=True)
    subprocess.run([py] + list(a), cwd=ROOT, check=True)
    print(f"   ({time.time() - t:.0f}s)", flush=True)


for d in ("results", "figures"):
    os.makedirs(os.path.join(ROOT, d), exist_ok=True)
tr = os.path.join(ROOT, "tests", "test_results.json")
if os.path.exists(tr):
    os.remove(tr)
run("tests/run_tests.py", "pre")  # T1-T8
run("src/stage4.py")  # coefficients, sweep(192) -> results.csv, sweep(256) -> results_n256.csv
run("src/gold_check.py")  # GOLD x-grid diagnostic
run("tests/run_tests.py", "post")  # T9, T10
run("src/extras.py", "8", "9")  # optional: Taylor, value-loss training
run("src/analysis.py")  # threshold table
run("src/figures.py")  # F1-F7, one-pager
run("src/report.py")  # results/report.txt
print("done")
