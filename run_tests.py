"""Test runner for all chapter scripts.

Runs each chapter file as a subprocess with a non-interactive matplotlib backend
(Agg), so no plot windows are opened and execution is never paused. Reports
pass/fail for each file with timing and any error output.

Usage:
    python run_tests.py
"""

import os
import subprocess
import sys
import time
from pathlib import Path

CHAPTERS = [
    "Chapter3.py",
    "Chapter4_1.py",
    "Chapter4_3.py",
    "Chapter4_4.py",
    "Chapter4_5.py",
    "Chapter5_1.py",
    "Chapter5_2.py",
    "Chapter5_3.py",
    "Chapter6_1.py",
    "Chapter6_2.py",
    "Chapter6_3.py",
    "Chapter7.py",
    "Chapter8_12.py",
    "Chapter8_3.py",
    "Chapter9_1.py",
    "Chapter9_2.py",
    "Chapter9_4.py",
    "Chapter9_5.py",
]

WORKSPACE = Path(__file__).parent

# Use the same Python interpreter that is running this script
PYTHON = sys.executable

# Set MPLBACKEND=Agg so matplotlib never opens a window, and inherit the rest
# of the current environment (PATH, venv, etc.)
env = os.environ.copy()
env["MPLBACKEND"] = "Agg"
env["PYTHONIOENCODING"] = "utf-8"

passed = []
failed = []

print(f"Running {len(CHAPTERS)} chapter scripts with MPLBACKEND=Agg\n")
print("-" * 60)

for chapter in CHAPTERS:
    script = WORKSPACE / chapter
    start = time.monotonic()
    result = subprocess.run(
        [PYTHON, str(script)],
        capture_output=True,
        text=True,
        cwd=str(WORKSPACE),
        env=env,
    )
    elapsed = time.monotonic() - start

    status = "PASS" if result.returncode == 0 else "FAIL"
    print(f"[{status}] {chapter:<25}  {elapsed:6.1f}s")

    if result.returncode == 0:
        passed.append(chapter)
    else:
        failed.append(chapter)
        # Print stderr (and stdout if it contains useful info) indented
        if result.stderr.strip():
            for line in result.stderr.strip().splitlines():
                print(f"        {line}")
        if result.stdout.strip() and not result.stderr.strip():
            for line in result.stdout.strip().splitlines()[-10:]:
                print(f"        {line}")

print("-" * 60)
print(f"\n{len(passed)}/{len(CHAPTERS)} passed", end="")
if failed:
    print(f" |  Failed: {', '.join(failed)}")
else:
    print(" — all good!")

sys.exit(0 if not failed else 1)
