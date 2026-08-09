#!/usr/bin/env python3
"""Run selected official levels through the same isolated runner as app.py."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
RUNNER = APP_ROOT / "src" / "robot_agent" / "task_subprocess_runner.py"
TASKS = (
    "For this task, you need to transport a blue, hollow plastic box. Please move it from the starting point \"Pick Station 2\" to the destination \"Place Station 3\". Please follow the Standard Operating Procedure (SOP).",
    "Current Task Material Information: Material Name: Green-rimmed storage bin; Starting Location: Pick Station 1; Target Location: Place Station 3; Quantity to Transport: 1",
    "Please follow the SOP. The object is a blue material transfer bin. The Pick Station is Pick Station 1, and the Place Station is Place Station 2.",
    "Please strictly adhere to the Standard Operating Procedure (SOP) for this task. The object is a blue, hollow plastic box. The Pick Station is Pick Station 5, and the Place Station is Place Station 2.",
    "Move the three white-rimmed storage bins from Pick Station 6 to Place Station 1.",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", nargs="+", choices=["L1", "L2", "L3", "L4", "L5"], default=["L1", "L2", "L3", "L4", "L5"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-knowledge", action="store_true")
    args = parser.parse_args()
    env = os.environ.copy()
    local_paths = [str(APP_ROOT / "src"), str(APP_ROOT), str(APP_ROOT / "robomimic"), str(APP_ROOT / "robosuite" / "robosuite")]
    env["PYTHONPATH"] = os.pathsep.join(local_paths + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env["PYTHONIOENCODING"] = "utf-8"

    for level in args.levels:
        index = int(level[1:]) - 1
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_SUBMISSION_{level}"
        env_name = ("FactorySorting1_3FO3ERFHISEM", "FactorySorting3_3FO3ERRPH7X9", "FactorySorting5_3FO3ERTPXEUT", "FactorySorting7_3FO3ERFKY9RN", "FactorySorting9_3FO3ERT2C5FP")[index]
        result = APP_ROOT / "recordings" / env_name / f"result_{stamp}.json"
        command = [sys.executable, str(RUNNER), "--task", TASKS[index], "--task-index", str(index), "--timestamp", stamp, "--result-json", str(result), "--app-dir", str(APP_ROOT), "--knowledge-enabled", str(not args.no_knowledge).lower()]
        print(" ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=APP_ROOT, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
