#!/usr/bin/env python3
"""Render canonical submission trajectories into compact review GIFs."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
APP_ROOT = HERE.parent
MANIFEST = HERE / "evidence" / "manifest.json"
DEMO_DIR = HERE / "demos"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--levels",
        nargs="+",
        choices=["L1", "L2", "L3", "L4", "L5"],
        default=["L1", "L2", "L3", "L4", "L5"],
    )
    parser.add_argument("--camera", default="birdview")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument(
        "--max-frames", type=int, default=180,
        help="uniformly sample long trajectories for a compact review GIF",
    )
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    from robot_agent.environments.robosuite_backend import RobosuiteBackend

    records = {
        item["level"]: item
        for item in json.loads(MANIFEST.read_text(encoding="utf-8"))["levels"]
    }
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    for level in args.levels:
        record = records[level]
        trajectory = (HERE / record["trajectory"]).resolve()
        output = DEMO_DIR / f"{level.lower()}_{args.camera}.gif"
        print(f"{level}: {trajectory.relative_to(APP_ROOT)} -> {output.relative_to(APP_ROOT)}", flush=True)
        source_data = json.loads(trajectory.read_text(encoding="utf-8"))
        frames = source_data.get("frames", [])
        replay_path = trajectory
        temporary_name: str | None = None
        if args.max_frames > 1 and len(frames) > args.max_frames:
            indices = np.linspace(0, len(frames) - 1, args.max_frames, dtype=int)
            source_data["frames"] = [frames[int(index)] for index in indices]
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", prefix=f"{level.lower()}_demo_",
                encoding="utf-8", delete=False,
            ) as temporary:
                json.dump(source_data, temporary, ensure_ascii=False)
                temporary_name = temporary.name
                replay_path = Path(temporary.name)

        backend = RobosuiteBackend(
            env_name=record["environment"],
            camera=args.camera,
            drive_mode="direct",
            headless=True,
        )
        try:
            backend.reset()
            rendered = backend.replay_trajectory(
                replay_path,
                output,
                camera=args.camera,
                width=args.width,
                height=args.height,
            )
            print(
                f"  rendered {len(rendered)} sampled frames from {len(frames)} trajectory frames",
                flush=True,
            )
        finally:
            backend.close()
            if temporary_name:
                Path(temporary_name).unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
