#!/usr/bin/env python3
"""Select the latest successful run for each level and package canonical evidence."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


HERE = Path(__file__).resolve().parent
APP_ROOT = HERE.parent
EVIDENCE = HERE / "evidence"

LEVELS = (
    {
        "level": "L1", "environment": "FactorySorting1_3FO3ERFHISEM",
        "source": "input_5", "target": "output_4",
        "source_xy": [7.186, 3.938], "target_xy": [-0.166, -7.290],
        "objects": ["line_5_container_h01_near"], "max_score": 10,
    },
    {
        "level": "L2", "environment": "FactorySorting3_3FO3ERRPH7X9",
        "source": "input_6", "target": "output_4",
        "source_xy": [11.937, 3.932], "target_xy": [-0.166, -7.290],
        "objects": ["green_tote_b01_upper"], "max_score": 15,
    },
    {
        "level": "L3", "environment": "FactorySorting5_3FO3ERTPXEUT",
        "source": "aux_input_1", "target": "output_5",
        "source_xy": [0.144, 8.473], "target_xy": [4.872, -7.261],
        "objects": ["blue_tote_b01_far_right", "blue_tote_b01_near_right"],
        "max_score": 20,
    },
    {
        "level": "L4", "environment": "FactorySorting7_3FO3ERFKY9RN",
        "source": "input_2", "target": "output_5",
        "source_xy": [-9.761, 5.010], "target_xy": [4.872, -7.261],
        "objects": ["blue_container_h01_back_upper"], "max_score": 25,
    },
    {
        "level": "L5", "environment": "FactorySorting9_3FO3ERT2C5FP",
        "source": "input_1", "target": "aux_output_1",
        "source_xy": [-14.544, 5.010], "target_xy": [0.144, 8.473],
        "objects": [
            "white_tote_b01_left_back", "white_tote_b01_left_center",
            "white_tote_b01_left_front",
        ],
        "max_score": 30,
    },
)


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def successful(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return bool(data.get("frames")) and path.name.endswith("_OK.json")
    except Exception:
        return False


def main() -> int:
    records = []
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    for item in LEVELS:
        recording_dir = APP_ROOT / "recordings" / item["environment"]
        candidates = [path for path in recording_dir.glob("trajectory_*_OK.json") if successful(path)]
        if not candidates:
            raise FileNotFoundError(f"no successful trajectory in {recording_dir}")
        trajectory = max(candidates, key=lambda path: path.stat().st_mtime_ns)
        stamp = trajectory.name[len("trajectory_"):-len("_OK.json")]
        result = recording_dir / f"result_{stamp}.json"
        if not result.is_file():
            raise FileNotFoundError(result)

        level_dir = EVIDENCE / item["level"]
        level_dir.mkdir(parents=True, exist_ok=True)
        canonical_trajectory = level_dir / "trajectory.json"
        canonical_result = level_dir / "result.json"
        shutil.copy2(trajectory, canonical_trajectory)
        shutil.copy2(result, canonical_result)
        # Canonical evidence must remain portable after cloning. Runtime result
        # files contain machine-local absolute paths, so rewrite those two
        # references without changing the execution payload.
        result_data = json.loads(canonical_result.read_text(encoding="utf-8"))
        result_data["trajectory"] = str(canonical_trajectory.relative_to(APP_ROOT))
        result_data["running_trajectory"] = None
        canonical_result.write_text(
            json.dumps(result_data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        record = dict(item)
        record.update({
            "trajectory": str(canonical_trajectory.relative_to(HERE)),
            "result": str(canonical_result.relative_to(HERE)),
            "source_trajectory": str(trajectory.relative_to(APP_ROOT)),
            "trajectory_bytes": canonical_trajectory.stat().st_size,
            "trajectory_sha256": digest(canonical_trajectory),
            "result_sha256": digest(canonical_result),
        })
        records.append(record)
        print(f"{item['level']}: {trajectory.name}")

    manifest = {
        "schema_version": 1,
        "official_base_commit": "fa0eaef",
        "selection": "latest successful trajectory by file modification time",
        "levels": records,
    }
    (EVIDENCE / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
