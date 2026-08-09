#!/usr/bin/env python3
"""Offline verifier for the submitted checkpoint and trajectory evidence.

The script intentionally does not start MuJoCo or call an API. It mirrors the
public score conditions in ``app.py`` and also checks the trajectory schema
requested by the submission rules: base pose, 27 joint angles, and movable
object positions at every recorded frame.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
APP_ROOT = HERE.parent
MANIFEST = HERE / "evidence" / "manifest.json"
MODEL = HERE / "models" / "jciiot_unified_l1_l5_bc_v4_epoch10_deploy.pth"
MODEL_SHA256 = "dd41174cdd1ed40d70f309024283326f0732de1aaeb0e3275b1573c13c824c5f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_position(frame: dict, object_name: str):
    values = frame.get("object_positions", {}).get(object_name)
    if not isinstance(values, list) or len(values) < 3:
        return None
    return tuple(float(value) for value in values[:3])


def successful_grasps(trajectory: dict, source: str, allowed: set[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for event in trajectory.get("events", []):
        if not isinstance(event, dict) or event.get("name") != "grasp_end":
            continue
        success = event.get("success") is True or str(event.get("success", "")).lower() in {
            "1", "true", "yes", "ok", "success", "succeeded",
        }
        name = str(event.get("object_name") or "")
        if success and event.get("source") == source and name in allowed:
            result.setdefault(name, int(event.get("frame", 0)))
    return result


def left_source(frames: list[dict], name: str, source_xy: tuple[float, float], start: int) -> bool:
    for frame in frames[max(0, start):]:
        pos = object_position(frame, name)
        if pos and (abs(pos[0] - source_xy[0]) > 1.0 or abs(pos[1] - source_xy[1]) > 1.0):
            return True
    return False


def validate_schema(frames: list[dict]) -> list[str]:
    issues: list[str] = []
    if not frames:
        return ["trajectory has no frames"]
    for index, frame in enumerate(frames):
        base = frame.get("base_pose", {})
        if len(base.get("position", [])) < 3 or len(base.get("orientation_xyzw", [])) < 4:
            issues.append(f"frame {index}: invalid base_pose")
        joints = frame.get("joint_positions", {})
        if not isinstance(joints, dict) or len(joints) != 27:
            issues.append(f"frame {index}: expected 27 joint positions, got {len(joints) if isinstance(joints, dict) else 0}")
        objects = frame.get("object_positions", {})
        if not isinstance(objects, dict) or not objects:
            issues.append(f"frame {index}: missing movable object positions")
        if len(issues) >= 20:
            issues.append("schema issue output truncated")
            break
    return issues


def score_one(record: dict) -> dict:
    path = (HERE / record["trajectory"]).resolve()
    trajectory = json.loads(path.read_text(encoding="utf-8"))
    frames = trajectory.get("frames", [])
    allowed = set(record["objects"])
    grasps = successful_grasps(trajectory, record["source"], allowed)
    collision_frames = sum(bool(frame.get("has_collision")) for frame in frames)
    issues = validate_schema(frames)
    details = []

    if record["level"] == "L5":
        total = 0
        for name in record["objects"]:
            grasped = name in grasps
            departed = grasped and left_source(frames, name, tuple(record["source_xy"]), grasps[name])
            final = object_position(frames[-1], name) if frames else None
            distance = math.dist(final[:2], record["target_xy"]) if final else None
            placed = grasped and distance is not None and distance < 0.80
            total += 5 if departed else 0
            total += 5 if placed else 0
            details.append({
                "object": name,
                "grasped": grasped,
                "left_source": departed,
                "target_distance_m": round(distance, 6) if distance is not None else None,
                "placed": placed,
                "final_position": final,
            })
    else:
        name = next((candidate for candidate in record["objects"] if candidate in grasps), None)
        final = object_position(frames[-1], name) if frames and name else None
        distance = math.dist(final[:2], record["target_xy"]) if final else None
        departed = bool(name) and left_source(frames, name, tuple(record["source_xy"]), grasps[name])
        placed = bool(name) and distance is not None and distance < 0.80
        first_half = max(1, int(record["max_score"]) // 2)
        total = (first_half if departed else 0) + (
            int(record["max_score"]) - first_half if placed else 0
        )
        details.append({
            "object": name,
            "grasped": bool(name),
            "left_source": departed,
            "target_distance_m": round(distance, 6) if distance is not None else None,
            "placed": placed,
            "final_position": final,
        })

    if collision_frames:
        total = max(0, total - 5)
    if total != int(record["max_score"]):
        issues.append(f"score is {total}/{record['max_score']}")
    return {
        "level": record["level"],
        "trajectory": str(path.relative_to(APP_ROOT)),
        "frames": len(frames),
        "successful_grasps": len(grasps),
        "collision_frames": collision_frames,
        "score": total,
        "max_score": int(record["max_score"]),
        "details": details,
        "issues": issues,
        "ok": not issues,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--output",
        type=Path,
        help="also write the complete verification report to this JSON file",
    )
    args = parser.parse_args()
    problems: list[str] = []
    if not MODEL.is_file():
        problems.append(f"missing checkpoint: {MODEL}")
        model_hash = None
    else:
        model_hash = sha256(MODEL)
        if model_hash != MODEL_SHA256:
            problems.append(f"checkpoint SHA-256 mismatch: {model_hash}")
    if not MANIFEST.is_file():
        problems.append(f"missing evidence manifest: {MANIFEST}")
        records = []
    else:
        records = json.loads(MANIFEST.read_text(encoding="utf-8"))["levels"]

    levels = []
    for record in records:
        try:
            levels.append(score_one(record))
        except Exception as exc:
            levels.append({"level": record.get("level"), "score": 0, "max_score": record.get("max_score"), "ok": False, "issues": [str(exc)]})
    total = sum(int(level.get("score", 0)) for level in levels)
    maximum = sum(int(level.get("max_score", 0)) for level in levels)
    ok = not problems and len(levels) == 5 and all(level.get("ok") for level in levels)
    report = {
        "ok": ok,
        "checkpoint": str(MODEL.relative_to(APP_ROOT)),
        "checkpoint_sha256": model_hash,
        "score": total,
        "max_score": maximum,
        "levels": levels,
        "problems": problems,
    }
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"checkpoint: {'OK' if not problems else 'FAIL'} ({model_hash})")
        for level in levels:
            print(f"{level['level']}: {level['score']}/{level['max_score']} frames={level.get('frames', 0)} collisions={level.get('collision_frames', 0)} {'OK' if level.get('ok') else 'FAIL'}")
            for issue in level.get("issues", []):
                print(f"  - {issue}")
        print(f"TOTAL: {total}/{maximum} {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
