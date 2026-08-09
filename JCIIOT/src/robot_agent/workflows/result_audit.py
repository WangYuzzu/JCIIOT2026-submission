"""Summarize and validate persisted JCIIOT score / trajectory artifacts.

Unlike the Streamlit UI, this audit does not depend on session state and does
not create a MuJoCo environment.  It is safe to run after an application or
machine restart.

Run from ``JCIIOT/`` with::

    PYTHONPATH=src:. python -m robot_agent.workflows.result_audit
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ResultAudit:
    task_index: int
    level: str
    env_name: str
    max_score: int
    score: int = 0
    status: str = "MISSING"
    elapsed_sec: float | None = None
    score_file: str = ""
    result_file: str = ""
    subprocess_log: str = ""
    trajectory_file: str = ""
    replay_gifs: list[str] = field(default_factory=list)
    available_replay_gifs: list[str] = field(default_factory=list)
    frame_count: int = 0
    event_count: int = 0
    grasp_success_count: int = 0
    collision_frame_count: int = 0
    issues: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ok"] = self.ok
        return data


def _event_succeeded(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "ok",
        "success",
        "succeeded",
    }


def _latest_score(recording_dir: Path) -> Path | None:
    candidates = list(recording_dir.glob("score_*.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))


def _resolve_trajectory(recording_dir: Path, value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if path.exists():
        return path
    relocated = recording_dir / path.name
    return relocated if relocated.exists() else None


def audit_task(app_dir: Path, task_index: int, task: dict[str, Any]) -> ResultAudit:
    level = str(task.get("level", f"L{task_index + 1}"))
    env_name = str(task["env_name"])
    max_score = int(task.get("max_score", 0))
    result = ResultAudit(
        task_index=task_index,
        level=level,
        env_name=env_name,
        max_score=max_score,
    )
    recording_dir = app_dir / "recordings" / env_name
    score_path = _latest_score(recording_dir)
    if score_path is None:
        result.issues.append("no persisted score file")
        return result

    result.score_file = str(score_path)
    timestamp_match = re.search(r"score_(\d{8}_\d{6})_", score_path.name)
    if timestamp_match:
        timestamp = timestamp_match.group(1)
        result_path = recording_dir / f"result_{timestamp}.json"
        log_path = recording_dir / f"subprocess_{timestamp}.log"
        if result_path.exists():
            result.result_file = str(result_path)
        if log_path.exists():
            result.subprocess_log = str(log_path)
    try:
        score_data = json.loads(score_path.read_text(encoding="utf-8"))
    except Exception as exc:
        result.issues.append(f"score JSON unreadable: {type(exc).__name__}: {exc}")
        return result

    result.score = int(score_data.get("score", 0))
    result.status = str(score_data.get("status", "UNKNOWN"))
    try:
        elapsed = score_data.get("elapsed_sec")
        result.elapsed_sec = None if elapsed is None else float(elapsed)
    except (TypeError, ValueError):
        result.issues.append("score elapsed_sec is not numeric")

    if result.status not in {"OK", "RECOVERED"}:
        result.issues.append(f"latest score status is {result.status}")
    if result.score != result.max_score:
        result.issues.append(f"score is {result.score}/{result.max_score}, not full")

    trajectory_path = _resolve_trajectory(
        recording_dir, score_data.get("trajectory")
    )
    if trajectory_path is None:
        result.issues.append("score references a missing trajectory")
        return result
    result.trajectory_file = str(trajectory_path)

    try:
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    except Exception as exc:
        result.issues.append(f"trajectory JSON unreadable: {type(exc).__name__}: {exc}")
        return result

    frames = trajectory.get("frames", [])
    events = trajectory.get("events", [])
    if not isinstance(frames, list):
        frames = []
        result.issues.append("trajectory frames is not a list")
    if not isinstance(events, list):
        events = []
        result.issues.append("trajectory events is not a list")
    result.frame_count = len(frames)
    result.event_count = len(events)
    result.grasp_success_count = sum(
        1
        for event in events
        if isinstance(event, dict)
        and event.get("name") == "grasp_end"
        and _event_succeeded(event.get("success"))
    )
    result.collision_frame_count = sum(
        1
        for frame in frames
        if isinstance(frame, dict) and bool(frame.get("has_collision"))
    )

    expected_grasps = 3 if level == "L5" else 1
    if result.frame_count == 0:
        result.issues.append("trajectory has no frames")
    if result.grasp_success_count < expected_grasps:
        result.issues.append(
            f"successful grasp events {result.grasp_success_count}/{expected_grasps}"
        )
    if result.collision_frame_count:
        result.issues.append(
            f"trajectory has {result.collision_frame_count} collision frames"
        )

    trajectory_tag = trajectory_path.stem.removeprefix("trajectory_")
    result.replay_gifs = sorted(
        str(path)
        for path in recording_dir.glob(f"replay_{trajectory_tag}*.gif")
    )
    result.available_replay_gifs = sorted(
        (str(path) for path in recording_dir.glob("replay_*.gif")),
        key=lambda value: Path(value).stat().st_mtime,
        reverse=True,
    )
    return result


def run_audit(app_dir: Path) -> list[ResultAudit]:
    config = json.loads(
        (app_dir / "knowledge" / "task_config.json").read_text(encoding="utf-8")
    )
    return [
        audit_task(app_dir, index, task)
        for index, task in enumerate(config.get("tasks", []))
    ]


def _print_human(results: list[ResultAudit]) -> None:
    for item in results:
        status = "PASS" if item.ok else f"FAIL({len(item.issues)})"
        elapsed = "n/a" if item.elapsed_sec is None else f"{item.elapsed_sec:.3f}s"
        print(
            f"{item.level} {status}: score={item.score}/{item.max_score}; "
            f"elapsed={elapsed}; frames={item.frame_count}; "
            f"grasps={item.grasp_success_count}; "
            f"collision_frames={item.collision_frame_count}; "
            f"replays={len(item.replay_gifs)}/{len(item.available_replay_gifs)} "
            "matching/available"
        )
        print(f"  score: {item.score_file or 'n/a'}")
        print(f"  result: {item.result_file or 'n/a'}")
        print(f"  subprocess log: {item.subprocess_log or 'n/a'}")
        print(f"  trajectory: {item.trajectory_file or 'n/a'}")
        for issue in item.issues:
            print(f"  - {issue}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--app-dir",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="JCIIOT application directory (default: inferred from this module)",
    )
    parser.add_argument("--json", action="store_true", help="print JSON results")
    parser.add_argument(
        "--strict", action="store_true", help="return 1 when any result is invalid"
    )
    args = parser.parse_args(argv)
    results = run_audit(args.app_dir.resolve())
    if args.json:
        print(json.dumps([item.to_dict() for item in results], indent=2, ensure_ascii=False))
    else:
        _print_human(results)
    return 1 if args.strict and any(not item.ok for item in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
