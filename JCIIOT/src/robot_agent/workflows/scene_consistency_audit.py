"""Audit task, semantic-map, and MuJoCo scene consistency.

This module is intentionally read-only.  It helps distinguish planner / skill
failures from mismatches in the competition assets before running a scored
task.

Run from ``JCIIOT/`` with::

    PYTHONPATH=src:. python -m robot_agent.workflows.scene_consistency_audit

Use ``--json`` for machine-readable output or ``--strict`` to return a non-zero
exit status when any warning is found.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from robot_agent.core.map_loader import load_map_files
from robot_agent.core.scene_context import SceneContext


@dataclass
class TaskAudit:
    """Serializable consistency result for one competition task."""

    task_index: int
    level: str
    env_name: str
    source: str
    target: str
    object_name: str
    semantic_source_xy: list[float] | None = None
    semantic_target_xy: list[float] | None = None
    object_xy: list[float] | None = None
    object_source_distance_m: float | None = None
    object_declared_port: str = ""
    runtime_input_ports: list[str] = field(default_factory=list)
    runtime_output_ports: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ok"] = self.ok
        return data


def _xy(value: Any) -> list[float] | None:
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size < 2:
            return None
        return [round(float(arr[0]), 6), round(float(arr[1]), 6)]
    except Exception:
        return None


def _runtime_object_xy(env: Any, object_name: str, metadata: dict) -> list[float] | None:
    fixed_pose = metadata.get("fixed_pose") if isinstance(metadata, dict) else None
    pos = _xy(fixed_pose)
    if pos is not None:
        return pos

    try:
        body_id = env.obj_body_id[object_name]
        return _xy(env.sim.data.body_xpos[body_id])
    except Exception:
        return None


def audit_task(app_dir: Path, task_index: int, task: dict[str, Any]) -> TaskAudit:
    """Create a fresh headless scene, inspect it, and always close it."""

    env_name = str(task["env_name"])
    prefix = str(task["scene_prefix"])
    source = str(task["source"])
    target = str(task["target"])
    object_name = str(task["object"])
    result = TaskAudit(
        task_index=task_index,
        level=str(task.get("level", f"L{task_index + 1}")),
        env_name=env_name,
        source=source,
        target=target,
        object_name=object_name,
    )

    map_dir = (
        app_dir
        / "robosuite"
        / "robosuite"
        / "environments"
        / "factory_sorting"
        / "generated_maps"
    )
    semantic_path = map_dir / f"{prefix}_scene_regenerated_semantic_map.json"
    grid_path = map_dir / f"{prefix}_scene_regenerated_occupancy_grid.npy"
    if not semantic_path.exists() or not grid_path.exists():
        result.issues.append("semantic map or occupancy grid is missing")
        return result

    scene, _grid = load_map_files(semantic_path, grid_path)
    context = SceneContext.from_semantic_map(scene)
    source_port = context.input_ports.get(source)
    target_port = context.output_ports.get(target)
    if source_port is None:
        result.issues.append(f"semantic source port missing: {source}")
    else:
        result.semantic_source_xy = _xy(source_port.center)
    if target_port is None:
        result.issues.append(f"semantic target port missing: {target}")
    else:
        result.semantic_target_xy = _xy(target_port.center)

    backend: Any = None
    try:
        # Delay the robosuite import so ``--json`` can redirect its verbose
        # startup diagnostics to stderr and keep stdout machine-readable.
        from robot_agent.environments import RobosuiteBackend

        backend = RobosuiteBackend(
            env_name=env_name,
            camera="birdview",
            drive_mode="direct",
            headless=True,
        )
        backend.reset()
        env = backend.env
        runtime_inputs = getattr(env, "input_ports", {}) or {}
        runtime_outputs = getattr(env, "output_ports", {}) or {}
        result.runtime_input_ports = sorted(str(name) for name in runtime_inputs)
        result.runtime_output_ports = sorted(str(name) for name in runtime_outputs)

        all_metadata = getattr(env, "material_metadata", {}) or {}
        metadata = all_metadata.get(object_name)
        if not isinstance(metadata, dict):
            result.issues.append(f"target object missing from material_metadata: {object_name}")
            metadata = {}
        result.object_declared_port = str(metadata.get("port_name") or "")
        result.object_xy = _runtime_object_xy(env, object_name, metadata)

        if not result.object_declared_port:
            result.issues.append("target object has no runtime port_name")
        elif result.object_declared_port != source:
            result.issues.append(
                f"object port mismatch: task={source}, runtime={result.object_declared_port}"
            )

        if source not in runtime_inputs:
            result.issues.append(f"exact runtime input port missing: {source}")
        if target not in runtime_outputs:
            result.issues.append(f"exact runtime output port missing: {target}")

        if result.object_xy is not None and result.semantic_source_xy is not None:
            distance = float(
                np.linalg.norm(
                    np.asarray(result.object_xy) - np.asarray(result.semantic_source_xy)
                )
            )
            result.object_source_distance_m = round(distance, 6)
            if distance > 2.0:
                result.issues.append(
                    f"object is {distance:.2f}m from semantic source center"
                )
    except Exception as exc:
        result.issues.append(f"scene inspection failed: {type(exc).__name__}: {exc}")
    finally:
        if backend is not None:
            backend.close()

    return result


def run_audit(app_dir: Path) -> list[TaskAudit]:
    """Audit every task declared in ``knowledge/task_config.json``."""

    config_path = app_dir / "knowledge" / "task_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    tasks = config.get("tasks", [])
    return [audit_task(app_dir, index, task) for index, task in enumerate(tasks)]


def _print_human(results: list[TaskAudit]) -> None:
    for item in results:
        status = "OK" if item.ok else f"WARN({len(item.issues)})"
        distance = (
            "n/a"
            if item.object_source_distance_m is None
            else f"{item.object_source_distance_m:.2f}m"
        )
        print(
            f"{item.level} {status}: {item.source} -> {item.target}; "
            f"object={item.object_name}; runtime_port={item.object_declared_port or 'n/a'}; "
            f"source_distance={distance}"
        )
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
        "--strict",
        action="store_true",
        help="return exit code 1 when any mismatch is found",
    )
    args = parser.parse_args(argv)
    if args.json:
        with contextlib.redirect_stdout(sys.stderr):
            results = run_audit(args.app_dir.resolve())
        print(json.dumps([item.to_dict() for item in results], indent=2, ensure_ascii=False))
    else:
        results = run_audit(args.app_dir.resolve())
        _print_human(results)
    return 1 if args.strict and any(not item.ok for item in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
