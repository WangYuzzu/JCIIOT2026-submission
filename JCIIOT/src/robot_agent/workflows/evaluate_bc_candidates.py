"""Evaluate BC checkpoints with the official physical FactorySorting rollout."""

from __future__ import annotations

import argparse
import ast
import contextlib
import gc
import io
import json
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[3]
ROBOSUITE_INNER = APP_ROOT / "robosuite"
for _path in (ROBOSUITE_INNER, APP_ROOT):
    _value = str(_path)
    if _value not in sys.path:
        sys.path.insert(0, _value)


def _parse_pose(value: str) -> tuple[str, list[float], list[float]]:
    fields = value.split(":")
    if len(fields) != 4:
        raise argparse.ArgumentTypeError("pose must be NAME:X:Y:YAW")
    name, x, y, yaw = fields
    return name, [float(x), float(y), 0.0], [0.0, 0.0, float(yaw)]


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


class _Tee(io.StringIO):
    """Capture rollout diagnostics while preserving normal terminal output."""

    def write(self, value: str) -> int:
        sys.__stdout__.write(value)
        sys.__stdout__.flush()
        return super().write(value)


def _last_literal(output: str, label: str):
    matches = re.findall(rf"{re.escape(label)}:\s*(\{{[^\n]+\}})", output)
    if not matches:
        return None
    try:
        return ast.literal_eval(matches[-1])
    except (SyntaxError, ValueError):
        return None


def _rollout_diagnostics(output: str) -> dict:
    return {
        "gripper_end_distances": _last_literal(output, "gripper end distances"),
        "fingerpad_contacts": _last_literal(output, "fingerpad contact status"),
        "grasp_status": _last_literal(output, "grasp status"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--factory-scene",
        default="factory_sorting_1_3fo3erfhisem",
    )
    parser.add_argument(
        "--object-name",
        default="line_5_container_h01_near",
    )
    parser.add_argument(
        "--poses",
        type=_parse_pose,
        nargs="+",
        default=[_parse_pose("runtime:8.0:4.6:3.139422")],
    )
    parser.add_argument("--eval-steps", type=int, default=360)
    parser.add_argument("--post-hold-steps", type=int, default=5)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="policy inference device; does not alter MuJoCo's EGL device",
    )
    parser.add_argument(
        "--wrapped-without-reset",
        action="store_true",
        help="mirror the deployment backend's lazy wrapped-env initialization",
    )
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("team_submission/training_artifacts/bc_candidate_eval.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from robosuite.environments.factory_sorting import (
        load_factory_sorting_evalization as evaluation,
    )
    from robot_agent.skills.bc_task_conditioning import maybe_condition_policy

    records = []
    for checkpoint in args.checkpoints:
        checkpoint = checkpoint.resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        for pose_index, (pose_name, base_pos, base_ori) in enumerate(args.poses):
            started = time.perf_counter()
            print(
                f"EVALUATE checkpoint={checkpoint.name} pose={pose_name} "
                f"base_pos={base_pos} yaw={base_ori[2]}",
                flush=True,
            )
            error = None
            result = None
            capture = _Tee()
            try:
                with contextlib.redirect_stdout(capture):
                    if args.wrapped_without_reset:
                        env_args = argparse.Namespace(
                            checkpoint=checkpoint,
                            factory_scene=args.factory_scene,
                            device=args.device,
                            verbose=False,
                            object_name=args.object_name,
                            site_below_offset=evaluation.DEFAULT_GRIPPER_TARGET_OFFSET,
                            robot_base_pos=base_pos,
                            robot_base_ori=base_ori,
                            renderer="mjviewer",
                            camera=evaluation.DEFAULT_CAMERA,
                            camera_height=evaluation.DEFAULT_CAMERA_HEIGHT,
                            camera_width=evaluation.DEFAULT_CAMERA_WIDTH,
                            controller=None,
                            gripper_types="Robotiq140Gripper",
                            seed=args.seed + pose_index,
                            render_sleep=0.0,
                            show_object_sites=False,
                            object_site_size=evaluation.DEFAULT_OBJECT_SITE_SIZE,
                        )
                        policy, config, checkpoint_dict = evaluation.load_policy_and_config(
                            env_args
                        )
                        policy = maybe_condition_policy(
                            policy,
                            checkpoint_dict,
                            args.object_name,
                        )
                        env = evaluation.make_eval_env(
                            env_args,
                            config=config,
                            ckpt_dict=checkpoint_dict,
                            render=False,
                        )
                        try:
                            result = evaluation.run_factory_sorting_grasp_in_wrapped_env(
                                env=env,
                                policy=policy,
                                eval_steps=args.eval_steps,
                                object_name=args.object_name,
                                post_hold_steps=args.post_hold_steps,
                                initial_view_steps=0,
                                render=False,
                            )
                        finally:
                            close = getattr(env, "close", None)
                            if callable(close):
                                close()
                            else:
                                raw_env = getattr(env, "env", None)
                                raw_close = getattr(raw_env, "close", None)
                                if callable(raw_close):
                                    raw_close()
                    else:
                        result = evaluation.run_factory_sorting_grasp(
                            checkpoint=checkpoint,
                            factory_scene=args.factory_scene,
                            num_rollouts=1,
                            eval_steps=args.eval_steps,
                            device=args.device,
                            object_name=args.object_name,
                            post_hold_steps=args.post_hold_steps,
                            initial_view_steps=0,
                            show_object_sites=False,
                            robot_base_pos=base_pos,
                            robot_base_ori=base_ori,
                            seed=args.seed + pose_index,
                            render=False,
                        )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                print(f"ERROR {error}\n{traceback.format_exc()}", flush=True)
            diagnostics = _rollout_diagnostics(capture.getvalue())
            record = {
                "checkpoint": str(checkpoint),
                "pose": pose_name,
                "base_pos": base_pos,
                "base_ori": base_ori,
                "success": bool(result and result.get("success")),
                "successes": int((result or {}).get("successes", 0)),
                "attempts": int((result or {}).get("num_rollouts", 1)),
                **diagnostics,
                "error": error,
                "elapsed_sec": round(time.perf_counter() - started, 3),
            }
            records.append(record)
            print(f"RESULT {json.dumps(record, sort_keys=True)}", flush=True)
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            _atomic_write(
                args.output,
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "factory_scene": args.factory_scene,
                        "object_name": args.object_name,
                        "eval_steps": args.eval_steps,
                        "records": records,
                    },
                    indent=2,
                )
                + "\n",
            )
    return 0 if all(record["error"] is None for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
