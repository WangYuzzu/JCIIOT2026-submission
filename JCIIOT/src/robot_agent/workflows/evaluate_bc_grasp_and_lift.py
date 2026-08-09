"""Verify a BC checkpoint with the strict grasp-then-lift physical gate."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[3]
for _path in (APP_ROOT, APP_ROOT / "robosuite"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from robosuite.environments.factory_sorting import (  # noqa: E402
    lift_after_grasp,
    load_factory_sorting_evalization as evaluation,
)
from robot_agent.skills.bc_task_conditioning import maybe_condition_policy  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--factory-scene",
        default="factory_sorting_1_3fo3erfhisem",
    )
    parser.add_argument(
        "--object-name",
        default="line_5_container_h01_near",
    )
    parser.add_argument("--base-x", type=float, default=8.0)
    parser.add_argument("--base-y", type=float, default=4.6)
    parser.add_argument("--yaw", type=float, default=3.139422)
    parser.add_argument("--eval-steps", type=int, default=360)
    parser.add_argument("--post-hold-steps", type=int, default=5)
    parser.add_argument("--lift-height", type=float, default=0.15)
    parser.add_argument("--lift-max-steps", type=int, default=300)
    parser.add_argument("--lift-hold-steps", type=int, default=20)
    parser.add_argument("--lift-tolerance", type=float, default=0.02)
    parser.add_argument("--lift-max-action", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument(
        "--reset-before-rollout",
        action="store_true",
        help=(
            "perform the legacy extra reset/reset_to cycle before policy execution; "
            "the default mirrors the deployment backend and uses the environment "
            "state created by make_eval_env"
        ),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _env_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        checkpoint=args.checkpoint.resolve(),
        factory_scene=args.factory_scene,
        device="auto",
        verbose=False,
        object_name=args.object_name,
        site_below_offset=evaluation.DEFAULT_GRIPPER_TARGET_OFFSET,
        robot_base_pos=[args.base_x, args.base_y, 0.0],
        robot_base_ori=[0.0, 0.0, args.yaw],
        renderer="mjviewer",
        camera=evaluation.DEFAULT_CAMERA,
        camera_height=evaluation.DEFAULT_CAMERA_HEIGHT,
        camera_width=evaluation.DEFAULT_CAMERA_WIDTH,
        controller=None,
        gripper_types="Robotiq140Gripper",
        seed=args.seed,
        render_sleep=0.0,
        show_object_sites=False,
        object_site_size=evaluation.DEFAULT_OBJECT_SITE_SIZE,
    )


def main() -> int:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    env_args = _env_args(args)
    policy, config, checkpoint_dict = evaluation.load_policy_and_config(env_args)
    policy = maybe_condition_policy(
        policy,
        checkpoint_dict,
        env_args.object_name,
    )
    env = evaluation.make_eval_env(
        env_args,
        config=config,
        ckpt_dict=checkpoint_dict,
        render=False,
    )
    try:
        policy.start_episode()
        if args.reset_before_rollout:
            env.reset()
            state = env.get_state()
            env.reset_to(state)
        grasp = evaluation.run_factory_sorting_grasp_in_wrapped_env(
            env=env,
            policy=policy,
            eval_steps=args.eval_steps,
            object_name=env_args.object_name,
            post_hold_steps=args.post_hold_steps,
            initial_view_steps=0,
            render=False,
        )
        lift = lift_after_grasp.lift_grasped_object(
            env=env,
            object_name=env_args.object_name,
            lift_height=args.lift_height,
            max_steps=args.lift_max_steps,
            hold_steps=args.lift_hold_steps,
            tolerance=args.lift_tolerance,
            max_action=args.lift_max_action,
            render=False,
        )
        result = {
            "checkpoint": str(args.checkpoint.resolve()),
            "base_pos": env_args.robot_base_pos,
            "base_ori": env_args.robot_base_ori,
            "eval_steps": args.eval_steps,
            "grasp": grasp,
            "lift": lift,
            "success": bool(grasp.get("success") and lift.get("success")),
        }
        print("GRASP_AND_LIFT_RESULT " + json.dumps(result, sort_keys=True))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_suffix(args.output.suffix + ".tmp")
            temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            temporary.replace(args.output)
        return 0 if result["success"] else 2
    finally:
        raw_env = evaluation.base_robosuite_env(env)
        if hasattr(raw_env, "close"):
            raw_env.close()
        gc.collect()


if __name__ == "__main__":
    raise SystemExit(main())
