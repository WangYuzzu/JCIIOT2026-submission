"""Collect L1 DAgger-style corrective demonstrations from BC failure states.

The current BC policy is executed for a configurable number of steps. At each
visited state, a geometric expert action is recorded as the supervision label
while the BC action is applied to the simulator (standard DAgger querying).
After the takeover point, the official scripted controller recovers from the
resulting state and must establish a real bilateral grasp before the episode is
kept.

The scripted controller is used only to produce training data. Runtime task
execution remains the official robomimic BC -> contact -> lift path.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np


APP_ROOT = Path(__file__).resolve().parents[3]
ROBOSUITE_INNER = APP_ROOT / "robosuite"
for _path in (APP_ROOT, ROBOSUITE_INNER):
    _value = str(_path)
    if _value not in sys.path:
        sys.path.insert(0, _value)

from robosuite.environments.factory_sorting import (  # noqa: E402
    load_factory_sorting_1_3fo3erfhisem_collect as expert,
)
from robosuite.environments.factory_sorting import (  # noqa: E402
    load_factory_sorting_evalization as evaluation,
)


DEFAULT_CHECKPOINT = (
    APP_ROOT
    / "team_submission"
    / "training_artifacts"
    / "models"
    / "jciiot_l1_rgb_bc_scratch_diverse_v2"
    / "20260723174043"
    / "models"
    / "model_epoch_150.pth"
)
DEFAULT_SOURCE = (
    APP_ROOT
    / "team_submission"
    / "training_artifacts"
    / "datasets_l1_diverse_v1"
    / "l1_line_5_container_h01_near.hdf5"
)
DEFAULT_OUTPUT = (
    APP_ROOT
    / "team_submission"
    / "training_artifacts"
    / "datasets_l1_corrective_v1"
    / "l1_corrective.hdf5"
)
DEFAULT_AGGREGATE = (
    APP_ROOT
    / "team_submission"
    / "training_artifacts"
    / "datasets_l1_corrective_v1"
    / "l1_aggregate.hdf5"
)
QUERY_MAX_ACTION = 0.25


@dataclass
class EpisodeBuffer:
    observations: dict[str, list[np.ndarray]] = field(
        default_factory=lambda: {key: [] for key in expert.OBS_KEYS}
    )
    states: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    policy_actions: list[np.ndarray] = field(default_factory=list)
    query_count: int = 0
    correction_count: int = 0

    def append_query(self, raw_env, expert_action, policy_action) -> None:
        expert.append_current_obs(raw_env, self.observations)
        self.states.append(np.asarray(raw_env.sim.get_state().flatten(), dtype=float))
        self.actions.append(np.asarray(expert_action, dtype=float))
        self.policy_actions.append(np.asarray(policy_action, dtype=float))
        self.query_count += 1


class _CorrectionRecorder:
    """Record expert actions while delegating steps to a wrapped environment."""

    def __init__(self, env, raw_env, buffer: EpisodeBuffer):
        self._env = env
        self._raw_env = raw_env
        self._buffer = buffer

    def step(self, action):
        action = np.asarray(action, dtype=float)
        self._buffer.states.append(
            np.asarray(self._raw_env.sim.get_state().flatten(), dtype=float)
        )
        self._buffer.actions.append(action)
        # Policy actions are undefined once the expert takes control. NaNs make
        # this explicit without affecting the BC target stored in ``actions``.
        self._buffer.policy_actions.append(np.full_like(action, np.nan))
        self._buffer.correction_count += 1
        return self._env.step(action)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--source-dataset", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--aggregate-out", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument(
        "--takeover-steps",
        type=int,
        nargs="+",
        default=[80, 120, 160, 200, 240, 280, 320, 360],
    )
    parser.add_argument("--position-jitter", type=float, default=0.025)
    parser.add_argument("--yaw-jitter", type=float, default=0.012)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _evaluation_args(
    checkpoint: Path,
    *,
    base_pos: list[float],
    base_ori: list[float],
    seed: int,
) -> argparse.Namespace:
    return argparse.Namespace(
        checkpoint=checkpoint,
        factory_scene="factory_sorting_1_3fo3erfhisem",
        device="auto",
        verbose=False,
        object_name=expert.DEFAULT_OBJECT_NAME,
        site_below_offset=expert.DEFAULT_SITE_BELOW_OFFSET,
        robot_base_pos=base_pos,
        robot_base_ori=base_ori,
        renderer="mjviewer",
        camera=expert.DEFAULT_CAMERA,
        camera_height=expert.DEFAULT_CAMERA_HEIGHT,
        camera_width=expert.DEFAULT_CAMERA_WIDTH,
        controller=None,
        gripper_types="Robotiq140Gripper",
        seed=seed,
        render_sleep=0.0,
        show_object_sites=False,
        object_site_size=expert.DEFAULT_OBJECT_SITE_SIZE,
    )


def _expert_args() -> SimpleNamespace:
    return SimpleNamespace(
        up_steps=expert.DEFAULT_UP_STEPS,
        xy_steps=expert.DEFAULT_XY_STEPS,
        down_steps=expert.DEFAULT_DOWN_STEPS,
        safe_z=expert.DEFAULT_SAFE_Z,
        site_above_clearance=expert.DEFAULT_SITE_ABOVE_CLEARANCE,
        site_below_offset=expert.DEFAULT_SITE_BELOW_OFFSET,
        arrival_tolerance=expert.DEFAULT_ARRIVAL_TOLERANCE,
        gripper_end_arrival_tolerance=expert.DEFAULT_GRIPPER_END_ARRIVAL_TOLERANCE,
        settle_steps=expert.DEFAULT_SETTLE_STEPS,
        grasp_steps=expert.DEFAULT_GRASP_STEPS,
        post_success_hold_steps=expert.DEFAULT_POST_SUCCESS_HOLD_STEPS,
        max_action=expert.DEFAULT_MAX_ACTION,
        initial_view_steps=0,
        render_sleep=0.0,
        camera=expert.DEFAULT_CAMERA,
        show_object_sites=False,
        object_site_size=expert.DEFAULT_OBJECT_SITE_SIZE,
    )


def _episode_pose(index: int, count: int, position_jitter: float, yaw_jitter: float):
    if index == 0:
        return [8.0, 4.6, 0.0], [0.0, 0.0, 3.139422]
    fraction = index / max(1, count - 1)
    radius = position_jitter * math.sqrt(fraction)
    angle = index * math.pi * (3.0 - math.sqrt(5.0))
    dx = radius * math.cos(angle)
    dy = radius * math.sin(angle)
    yaw_delta = yaw_jitter * math.sin(index * 1.61803398875)
    return [8.0 + dx, 4.6 + dy, 0.0], [0.0, 0.0, 3.139422 + yaw_delta]


def _geometric_expert_action(raw_env, targets, safe_z: float):
    """Return a recovery action for the current (possibly off-policy) state."""

    robot = raw_env.robots[0]
    eef = {arm: expert.get_eef_pos(raw_env, robot, arm) for arm in expert.ARMS}
    ends = {
        arm: expert.gripper_end_center_pos(raw_env, robot, arm)
        for arm in expert.ARMS
    }
    end_distances = {
        arm: float(np.linalg.norm(ends[arm] - targets[arm]))
        for arm in expert.ARMS
    }
    if all(
        distance <= expert.DEFAULT_GRIPPER_END_ARRIVAL_TOLERANCE
        for distance in end_distances.values()
    ):
        return expert.build_action(raw_env, robot, {}, gripper_value=1.0)

    site_positions = {
        arm: targets[arm] + np.array([0.0, 0.0, expert.DEFAULT_SITE_BELOW_OFFSET])
        for arm in expert.ARMS
    }
    xy_errors = {
        arm: float(np.linalg.norm(eef[arm][:2] - site_positions[arm][:2]))
        for arm in expert.ARMS
    }
    arm_actions = {}
    for arm in expert.ARMS:
        if xy_errors[arm] > expert.DEFAULT_ARRIVAL_TOLERANCE:
            if eef[arm][2] < safe_z - expert.DEFAULT_ARRIVAL_TOLERANCE:
                target = np.array([eef[arm][0], eef[arm][1], safe_z])
            else:
                target = np.array(
                    [site_positions[arm][0], site_positions[arm][1], safe_z]
                )
            delta = target - eef[arm]
        elif eef[arm][2] > targets[arm][2] + expert.DEFAULT_ARRIVAL_TOLERANCE:
            delta = targets[arm] - eef[arm]
        else:
            delta = targets[arm] - ends[arm]
        controller_delta = expert.world_delta_to_controller_frame(robot, arm, delta)
        # A DAgger label is a one-step corrective command, not permission to
        # apply the full far-away waypoint error in one step. Keep it within
        # the range of the smooth scripted trajectories; the former 0.65
        # saturation repeated for tens of steps and destabilized fine-tuning.
        arm_actions[arm] = expert.arm_delta_to_normalized_action(
            robot=robot,
            arm=arm,
            delta_pos=controller_delta,
            max_action=QUERY_MAX_ACTION,
        )
    return expert.build_action(
        raw_env,
        robot,
        arm_actions,
        gripper_value=-1.0,
    )


def _recover_and_grasp(env, raw_env, buffer: EpisodeBuffer) -> tuple[bool, str]:
    """Apply the official staged expert from the current off-policy state."""

    args = _expert_args()
    recorder = _CorrectionRecorder(env, raw_env, buffer)
    robot = raw_env.robots[0]
    setattr(
        robot,
        expert.CAMERA_HOLD_TARGET_ATTR,
        expert.capture_camera_hold_targets(robot),
    )
    object_name = expert.DEFAULT_OBJECT_NAME
    targets, _ = expert.get_target_positions(
        raw_env,
        object_name,
        args.site_below_offset,
    )
    site_positions = {
        arm: targets[arm] + np.array([0.0, 0.0, args.site_below_offset])
        for arm in expert.ARMS
    }

    # Release any accidental one-finger contact before moving to a safe height.
    for _ in range(15):
        action = expert.build_action(raw_env, robot, {}, gripper_value=-1.0)
        expert.step_with_record(
            recorder, raw_env, action, buffer.observations, False, args
        )

    starts = {arm: expert.get_eef_pos(raw_env, robot, arm) for arm in expert.ARMS}
    safe_z = max(
        args.safe_z,
        max(starts[arm][2] for arm in expert.ARMS),
        max(
            site_positions[arm][2] + args.site_above_clearance
            for arm in expert.ARMS
        ),
    )
    safe_targets = {
        arm: np.array([starts[arm][0], starts[arm][1], safe_z])
        for arm in expert.ARMS
    }
    xy_targets = {
        arm: np.array([site_positions[arm][0], site_positions[arm][1], safe_z])
        for arm in expert.ARMS
    }

    stages = (
        (
            expert.move_along_linear_segment,
            {
                "goal_targets": safe_targets,
                "num_steps": args.up_steps,
                "reject_object_contact": False,
                "label": "corrective safe vertical lift",
            },
        ),
        (
            expert.move_along_linear_segment,
            {
                "goal_targets": xy_targets,
                "num_steps": args.xy_steps,
                "reject_object_contact": False,
                "label": "corrective XY approach",
            },
        ),
    )
    for function, keywords in stages:
        ok, reason = function(
            env=recorder,
            base_env=raw_env,
            robot=robot,
            object_name=object_name,
            gripper_value=-1.0,
            render=False,
            args=args,
            obs_buffer=buffer.observations,
            **keywords,
        )
        if not ok:
            return False, reason

    ok, reason = expert.move_vertically_below_sites(
        env=recorder,
        base_env=raw_env,
        robot=robot,
        goal_targets=targets,
        site_positions=site_positions,
        num_steps=args.down_steps,
        gripper_value=-1.0,
        render=False,
        args=args,
        obs_buffer=buffer.observations,
        label="corrective vertical descent",
    )
    if not ok:
        return False, reason

    ok, reason = expert.settle_gripper_end_centers_at_targets(
        env=recorder,
        base_env=raw_env,
        robot=robot,
        goal_targets=targets,
        gripper_value=-1.0,
        render=False,
        args=args,
        obs_buffer=buffer.observations,
        label="corrective gripper arrival",
    )
    if not ok:
        return False, reason

    for _ in range(args.grasp_steps):
        action = expert.build_action(raw_env, robot, {}, gripper_value=1.0)
        expert.step_with_record(
            recorder, raw_env, action, buffer.observations, False, args
        )
    contacts, grasps = expert.print_grasp_debug_info(
        raw_env,
        robot,
        object_name,
        targets,
        "Corrective grasp",
    )
    if not all(grasps.values()):
        return False, f"grasp_status={grasps}, contacts={contacts}"

    for _ in range(args.post_success_hold_steps):
        action = expert.build_action(raw_env, robot, {}, gripper_value=1.0)
        expert.step_with_record(
            recorder, raw_env, action, buffer.observations, False, args
        )
    return True, "bilateral grasp established"


def _collect_episode(
    *,
    policy,
    config,
    checkpoint_dict,
    checkpoint: Path,
    takeover_steps: int,
    base_pos: list[float],
    base_ori: list[float],
    seed: int,
) -> tuple[EpisodeBuffer, dict, str] | tuple[None, dict, str]:
    args = _evaluation_args(
        checkpoint,
        base_pos=base_pos,
        base_ori=base_ori,
        seed=seed,
    )
    env = evaluation.make_eval_env(
        args,
        config=config,
        ckpt_dict=checkpoint_dict,
        render=False,
    )
    metadata = {
        "takeover_steps": takeover_steps,
        "base_pos": base_pos,
        "base_ori": base_ori,
        "seed": seed,
    }
    try:
        policy.start_episode()
        obs = env.reset()
        state = env.get_state()
        obs = env.reset_to(state)
        raw_env = evaluation.base_robosuite_env(env)
        robot = raw_env.robots[0]
        setattr(
            robot,
            expert.CAMERA_HOLD_TARGET_ATTR,
            expert.capture_camera_hold_targets(robot),
        )
        targets, _ = expert.get_target_positions(
            raw_env,
            expert.DEFAULT_OBJECT_NAME,
            expert.DEFAULT_SITE_BELOW_OFFSET,
        )
        starts = {
            arm: expert.get_eef_pos(raw_env, robot, arm)
            for arm in expert.ARMS
        }
        safe_z = max(
            expert.DEFAULT_SAFE_Z,
            max(starts[arm][2] for arm in expert.ARMS),
            max(
                targets[arm][2]
                + expert.DEFAULT_SITE_BELOW_OFFSET
                + expert.DEFAULT_SITE_ABOVE_CLEARANCE
                for arm in expert.ARMS
            ),
        )
        buffer = EpisodeBuffer()
        for _ in range(takeover_steps):
            policy_action = np.asarray(policy(ob=obs), dtype=float)
            expert_action = _geometric_expert_action(raw_env, targets, safe_z)
            buffer.append_query(raw_env, expert_action, policy_action)
            obs, _, done, _ = env.step(policy_action)
            if done:
                break

        success, reason = _recover_and_grasp(env, raw_env, buffer)
        metadata.update(
            {
                "success": success,
                "reason": reason,
                "query_samples": buffer.query_count,
                "correction_samples": buffer.correction_count,
            }
        )
        if not success:
            return None, metadata, raw_env.sim.model.get_xml()
        return buffer, metadata, raw_env.sim.model.get_xml()
    finally:
        raw_to_close = evaluation.base_robosuite_env(env)
        if hasattr(raw_to_close, "close"):
            raw_to_close.close()
        gc.collect()


def _write_dataset(
    path: Path,
    episodes: list[tuple[EpisodeBuffer, dict, str]],
    *,
    source_dataset: Path,
    checkpoint: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with h5py.File(source_dataset, "r") as source, h5py.File(temporary, "w") as out:
        data = out.create_group("data")
        for key, value in source["data"].attrs.items():
            data.attrs[key] = value
        data.attrs["num_successful_demos"] = len(episodes)
        data.attrs["purpose"] = "DAgger expert labels plus successful corrective rollouts"
        data.attrs["source_checkpoint"] = str(checkpoint)
        for index, (buffer, metadata, model_xml) in enumerate(episodes, start=1):
            demo = data.create_group(f"demo_{index}")
            demo.attrs["model_file"] = model_xml
            demo.attrs["corrective_metadata"] = json.dumps(metadata, sort_keys=True)
            demo.attrs["num_samples"] = len(buffer.actions)
            demo.create_dataset("states", data=np.asarray(buffer.states))
            demo.create_dataset("actions", data=np.asarray(buffer.actions))
            demo.create_dataset(
                "policy_actions",
                data=np.asarray(buffer.policy_actions),
            )
            obs = demo.create_group("obs")
            for key, values in buffer.observations.items():
                array = np.asarray(values)
                if len(array) != len(buffer.actions):
                    raise RuntimeError(
                        f"{key}: obs={len(array)} actions={len(buffer.actions)}"
                    )
                options = (
                    {"compression": "gzip", "compression_opts": 4}
                    if key.endswith("_image")
                    else {}
                )
                obs.create_dataset(key, data=array, **options)
    temporary.replace(path)


def _decode_mask(values) -> list[str]:
    return [
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in values
    ]


def _write_mask(group, key: str, values: list[str]) -> None:
    dtype = h5py.string_dtype(encoding="utf-8")
    group.create_dataset(key, data=np.asarray(values, dtype=object), dtype=dtype)


def _merge_datasets(source_path: Path, corrections_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with (
        h5py.File(source_path, "r") as source,
        h5py.File(corrections_path, "r") as corrections,
        h5py.File(temporary, "w") as out,
    ):
        data = out.create_group("data")
        for key, value in source["data"].attrs.items():
            data.attrs[key] = value
        name_map = {}
        next_index = 1
        for dataset_name, dataset in (
            ("source", source),
            ("corrective", corrections),
        ):
            for old_name in sorted(
                dataset["data"],
                key=lambda name: int(name.split("_")[-1]),
            ):
                new_name = f"demo_{next_index}"
                dataset.copy(dataset["data"][old_name], data, name=new_name)
                data[new_name].attrs["aggregate_source"] = dataset_name
                name_map[(dataset_name, old_name)] = new_name
                next_index += 1
        data.attrs["num_successful_demos"] = next_index - 1
        data.attrs["aggregate_source_dataset"] = str(source_path)
        data.attrs["aggregate_corrective_dataset"] = str(corrections_path)

        mask = out.create_group("mask")
        if "mask" in source and "train" in source["mask"]:
            original_train = _decode_mask(source["mask"]["train"][:])
        else:
            original_train = list(source["data"])
        if "mask" in source and "valid" in source["mask"]:
            original_valid = _decode_mask(source["mask"]["valid"][:])
        else:
            original_valid = []
        train_names = [
            name_map[("source", name)] for name in original_train
        ] + [
            name_map[("corrective", name)] for name in corrections["data"]
        ]
        valid_names = [name_map[("source", name)] for name in original_valid]
        _write_mask(mask, "train", train_names)
        _write_mask(mask, "valid", valid_names)
    temporary.replace(output_path)


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    source = args.source_dataset.resolve()
    output = args.output.resolve()
    aggregate = args.aggregate_out.resolve()
    for path in (checkpoint, source):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (output, aggregate):
        if path.exists() and not args.force:
            raise FileExistsError(f"{path} exists; pass --force to replace it")

    load_args = _evaluation_args(
        checkpoint,
        base_pos=[8.0, 4.6, 0.0],
        base_ori=[0.0, 0.0, 3.139422],
        seed=args.seed,
    )
    policy, config, checkpoint_dict = evaluation.load_policy_and_config(load_args)

    episodes = []
    attempts = []
    total = len(args.takeover_steps)
    for index, takeover_steps in enumerate(args.takeover_steps):
        base_pos, base_ori = _episode_pose(
            index,
            total,
            args.position_jitter,
            args.yaw_jitter,
        )
        print(
            f"CORRECTIVE episode={index + 1}/{total} takeover={takeover_steps} "
            f"base={np.round(base_pos, 4).tolist()} yaw={base_ori[2]:.6f}",
            flush=True,
        )
        buffer, metadata, model_xml = _collect_episode(
            policy=policy,
            config=config,
            checkpoint_dict=checkpoint_dict,
            checkpoint=checkpoint,
            takeover_steps=takeover_steps,
            base_pos=base_pos,
            base_ori=base_ori,
            seed=args.seed + index,
        )
        attempts.append(metadata)
        print(json.dumps(metadata, sort_keys=True), flush=True)
        if buffer is not None:
            episodes.append((buffer, metadata, model_xml))

    if not episodes:
        raise RuntimeError(f"No successful corrective episodes: {attempts}")
    _write_dataset(
        output,
        episodes,
        source_dataset=source,
        checkpoint=checkpoint,
    )
    _merge_datasets(source, output, aggregate)
    manifest = {
        "schema": 1,
        "checkpoint": str(checkpoint),
        "source_dataset": str(source),
        "corrective_dataset": str(output),
        "aggregate_dataset": str(aggregate),
        "attempts": attempts,
        "successful_episodes": len(episodes),
    }
    manifest_path = output.parent / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    print(aggregate)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
