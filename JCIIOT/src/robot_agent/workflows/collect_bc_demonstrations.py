"""Collect hidden-marker BC demonstrations for the five competition tasks.

This workflow deliberately reuses the official scripted demonstration
collector.  The script controller is used only to create training data; task
execution still goes through the robomimic BC checkpoint and the official
grasp / lift verification path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import h5py


APP_ROOT = Path(__file__).resolve().parents[3]
ROBOSUITE_INNER = APP_ROOT / "robosuite"
for path in (APP_ROOT, ROBOSUITE_INNER):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

import robosuite as suite  # noqa: E402
from robosuite.environments.factory_sorting import (  # noqa: E402,F401
    factory_sorting_1_3fo3erfhisem,
    factory_sorting_3_3fo3errph7x9,
    factory_sorting_5_3fo3ertpxeut,
    factory_sorting_7_3fo3erfky9rn,
    factory_sorting_9_3fo3ert2c5fp,
)
from robosuite.environments.factory_sorting import (  # noqa: E402
    lift_after_grasp as expert_lift,
    load_factory_sorting_1_3fo3erfhisem_collect as official,
)
from robosuite.wrappers import DataCollectionWrapper  # noqa: E402
from robosuite.environments.factory_sorting.turn_to_station import (  # noqa: E402
    set_base_world_yaw_direct,
    set_base_xy_direct,
    zero_base_velocity,
)


@dataclass(frozen=True)
class DemoTarget:
    level: str
    env_name: str
    source: str
    object_name: str
    base_pos: tuple[float, float, float]
    base_ori: tuple[float, float, float]

    @property
    def key(self) -> str:
        return f"{self.level.lower()}_{self.object_name}"


TARGETS = (
    DemoTarget(
        "L1",
        "FactorySorting1_3FO3ERFHISEM",
        "input_5",
        "line_5_container_h01_near",
        (8.0, 4.619, 0.0),
        (0.0, 0.0, -3.139453),
    ),
    DemoTarget(
        "L2",
        "FactorySorting3_3FO3ERRPH7X9",
        "input_6",
        "green_tote_b01_upper",
        (13.0, 3.932, 0.0),
        (0.0, 0.0, -3.139453),
    ),
    # The 2026-08-09 SOP correction moves L3 to the auxiliary input table and
    # replaces the former orange tote with a blue tote.  This is the exact
    # collision-free stance used by deployment after navigation.
    DemoTarget(
        "L3",
        "FactorySorting5_3FO3ERTPXEUT",
        "aux_input_1",
        "blue_tote_b01_near_right",
        (1.45681, 8.473143, 0.0),
        (0.0, 0.0, math.pi),
    ),
    DemoTarget(
        "L4",
        "FactorySorting7_3FO3ERFKY9RN",
        "input_2",
        "blue_container_h01_back_upper",
        (-8.3, 5.01, 0.0),
        (0.0, 0.0, -3.14),
    ),
    DemoTarget(
        "L5",
        "FactorySorting9_3FO3ERT2C5FP",
        "input_1",
        "white_tote_b01_left_back",
        (-13.1, 5.01, 0.0),
        (0.0, 0.0, -3.14),
    ),
    DemoTarget(
        "L5",
        "FactorySorting9_3FO3ERT2C5FP",
        "input_1",
        "white_tote_b01_left_center",
        (-13.1, 5.01, 0.0),
        (0.0, 0.0, -3.14),
    ),
    DemoTarget(
        "L5",
        "FactorySorting9_3FO3ERT2C5FP",
        "input_1",
        "white_tote_b01_left_front",
        (-13.1, 5.01, 0.0),
        (0.0, 0.0, -3.14),
    ),
)


def _collector_args(
    target: DemoTarget,
    rollouts: int,
    seed: int,
    *,
    base_pos: tuple[float, float, float] | None = None,
    base_ori: tuple[float, float, float] | None = None,
) -> SimpleNamespace:
    """Mirror official collector defaults, with runtime-consistent images."""

    return SimpleNamespace(
        num_rollouts=rollouts,
        object_name=target.object_name,
        up_steps=official.DEFAULT_UP_STEPS,
        xy_steps=official.DEFAULT_XY_STEPS,
        down_steps=official.DEFAULT_DOWN_STEPS,
        safe_z=official.DEFAULT_SAFE_Z,
        # Tall / upper-tier bins need room for the fingers below the EEF
        # during the horizontal approach.  More than 0.20 m places the EEF
        # outside Tiago's bilateral reachable workspace at the collision-free
        # shelf stance; the custom side-entry path supplies the remaining rim
        # clearance.
        site_above_clearance=(
            official.DEFAULT_SITE_ABOVE_CLEARANCE if target.level == "L1" else 0.20
        ),
        site_below_offset=(
            official.DEFAULT_SITE_BELOW_OFFSET if target.level == "L1" else 0.055
        ),
        arrival_tolerance=(
            official.DEFAULT_ARRIVAL_TOLERANCE if target.level == "L1" else 0.03
        ),
        gripper_end_arrival_tolerance=(
            official.DEFAULT_GRIPPER_END_ARRIVAL_TOLERANCE
            if target.level == "L1"
            else 0.02
        ),
        settle_steps=(
            official.DEFAULT_SETTLE_STEPS if target.level == "L1" else 200
        ),
        grasp_steps=official.DEFAULT_GRASP_STEPS,
        post_success_hold_steps=official.DEFAULT_POST_SUCCESS_HOLD_STEPS,
        # Upper-tier bins topple when one arm reaches the rim well before the
        # other.  A smaller Cartesian action cap plus the tighter bilateral
        # tolerance above keeps the final approach synchronized.
        max_action=official.DEFAULT_MAX_ACTION if target.level == "L1" else 0.35,
        initial_view_steps=official.DEFAULT_INITIAL_VIEW_STEPS,
        render_sleep=0.0,
        camera_height=official.DEFAULT_CAMERA_HEIGHT,
        camera_width=official.DEFAULT_CAMERA_WIDTH,
        # Runtime evaluation hides these debug markers, so training does too.
        show_object_sites=False,
        object_site_size=official.DEFAULT_OBJECT_SITE_SIZE,
        robot_base_pos=list(base_pos or target.base_pos),
        robot_base_ori=list(base_ori or target.base_ori),
        directory="",
        output_name=target.key,
        renderer="mjviewer",
        camera=official.DEFAULT_CAMERA,
        controller=None,
        gripper_types="Robotiq140Gripper",
        seed=seed,
        no_render=True,
    )


def _env_kwargs(args: SimpleNamespace) -> dict:
    kwargs = official.make_env_kwargs(args, render=False)
    # The competition backend enables the physical material objects.
    kwargs["include_material_objects"] = True
    kwargs["has_renderer"] = False
    kwargs["has_offscreen_renderer"] = True
    return kwargs


def _first_observation_digest(observations) -> str:
    """Hash the first frame across every available observation key."""

    digest = hashlib.sha256()
    for key in sorted(observations):
        values = observations[key]
        if len(values) == 0:
            continue
        digest.update(key.encode("utf-8"))
        first = np.asarray(values[0])
        digest.update(str(first.dtype).encode("ascii"))
        digest.update(np.asarray(first.shape, dtype=np.int64).tobytes())
        digest.update(first.tobytes())
    return digest.hexdigest()


def _external_penetrations(raw_env) -> tuple[int, int, int]:
    """Count unsafe robot contacts for one candidate base pose."""

    hard: set[str] = set()
    proxies: set[str] = set()
    external_names: set[str] = set()
    model = raw_env.sim.model
    data = raw_env.sim.data
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        if float(contact.dist) >= -1e-4:
            continue
        first = model.geom_id2name(int(contact.geom1)) or ""
        second = model.geom_id2name(int(contact.geom2)) or ""
        first_robot = first.startswith(("robot0_", "gripper0_"))
        second_robot = second.startswith(("robot0_", "gripper0_"))
        if first_robot == second_robot:
            continue
        robot_geom = first if first_robot else second
        external = second if first_robot else first
        if "floor" in external.lower():
            continue
        external_names.add(external)
        if external.startswith("scene_aabb_proxy_"):
            proxies.add(external)
        if any(token in robot_geom for token in ("base", "wheel", "caster", "torso")):
            hard.add(f"{robot_geom}->{external}")
    return len(hard), len(proxies), len(external_names)


def _remove_completed_l5_totes(raw_env, target: DemoTarget) -> None:
    """Relocate totes already completed earlier in the sequential L5 task."""

    if target.level != "L5":
        return
    completed_before = {
        "white_tote_b01_left_back": (),
        "white_tote_b01_left_center": ("white_tote_b01_left_back",),
        "white_tote_b01_left_front": (
            "white_tote_b01_left_back",
            "white_tote_b01_left_center",
        ),
    }
    for index, object_name in enumerate(completed_before.get(target.object_name, ())):
        for suffix in ("_joint0", "_free"):
            joint_name = f"{object_name}{suffix}"
            try:
                qpos = np.asarray(
                    raw_env.sim.data.get_joint_qpos(joint_name),
                    dtype=float,
                ).copy()
                qpos[:3] = np.array([-20.0 - index, -20.0, 1.0])
                raw_env.sim.data.set_joint_qpos(joint_name, qpos)
                try:
                    raw_env.sim.data.set_joint_qvel(
                        joint_name,
                        np.zeros(6, dtype=float),
                    )
                except Exception:
                    pass
                break
            except Exception:
                continue
    raw_env.sim.forward()


def _derive_aligned_pose(
    target: DemoTarget,
    args: SimpleNamespace,
    *,
    relocate_completed_l5_totes: bool,
) -> tuple:
    """Find a collision-free two-arm stance and whether sites must be swapped."""

    probe = suite.make(env_name=target.env_name, **_env_kwargs(args))
    try:
        raw = probe
        raw.reset()
        if relocate_completed_l5_totes:
            _remove_completed_l5_totes(raw, target)
        robot = raw.robots[0]
        authored, _ = official.get_target_positions(
            raw,
            target.object_name,
            official.DEFAULT_SITE_BELOW_OFFSET,
        )
        midpoint = 0.5 * (authored["right"][:2] + authored["left"][:2])
        authored_right_axis = authored["right"][:2] - authored["left"][:2]
        norm = float(np.linalg.norm(authored_right_axis))
        if norm <= 1e-6:
            raise RuntimeError(f"{target.object_name}: coincident grasp sites")
        authored_right_axis /= norm

        seed_xy = np.asarray(target.base_pos[:2], dtype=float)
        candidates = []
        for reversed_side in (False, True):
            right_axis = authored_right_axis * (-1.0 if reversed_side else 1.0)
            yaw = float(np.arctan2(right_axis[0], -right_axis[1]))
            forward = np.array([np.cos(yaw), np.sin(yaw)], dtype=float)
            for standoff in (0.65, 0.80, 0.90, 0.95, 1.10):
                for lateral_offset in (0.0, -0.30, 0.30, -0.45, 0.45, -0.60, 0.60):
                    xy = midpoint - standoff * forward + lateral_offset * right_axis
                    set_base_world_yaw_direct(raw, robot, yaw)
                    set_base_xy_direct(raw, robot, xy)
                    zero_base_velocity(raw, robot)
                    raw.sim.forward()
                    hard, proxy, external = _external_penetrations(raw)
                    candidates.append(
                        {
                            "xy": xy.copy(),
                            "yaw": yaw,
                            "reversed": reversed_side,
                            "standoff": standoff,
                            "lateral_offset": lateral_offset,
                            "hard": hard,
                            "proxy": proxy,
                            "external": external,
                            "travel": float(np.linalg.norm(xy - seed_xy)),
                        }
                    )
        preferred_standoff = 0.65 if target.level == "L1" else 0.80
        candidates.sort(
            key=lambda item: (
                item["hard"],
                item["proxy"],
                item["external"],
                # Bilateral reach is more sensitive to lateral offset than to
                # a small standoff difference.  Prefer the most centered
                # collision-free stance and let the arms cover the remaining
                # forward distance.
                abs(item["lateral_offset"]),
                abs(item["standoff"] - preferred_standoff),
                item["travel"],
            )
        )
        selected = candidates[0]
        if selected["hard"] or selected["proxy"]:
            raise RuntimeError(
                f"{target.object_name}: no collision-free BC stance; best={selected}"
            )
        pos = (
            float(selected["xy"][0]),
            float(selected["xy"][1]),
            0.0,
        )
        ori = (0.0, 0.0, float(selected["yaw"]))
        print(
            f"[{target.level}] aligned BC stance: pos={np.round(pos, 4).tolist()} "
            f"yaw={ori[2]:.6f} swap_sites={selected['reversed']}",
            flush=True,
        )
        return pos, ori, bool(selected["reversed"])
    finally:
        probe.close()


def _make_rollout_variations(
    *,
    rollouts: int,
    seed: int,
    position_jitter: float,
    yaw_jitter: float,
    timing_jitter: float,
) -> list[dict[str, float]]:
    """Return deterministic reset / trajectory variations, including baseline."""

    rng = np.random.default_rng(seed)
    variations = [{
        "world_dx": 0.0,
        "world_dy": 0.0,
        "yaw_delta": 0.0,
        "timing_scale": 1.0,
    }]
    # A radial design covers the local workspace better than independent
    # Gaussian noise and still gives reproducible, bounded demonstrations.
    phase = float(rng.uniform(0.0, 2.0 * math.pi))
    for index in range(1, rollouts):
        fraction = (index - 0.5) / max(1.0, rollouts - 1.0)
        radius = position_jitter * math.sqrt(min(1.0, fraction))
        angle = phase + index * math.pi * (3.0 - math.sqrt(5.0))
        variations.append({
            "world_dx": radius * math.cos(angle),
            "world_dy": radius * math.sin(angle),
            "yaw_delta": float(rng.uniform(-yaw_jitter, yaw_jitter)),
            "timing_scale": float(rng.uniform(1.0 - timing_jitter, 1.0 + timing_jitter)),
        })
    return variations


def _append_expert_lift(
    env,
    *,
    object_name: str,
    observations: dict[str, list],
    args: SimpleNamespace,
    lift_height: float,
    hold_steps: int,
) -> tuple[bool, str]:
    """Record a physical two-arm lift after the scripted training grasp."""

    raw_env = env.unwrapped
    robot = raw_env.robots[0]
    initial_status = official.grasp_status(raw_env, robot, object_name)
    if not all(initial_status.values()):
        return False, f"expert lift started without bilateral grasp: {initial_status}"

    start_object_z = float(expert_lift.object_center_pos(raw_env, object_name)[2])
    target_object_z = start_object_z + float(lift_height)
    target_gripper_positions = {
        arm: expert_lift.gripper_end_center_pos(raw_env, robot, arm)
        + np.array([0.0, 0.0, lift_height], dtype=float)
        for arm in expert_lift.ARMS
    }
    hold_targets = expert_lift.capture_hold_targets(robot)
    success = False

    for _ in range(300):
        current_object_z = float(expert_lift.object_center_pos(raw_env, object_name)[2])
        if current_object_z >= target_object_z - 0.02:
            success = True
            break
        robot.composite_controller.update_state()
        arm_actions = {}
        for arm in expert_lift.ARMS:
            world_delta = (
                target_gripper_positions[arm]
                - expert_lift.gripper_end_center_pos(raw_env, robot, arm)
            )
            controller_delta = expert_lift.world_delta_to_controller_frame(
                robot,
                arm,
                world_delta,
            )
            arm_actions[arm] = expert_lift.arm_delta_to_normalized_action(
                robot,
                arm,
                controller_delta,
                0.80,
            )
        action = expert_lift.build_action(
            robot,
            arm_actions=arm_actions,
            gripper_value=1.0,
            hold_targets=hold_targets,
        )
        official.step_with_record(
            env,
            raw_env,
            action,
            observations,
            False,
            args,
        )

    final_status = official.grasp_status(raw_env, robot, object_name)
    final_z = float(expert_lift.object_center_pos(raw_env, object_name)[2])
    if not success or not all(final_status.values()):
        return (
            False,
            "expert lift failed: "
            f"lifted={final_z - start_object_z:.6f}, grasp_status={final_status}",
        )

    env.successful = True
    for _ in range(max(1, int(hold_steps))):
        action = expert_lift.build_action(
            robot,
            arm_actions={},
            gripper_value=1.0,
            hold_targets=hold_targets,
        )
        official.step_with_record(
            env,
            raw_env,
            action,
            observations,
            False,
            args,
        )
    final_status = official.grasp_status(raw_env, robot, object_name)
    final_z = float(expert_lift.object_center_pos(raw_env, object_name)[2])
    if not all(final_status.values()):
        env.successful = False
        return (
            False,
            "expert lift hold lost grasp: "
            f"lifted={final_z - start_object_z:.6f}, grasp_status={final_status}",
        )
    return True, f"expert physical lift held object by {final_z - start_object_z:.6f} m"


def collect_target(
    target: DemoTarget,
    *,
    rollouts: int,
    seed: int,
    output_dir: Path,
    force: bool,
    position_jitter: float,
    yaw_jitter: float,
    timing_jitter: float,
    pose_source: str,
    lowdim_only: bool,
    relocate_completed_l5_totes: bool,
    descent_mode: str,
    expert_lift_height: float,
    expert_lift_hold_steps: int,
    pose_overrides: dict[str, tuple[tuple[float, float, float], tuple[float, float, float], bool]],
) -> dict:
    output_path = output_dir / f"{target.key}.hdf5"
    if output_path.exists() and not force:
        raise FileExistsError(f"{output_path} already exists (pass --force to replace it)")

    seed_args = _collector_args(target, rollouts, seed)
    override = pose_overrides.get(target.level)
    if override is not None:
        geometric_pos, geometric_ori, swap_sites = override
    else:
        geometric_pos, geometric_ori, swap_sites = _derive_aligned_pose(
            target,
            seed_args,
            relocate_completed_l5_totes=relocate_completed_l5_totes,
        )
    effective_pose_source = pose_source
    if override is not None:
        aligned_pos, aligned_ori, swap_sites = override
        effective_pose_source = "override"
    elif pose_source == "deployment":
        # The formal runtime passes the navigation endpoint to pick_up and
        # forces the yaw listed for this target.  Near ±pi, an equivalent
        # positive/negative yaw can produce opposite-sign EEF quaternions.
        # Collecting on the exact deployment branch prevents that otherwise
        # invisible quaternion discontinuity from becoming policy OOD.
        aligned_pos = target.base_pos
        aligned_ori = target.base_ori
    else:
        aligned_pos = geometric_pos
        aligned_ori = geometric_ori
    print(
        f"[{target.level}] collection pose source={effective_pose_source}: "
        f"pos={np.round(aligned_pos, 4).tolist()} yaw={aligned_ori[2]:.6f} "
        f"(geometric yaw={geometric_ori[2]:.6f})",
        flush=True,
    )
    args = _collector_args(
        target,
        rollouts,
        seed,
        base_pos=aligned_pos,
        base_ori=aligned_ori,
    )
    env_kwargs = _env_kwargs(args)
    if lowdim_only:
        env_kwargs["use_camera_obs"] = False
        env_kwargs["has_offscreen_renderer"] = False
    raw_env = suite.make(env_name=target.env_name, **env_kwargs)

    temp_root = Path(tempfile.mkdtemp(prefix=f"jciiot_bc_{target.key}_"))
    original_obs_keys = official.OBS_KEYS
    if lowdim_only:
        official.OBS_KEYS = tuple(
            key for key in original_obs_keys if not key.endswith("_image")
        )
    wrapped = DataCollectionWrapper(raw_env, str(temp_root), collect_freq=1, flush_freq=1000)
    successes = 0
    obs_cache: dict = {}
    successful_variations: list[dict[str, float]] = []
    episode_variations: dict[str, dict[str, float]] = {}
    rollout_variations = _make_rollout_variations(
        rollouts=rollouts,
        seed=seed,
        position_jitter=position_jitter,
        yaw_jitter=yaw_jitter,
        timing_jitter=timing_jitter,
    )
    nominal_steps = {
        "up_steps": int(args.up_steps),
        "xy_steps": int(args.xy_steps),
        "down_steps": int(args.down_steps),
    }
    active_variation = rollout_variations[0]
    applied_variation: dict[str, float] = dict(active_variation)
    original_wrapped_reset = wrapped.reset

    def reset_with_pose_variation(*positional, **keyword):
        nonlocal applied_variation
        observation = original_wrapped_reset(*positional, **keyword)
        if relocate_completed_l5_totes:
            _remove_completed_l5_totes(raw_env, target)
        base_xy = np.asarray(aligned_pos[:2], dtype=float)
        requested_offset = np.array(
            [active_variation["world_dx"], active_variation["world_dy"]],
            dtype=float,
        )
        requested_yaw_delta = float(active_variation["yaw_delta"])
        # Back off a candidate that introduces a reset penetration rather than
        # accepting an invalid demonstration or aborting the complete batch.
        for factor in (1.0, 0.5, 0.0):
            xy = base_xy + factor * requested_offset
            yaw = float(aligned_ori[2]) + factor * requested_yaw_delta
            set_base_world_yaw_direct(raw_env, raw_env.robots[0], yaw)
            set_base_xy_direct(raw_env, raw_env.robots[0], xy)
            zero_base_velocity(raw_env, raw_env.robots[0])
            raw_env.sim.forward()
            hard, proxy, _ = _external_penetrations(raw_env)
            if hard == 0 and proxy == 0:
                applied_variation = {
                    **active_variation,
                    "world_dx": float(factor * requested_offset[0]),
                    "world_dy": float(factor * requested_offset[1]),
                    "yaw_delta": float(factor * requested_yaw_delta),
                    "collision_backoff_factor": float(factor),
                }
                break
        else:
            raise RuntimeError(f"{target.object_name}: no safe reset pose for variation")
        print(
            f"[{target.level}] reset variation: "
            f"{json.dumps(applied_variation, sort_keys=True)}",
            flush=True,
        )
        return observation

    wrapped.reset = reset_with_pose_variation
    original_get_targets = official.get_target_positions
    original_move_segment = official.move_along_linear_segment
    original_settle_gripper_ends = official.settle_gripper_end_centers_at_targets
    original_vertical_descent = official.move_vertically_below_sites
    original_gripper_touches_object = official.gripper_touches_object

    def get_training_targets(env, object_name, site_below_offset):
        targets, names = original_get_targets(env, object_name, site_below_offset)
        if swap_sites and object_name == target.object_name:
            targets = {"right": targets["left"], "left": targets["right"]}
            names = {"right": names["left"], "left": names["right"]}
        if (
            target.level == "L3"
            and object_name.startswith("blue_tote_b01_")
        ):
            # The August L3 tote is rotated 90 degrees. Its authored markers
            # lie along the front wall, while Tiago's finger closing axis at
            # the collision-free x-side stance is parallel to that wall. Use
            # the two lateral walls instead so each gripper can physically
            # straddle one rim. This is expert-data geometry only; deployment
            # remains a learned BC rollout with no marker observations.
            center_id = env.sim.model.site_name2id(f"{object_name}_center_site")
            center = np.asarray(env.sim.data.site_xpos[center_id], dtype=float)
            approach_sign = float(np.sign(aligned_pos[0] - center[0])) or 1.0
            grasp_x = center[0] + approach_sign * 0.08
            lateral = 0.293
            grasp_z = center[2] - float(site_below_offset)
            targets = {
                "right": np.array([grasp_x, center[1] + lateral, grasp_z]),
                "left": np.array([grasp_x, center[1] - lateral, grasp_z]),
            }
            return targets, names
        if target.level != "L1" and object_name == target.object_name:
            # Some upper-tier objects author both grasp sites on the aisle
            # side that is blocked by shelf geometry.  If the collision-free
            # base is on the opposite side, mirror the targets through the
            # object center so the arms grasp the physically accessible wall
            # instead of pushing through the entire bin.
            center_xy = 0.5 * (targets["right"][:2] + targets["left"][:2])
            try:
                center_id = env.sim.model.site_name2id(f"{object_name}_center_site")
                center_xy = np.asarray(env.sim.data.site_xpos[center_id][:2], dtype=float)
                base_side = np.asarray(aligned_pos[:2], dtype=float) - center_xy
                for arm in ("right", "left"):
                    target_side = np.asarray(targets[arm][:2], dtype=float) - center_xy
                    if float(np.dot(target_side, base_side)) < 0.0:
                        targets[arm] = targets[arm].copy()
                        targets[arm][:2] = center_xy - target_side
            except Exception as exc:
                print(
                    f"[{target.level}] accessible-side target reflection skipped: {exc}",
                    flush=True,
                )
            # The authored marker is just outside the tote wall.  Move the
            # gripper end-center target slightly through that surface so the
            # closing fingerpads straddle the real collision wall instead of
            # stopping a centimetre short of it.
            for arm in ("right", "left"):
                toward_center = center_xy - targets[arm][:2]
                norm = float(np.linalg.norm(toward_center))
                if norm > 1e-6:
                    targets[arm] = targets[arm].copy()
                    targets[arm][:2] += 0.03 * toward_center / norm
            midpoint = 0.5 * (targets["right"][:2] + targets["left"][:2])
            for arm in ("right", "left"):
                inward = midpoint - targets[arm][:2]
                norm = float(np.linalg.norm(inward))
                if norm > 1e-6:
                    targets[arm] = targets[arm].copy()
                    targets[arm][:2] += 0.04 * inward / norm
        return targets, names

    official.get_target_positions = get_training_targets
    if target.level != "L1":
        # Side-entry trajectories intentionally skim the outer rim while the
        # grippers are open. Final success still requires fingerpad contacts.
        official.gripper_touches_object = lambda *args, **kwargs: False

    def move_training_segment(*positional, **keyword):
        """Use a looser tolerance only for non-contact transit waypoints.

        Upper-tier tote sites force a high initial clearance.  One Tiago arm
        can stop a few centimetres below that arbitrary waypoint even though
        both arms remain safely above the rim and can continue toward the
        object.  The final side entry and physical grasp checks stay strict.
        """

        label = str(keyword.get("label", "segment"))
        call_args = keyword.get("args")
        if call_args is None and len(positional) > 8:
            call_args = positional[8]
        transit = target.level != "L1" and label in {
            "safe vertical lift",
            "high outward clearance",
        }
        corrected_l3_descent = (
            target.level == "L3"
            and target.object_name.startswith("blue_tote_b01_")
            and label == "vertical descent below sites"
        )
        if call_args is None or not (transit or corrected_l3_descent):
            return original_move_segment(*positional, **keyword)
        original_tolerance = float(call_args.arrival_tolerance)
        call_args.arrival_tolerance = max(
            original_tolerance,
            0.07 if corrected_l3_descent else 0.055,
        )
        try:
            return original_move_segment(*positional, **keyword)
        finally:
            call_args.arrival_tolerance = original_tolerance

    official.move_along_linear_segment = move_training_segment

    def side_entry_descent(
        env,
        base_env,
        robot,
        goal_targets,
        site_positions,
        num_steps,
        gripper_value,
        render,
        args,
        obs_buffer,
        label="side-entry descent",
    ):
        use_vertical_descent = (
            target.level == "L1"
            or descent_mode == "vertical"
            or (
                descent_mode == "auto"
                and (
                    target.level in {"L2", "L3", "L4"}
                    or (
                        target.level == "L5"
                        and target.object_name == "white_tote_b01_left_back"
                    )
                )
            )
        )
        if use_vertical_descent:
            original_tolerance = float(args.arrival_tolerance)
            if (
                target.level == "L3"
                and target.object_name.startswith("blue_tote_b01_")
            ):
                args.arrival_tolerance = max(original_tolerance, 0.07)
            try:
                return original_vertical_descent(
                    env,
                    base_env,
                    robot,
                    goal_targets,
                    site_positions,
                    num_steps,
                    gripper_value,
                    render,
                    args,
                    obs_buffer,
                    label=label,
                )
            finally:
                args.arrival_tolerance = original_tolerance

        midpoint = 0.5 * (goal_targets["right"][:2] + goal_targets["left"][:2])
        # Descend with both grippers outside the accessible object wall, then
        # enter horizontally together.  Moving each gripper away from the
        # inter-arm midpoint would spread them left/right but would not clear
        # the wall that caused the premature collision.
        outward = {}
        corrected_l3_tote = (
            target.level == "L3"
            and target.object_name.startswith("blue_tote_b01_")
        )
        if corrected_l3_tote:
            # Descend outside the two lateral walls, then close in from
            # opposite sides. Approaching both hands from the robot-facing x
            # side would push this rotated tote off its support before grasp.
            for arm in ("right", "left"):
                outward[arm] = goal_targets[arm].copy()
                lateral_direction = goal_targets[arm][:2] - midpoint
                lateral_direction /= max(
                    float(np.linalg.norm(lateral_direction)),
                    1e-6,
                )
                outward[arm][:2] += 0.12 * lateral_direction
        else:
            approach_direction = np.asarray(aligned_pos[:2], dtype=float) - midpoint
            approach_direction /= max(
                float(np.linalg.norm(approach_direction)),
                1e-6,
            )
            for arm in ("right", "left"):
                outward[arm] = goal_targets[arm].copy()
                outward[arm][:2] += 0.18 * approach_direction

        starts = {
            arm: official.get_eef_pos(base_env, robot, arm)
            for arm in ("right", "left")
        }
        high_outward = {
            arm: np.array([outward[arm][0], outward[arm][1], starts[arm][2]])
            for arm in ("right", "left")
        }
        stages = (
            ("high outward clearance", high_outward, max(30, num_steps // 2)),
            ("outside-rim descent", outward, num_steps),
            ("horizontal side entry", goal_targets, max(60, num_steps)),
        )
        for stage_label, stage_targets, stage_steps in stages:
            ok, reason = official.move_along_linear_segment(
                env=env,
                base_env=base_env,
                robot=robot,
                object_name=target.object_name,
                goal_targets=stage_targets,
                num_steps=stage_steps,
                gripper_value=gripper_value,
                render=render,
                args=args,
                obs_buffer=obs_buffer,
                reject_object_contact=False,
                label=stage_label,
            )
            if not ok:
                if corrected_l3_tote and stage_label == "horizontal side entry":
                    print(
                        f"[L3] lateral rim contact boundary reached: {reason}",
                        flush=True,
                    )
                    return True, ""
                return ok, reason
        return True, ""

    official.move_vertically_below_sites = side_entry_descent

    def settle_training_gripper_ends(*positional, **keyword):
        ok, reason = original_settle_gripper_ends(*positional, **keyword)
        if not ok and target.level != "L1":
            # High-tier tote collision walls stop the end-center controller
            # just outside its mathematical target. Continue to the physical
            # close; rollout success still requires real bilateral contacts.
            print(f"[{target.level}] end-center settle reached contact boundary: {reason}")
            return True, ""
        return ok, reason

    official.settle_gripper_end_centers_at_targets = settle_training_gripper_ends
    try:
        for index in range(rollouts):
            active_variation = rollout_variations[index]
            timing_scale = float(active_variation["timing_scale"])
            for field, nominal in nominal_steps.items():
                setattr(args, field, max(1, int(round(nominal * timing_scale))))
            if expert_lift_height > 0:
                # The successful marker is recorded only after the physical
                # expert lift passes, not immediately after gripper closure.
                args.post_success_hold_steps = 0
            print(
                f"\n[{target.level}] {target.object_name}: "
                f"rollout {index + 1}/{rollouts}",
                flush=True,
            )
            success, reason, episode_dir, observations = official.rollout_once(
                wrapped,
                render=False,
                args=args,
            )
            if success and expert_lift_height > 0:
                wrapped.successful = False
                success, reason = _append_expert_lift(
                    wrapped,
                    object_name=target.object_name,
                    observations=observations,
                    args=args,
                    lift_height=expert_lift_height,
                    hold_steps=expert_lift_hold_steps,
                )
            print(f"[{target.level}] result: {reason}", flush=True)
            if not success:
                robot = wrapped.unwrapped.robots[0]
                debug_positions = {}
                for arm in ("right", "left"):
                    for group, geom_names in robot.gripper[arm].important_geoms.items():
                        if "fingerpad" not in group:
                            continue
                        debug_positions[f"{arm}.{group}"] = [
                            np.round(
                                wrapped.unwrapped.sim.data.geom_xpos[
                                    wrapped.unwrapped.sim.model.geom_name2id(name)
                                ],
                                4,
                            ).tolist()
                            for name in geom_names
                        ]
                print(
                    f"[{target.level}] fingerpad positions: "
                    f"{json.dumps(debug_positions, sort_keys=True)}",
                    flush=True,
                )
                geom_debug = {}
                sim = wrapped.unwrapped.sim
                for name in official.object_collision_geoms(
                    wrapped.unwrapped,
                    target.object_name,
                ):
                    geom_id = sim.model.geom_name2id(name)
                    geom_debug[name] = {
                        "pos": np.round(sim.data.geom_xpos[geom_id], 4).tolist(),
                        "size": np.round(sim.model.geom_size[geom_id], 4).tolist(),
                        "xmat": np.round(
                            sim.data.geom_xmat[geom_id].reshape(3, 3),
                            3,
                        ).tolist(),
                    }
                print(
                    f"[{target.level}] object collision geoms: "
                    f"{json.dumps(geom_debug, sort_keys=True)}",
                    flush=True,
                )
            if success:
                successes += 1
                episode_key = str(Path(episode_dir).resolve())
                obs_cache[episode_key] = observations
                episode_variations[episode_key] = dict(applied_variation)
                successful_variations.append(dict(applied_variation))
    except BaseException:
        official.OBS_KEYS = original_obs_keys
        raise
    finally:
        official.get_target_positions = original_get_targets
        official.move_along_linear_segment = original_move_segment
        official.settle_gripper_end_centers_at_targets = original_settle_gripper_ends
        official.move_vertically_below_sites = original_vertical_descent
        official.gripper_touches_object = original_gripper_touches_object
        wrapped.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        gathered_path, saved = official.gather_successful_demonstrations_as_hdf5(
            str(temp_root),
            str(output_dir),
            hdf5_name=output_path.name,
            env_name=target.env_name,
            env_kwargs=env_kwargs,
            policy_info={
                **vars(args),
                "source": target.source,
                "workflow": "collect_bc_demonstrations.py",
                "script_controller_used_for_training_data_only": True,
                "position_jitter_m": position_jitter,
                "yaw_jitter_rad": yaw_jitter,
                "timing_jitter_fraction": timing_jitter,
                "pose_source": effective_pose_source,
                "lowdim_only": lowdim_only,
                "successful_variations": successful_variations,
            },
            obs_cache=obs_cache,
        )
        # The bundled training loader requires these standard robomimic length
        # attributes, but the supplied FactorySorting collector omits them.
        with h5py.File(gathered_path, "a") as dataset_file:
            data_group = dataset_file["data"]
            total = 0
            variation_by_observation_digest = {}
            for episode_key, observations in obs_cache.items():
                digest = _first_observation_digest(observations)
                variation_by_observation_digest[digest] = episode_variations[episode_key]
            ordered_variations = []
            for demo_group in data_group.values():
                num_samples = int(demo_group["actions"].shape[0])
                demo_group.attrs["num_samples"] = num_samples
                total += num_samples
                digest = _first_observation_digest(demo_group["obs"])
                variation = variation_by_observation_digest.get(digest)
                if variation is None:
                    raise RuntimeError("could not bind collection variation to saved demo")
                demo_group.attrs["collection_variation"] = json.dumps(variation, sort_keys=True)
                ordered_variations.append(variation)
            data_group.attrs["total"] = total
            policy_info = json.loads(data_group.attrs["policy_info"])
            policy_info["successful_variations"] = ordered_variations
            data_group.attrs["policy_info"] = json.dumps(policy_info)
    finally:
        official.OBS_KEYS = original_obs_keys
    successful_variations = ordered_variations
    shutil.rmtree(temp_root, ignore_errors=True)

    record = {
        "level": target.level,
        "env_name": target.env_name,
        "source": target.source,
        "object_name": target.object_name,
        "semantic_approach_seed": list(target.base_pos),
        "aligned_base_pos": list(aligned_pos),
        "aligned_base_ori": list(aligned_ori),
        "swapped_grasp_sites": swap_sites,
        "attempts": rollouts,
        "successes": successes,
        "saved_demos": saved,
        "dataset": str(Path(gathered_path).resolve()),
        "show_object_sites": False,
        "position_jitter_m": position_jitter,
        "yaw_jitter_rad": yaw_jitter,
        "timing_jitter_fraction": timing_jitter,
        "pose_source": effective_pose_source,
        "lowdim_only": lowdim_only,
        "geometric_base_pos": list(geometric_pos),
        "geometric_base_ori": list(geometric_ori),
        "successful_variations": successful_variations,
        "relocate_completed_l5_totes": relocate_completed_l5_totes,
        "descent_mode": descent_mode,
        "expert_lift_height": expert_lift_height,
        "expert_lift_hold_steps": expert_lift_hold_steps,
    }
    print(json.dumps(record, ensure_ascii=False), flush=True)
    if saved != rollouts:
        raise RuntimeError(
            f"{target.key}: expected {rollouts} successful demos, saved {saved}"
        )
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", nargs="+", default=["L1", "L2", "L3", "L4", "L5"])
    parser.add_argument(
        "--objects",
        nargs="+",
        default=None,
        help="optional exact object-name filter within the selected levels",
    )
    parser.add_argument("--rollouts", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--position-jitter", type=float, default=0.06)
    parser.add_argument("--yaw-jitter", type=float, default=0.035)
    parser.add_argument("--timing-jitter", type=float, default=0.10)
    parser.add_argument(
        "--lowdim-only",
        action="store_true",
        help="omit RGB rendering / storage for a low-dimensional BC dataset",
    )
    parser.add_argument(
        "--pose-source",
        choices=("deployment", "aligned"),
        default="deployment",
        help=(
            "deployment uses DemoTarget base pose / yaw, matching the formal "
            "navigation-to-grasp path; aligned uses the geometric search pose"
        ),
    )
    parser.add_argument(
        "--pose-override",
        action="append",
        default=[],
        metavar="LEVEL:X:Y:YAW:SWAP",
        help=(
            "training-only pose pilot override; may be repeated. SWAP is "
            "true/false and controls whether left/right grasp sites are exchanged"
        ),
    )
    parser.add_argument(
        "--keep-completed-l5-totes",
        action="store_true",
        help=(
            "leave all L5 totes at their reset poses. This matches the fresh "
            "BC environment created by the official runtime for every grasp."
        ),
    )
    parser.add_argument(
        "--descent-mode",
        choices=("auto", "vertical", "side"),
        default="auto",
        help=(
            "training expert approach. auto uses vertical descent for L1-L4 "
            "and the L5 back tote, side entry for the other L5 totes."
        ),
    )
    parser.add_argument(
        "--expert-lift-height",
        type=float,
        default=0.0,
        help="append a recorded physical two-arm lift of this height to each demo",
    )
    parser.add_argument(
        "--expert-lift-hold-steps",
        type=int,
        default=20,
        help="recorded hold steps after a successful expert lift",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=APP_ROOT / "team_submission" / "training_artifacts" / "datasets",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pose_overrides: dict[
        str,
        tuple[tuple[float, float, float], tuple[float, float, float], bool],
    ] = {}
    for raw_override in args.pose_override:
        fields = str(raw_override).split(":")
        if len(fields) != 5:
            raise ValueError(
                f"invalid --pose-override {raw_override!r}; "
                "expected LEVEL:X:Y:YAW:SWAP"
            )
        level, x, y, yaw, raw_swap = fields
        normalized_level = level.upper()
        if normalized_level not in {"L1", "L2", "L3", "L4", "L5"}:
            raise ValueError(f"invalid pose override level: {level!r}")
        if raw_swap.strip().lower() not in {"true", "false", "1", "0", "yes", "no"}:
            raise ValueError(f"invalid pose override SWAP: {raw_swap!r}")
        pose_overrides[normalized_level] = (
            (float(x), float(y), 0.0),
            (0.0, 0.0, float(yaw)),
            raw_swap.strip().lower() in {"true", "1", "yes"},
        )
    requested = {level.upper() for level in args.levels}
    unknown = requested - {"L1", "L2", "L3", "L4", "L5"}
    if unknown:
        raise ValueError(f"unknown levels: {sorted(unknown)}")
    if args.rollouts < 1:
        raise ValueError("--rollouts must be >= 1")
    if min(args.position_jitter, args.yaw_jitter, args.timing_jitter) < 0:
        raise ValueError("jitter values must be non-negative")
    if args.expert_lift_height < 0 or args.expert_lift_hold_steps < 0:
        raise ValueError("expert lift settings must be non-negative")
    if args.timing_jitter >= 1:
        raise ValueError("--timing-jitter must be less than 1")

    records = []
    requested_objects = set(args.objects or ())
    for index, target in enumerate(TARGETS):
        if target.level not in requested:
            continue
        if requested_objects and target.object_name not in requested_objects:
            continue
        records.append(
            collect_target(
                target,
                rollouts=args.rollouts,
                seed=args.seed + index,
                output_dir=args.output_dir,
                force=args.force,
                position_jitter=args.position_jitter,
                yaw_jitter=args.yaw_jitter,
                timing_jitter=args.timing_jitter,
                pose_source=args.pose_source,
                lowdim_only=args.lowdim_only,
                relocate_completed_l5_totes=not args.keep_completed_l5_totes,
                descent_mode=args.descent_mode,
                expert_lift_height=args.expert_lift_height,
                expert_lift_hold_steps=args.expert_lift_hold_steps,
                pose_overrides=pose_overrides,
            )
        )

    manifest = {
        "schema": 1,
        "purpose": "BC fine-tuning data; scripted controller is not used at task runtime",
        "records": records,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nManifest: {manifest_path.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
