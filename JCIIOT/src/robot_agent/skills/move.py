"""Move skill — navigate the robot base to a target via A* + backend."""

from __future__ import annotations

import logging
import re
import json
import math
from collections import deque
from pathlib import Path

import numpy as np

from robot_agent.core.types import ExecutionContext, SkillResult
from robot_agent.skills.base import BaseSkill
from robot_agent.skills.pick_up import corrected_station_name

logger = logging.getLogger(__name__)


class MoveSkill(BaseSkill):
    """Navigate the mobile base to a named station or world coordinate.

    Requires a backend, scene context, and occupancy grid — no mock fallback.
    """

    def __init__(
        self,
        *,
        backend,
        scene_context,
        grid: np.ndarray,
        path_spacing: float = 0.35,
    ) -> None:
        super().__init__(
            name="move",
            description="Move to a specified location",
            keywords=(
                "move", "go", "navigate",
                "move", "go", "navigate", "travel", "drive", "approach",
            ),
        )
        self._backend = backend
        self._scene = scene_context
        self._grid = grid
        self._path_spacing = path_spacing
        self._approach_counts: dict[str, int] = {}

    # ── public API ──────────────────────────────────────────

    def run(self, context: ExecutionContext) -> SkillResult:
        target: str = (
            context.metadata.get("inputs", {}).get("target")
            or context.task
        )
        scene_name = str(getattr(self._backend, "_env_name", ""))
        target = corrected_station_name(scene_name, target)

        approach_override = self._configured_approach_override(target)
        goal_xy = (
            approach_override[0]
            if approach_override is not None
            else self._resolve_target(target)
        )
        if goal_xy is None:
            return SkillResult(
                skill_name=self.name,
                success=False,
                message=f"Cannot resolve target location: {target}",
                payload={"action": "move", "target": target},
            )

        start_xy, start_yaw = self._backend.get_base_pose()
        pre_turn_xy = approach_override[3] if approach_override is not None else None
        final_approach_max_linear = (
            approach_override[4] if approach_override is not None else None
        )
        planning_goal_xy = pre_turn_xy if pre_turn_xy is not None else goal_xy
        path = self._plan(start_xy, planning_goal_xy)
        if path is None:
            return SkillResult(
                skill_name=self.name,
                success=False,
                message=f"A* planning failed: {target}",
                payload={"action": "move", "target": target, "start": start_xy.tolist()},
            )
        if approach_override is not None and pre_turn_xy is None:
            # A* returns occupancy-cell centres. The final scene-specific BC
            # pose can lie safely within that cell but still be several
            # centimetres away, which is significant for bimanual grasping.
            # Append the exact pose as a normal physical navigation waypoint.
            if not path or float(np.linalg.norm(path[-1] - goal_xy)) > 1e-6:
                path = [*path, goal_xy.copy()]

        original_recorder = getattr(self._backend, "_record_trajectory_frame", None)
        navigation_sample_interval = 10
        sample_counter = 0

        if callable(original_recorder):
            def sampled_recorder(*args, **kwargs):
                nonlocal sample_counter
                sample_counter += 1
                if sample_counter % navigation_sample_interval == 0:
                    return original_recorder(*args, **kwargs)
                return None

            self._backend._record_trajectory_frame = sampled_recorder
        try:
            reached = self._backend.follow_path(path)
            if reached and pre_turn_xy is not None:
                # Rotate while the full arm / torso footprint is still clear
                # of the station proxy, then approach in the already-correct
                # direction. Large rotations at the close BC pose are judged
                # as collisions even when the final pose itself is valid.
                reached = self._backend.follow_path(
                    [pre_turn_xy.copy()],
                    max_steps=100,
                    waypoint_tolerance=0.002,
                )
                if reached:
                    reached = self._turn_to_configured_yaw(approach_override[1])
                if reached:
                    original_max_linear = getattr(self._backend, "_max_linear", None)
                    try:
                        if final_approach_max_linear is not None:
                            self._backend._max_linear = final_approach_max_linear
                        reached = self._backend.follow_path(
                            [goal_xy.copy()],
                            max_steps=500,
                            waypoint_tolerance=0.002,
                        )
                    finally:
                        if original_max_linear is not None:
                            self._backend._max_linear = original_max_linear
                if reached:
                    # Only a milliradian-scale terminal correction should be
                    # required here; the collision-risking sweep happened at
                    # the pre-turn point.
                    reached = self._turn_to_configured_yaw(approach_override[1])
            elif reached and approach_override is not None:
                # The normal navigation tolerance can accept the occupancy
                # cell centre without moving to a nearby (centimetre-scale)
                # BC deployment pose. Run one final, tightly-toleranced
                # physical correction so the policy sees the pose it was
                # trained around.
                reached = self._backend.follow_path(
                    [goal_xy.copy()],
                    max_steps=100,
                    waypoint_tolerance=0.002,
                )
            if reached and approach_override is not None:
                reached = self._turn_to_configured_yaw(approach_override[1])
        finally:
            if callable(original_recorder):
                self._backend._record_trajectory_frame = original_recorder
                original_recorder()
        final_xy, final_yaw = self._backend.get_base_pose()
        if reached and approach_override is not None and approach_override[2] is not None:
            counter_key = approach_override[2]
            self._approach_counts[counter_key] = self._approach_counts.get(counter_key, 0) + 1
        return SkillResult(
            skill_name=self.name,
            success=reached,
            message=f"Moved to: {target}" if reached else f"Failed to reach: {target}",
            payload={
                "action": "move",
                "target": target,
                "goal_xy": goal_xy.tolist(),
                "start_base_pose": {
                    "xy": start_xy.tolist(),
                    "yaw": float(start_yaw),
                    "robot_base_pos": [float(start_xy[0]), float(start_xy[1]), 0.0],
                    "robot_base_ori": [0.0, 0.0, float(start_yaw)],
                },
                "final_base_pose": {
                    "xy": final_xy.tolist(),
                    "yaw": float(final_yaw),
                    "robot_base_pos": [float(final_xy[0]), float(final_xy[1]), 0.0],
                    "robot_base_ori": [0.0, 0.0, float(final_yaw)],
                },
                "waypoints": len(path),
                "reached": reached,
            },
        )

    # ── internal ────────────────────────────────────────────

    def _configured_approach_override(
        self,
        target: str,
    ) -> tuple[
        np.ndarray,
        float,
        str | None,
        np.ndarray | None,
        float | None,
    ] | None:
        """Return an allowed scene-specific BC approach pose, if configured."""

        params_path = Path(__file__).resolve().parents[3] / "knowledge" / "robot_params.json"
        try:
            params = json.loads(params_path.read_text(encoding="utf-8"))
            scene_name = str(getattr(self._backend, "_env_name", ""))
            scene_overrides = (
                params.get("move_skill", {})
                .get("scene_approach_overrides", {})
                .get(scene_name, {})
            )
            for station_name, raw_entry in scene_overrides.items():
                if station_name != target and station_name not in target:
                    continue
                counter_key = None
                sequence_index = None
                if isinstance(raw_entry, list):
                    if not raw_entry:
                        raise ValueError(
                            f"empty approach sequence for {scene_name}/{station_name}"
                        )
                    counter_key = f"{scene_name}/{station_name}"
                    sequence_index = min(
                        self._approach_counts.get(counter_key, 0),
                        len(raw_entry) - 1,
                    )
                    raw_entry = raw_entry[sequence_index]
                xy = np.asarray(raw_entry["xy"], dtype=float).reshape(-1)
                if xy.size < 2:
                    raise ValueError(f"invalid approach xy for {scene_name}/{station_name}")
                yaw = float(raw_entry["yaw"])
                pre_turn_value = raw_entry.get("pre_turn_xy")
                pre_turn_xy = None
                if pre_turn_value is not None:
                    pre_turn_arr = np.asarray(pre_turn_value, dtype=float).reshape(-1)
                    if pre_turn_arr.size < 2:
                        raise ValueError(
                            f"invalid pre-turn xy for {scene_name}/{station_name}"
                        )
                    pre_turn_xy = pre_turn_arr[:2].copy()
                final_approach_max_linear = raw_entry.get(
                    "final_approach_max_linear"
                )
                if final_approach_max_linear is not None:
                    final_approach_max_linear = float(final_approach_max_linear)
                    if final_approach_max_linear <= 0:
                        raise ValueError(
                            "final_approach_max_linear must be positive for "
                            f"{scene_name}/{station_name}"
                        )
                logger.info(
                    "using configured BC approach for %s/%s%s: xy=%s "
                    "pre_turn_xy=%s yaw=%.6f",
                    scene_name,
                    station_name,
                    (
                        f"[{sequence_index}]"
                        if sequence_index is not None
                        else ""
                    ),
                    np.round(xy[:2], 4).tolist(),
                    (
                        np.round(pre_turn_xy, 4).tolist()
                        if pre_turn_xy is not None
                        else None
                    ),
                    yaw,
                )
                return (
                    xy[:2].copy(),
                    yaw,
                    counter_key,
                    pre_turn_xy,
                    final_approach_max_linear,
                )
        except Exception as exc:
            logger.warning("scene approach override unavailable: %s", exc)
        return None

    def _turn_to_configured_yaw(self, target_yaw: float) -> bool:
        """Animate the final base rotation required by a scene BC policy."""

        try:
            from robosuite.environments.factory_sorting.turn_to_station import turn_to_face_xy

            final_xy, _ = self._backend.get_base_pose()
            facing_xy = final_xy + np.array(
                [math.cos(target_yaw), math.sin(target_yaw)],
                dtype=float,
            )
            turn_params = getattr(self._backend, "_rp", {}).get("turn", {})
            recorder = getattr(self._backend, "_record_trajectory_frame", None)
            result = turn_to_face_xy(
                env=self._backend.env,
                target_xy=facing_xy,
                # Bimanual low-dimensional BC is sensitive to milliradian
                # base-yaw errors because both world-frame EEF trajectories
                # must stay synchronized. Use a dedicated tight terminal yaw
                # tolerance while retaining the normal physical turn.
                tolerance=min(float(turn_params.get("tolerance", 0.02)), 1e-3),
                max_iters=int(turn_params.get("max_iters", 8)),
                turn_steps=int(turn_params.get("turn_steps", 40)),
                settle_steps=int(turn_params.get("settle_steps", 10)),
                render=False,
                render_sleep=0.0,
                sync_attachment=True,
                post_step_callback=recorder if callable(recorder) else None,
            )
            success = bool(result.get("success"))
            if not success:
                logger.warning(
                    "configured approach yaw failed: target=%.6f result=%s",
                    target_yaw,
                    result,
                )
            return success
        except Exception:
            logger.exception("configured approach yaw failed")
            return False

    def _resolve_target(self, target: str) -> np.ndarray | None:
        """Convert a target description to a (2,) world xy position.

        Resolution order:
        1. Known station name via ``SceneContext.approach_xy()``
        2. Direct (x, y) tuple in the target string
        """
        # 1) exact named station. This must precede substring matching because
        # names such as ``aux_output_1`` contain the legacy ``output_1``.
        if target in self._scene.all_port_names():
            return self._scene.approach_xy(target)

        # 2) station mentioned in a longer natural-language target
        for name in self._scene.all_port_names():
            if name in target:
                return self._scene.approach_xy(name)

        # 3) numeric "x, y"
        nums = re.findall(r"[-+]?\d*\.?\d+", target)
        if len(nums) >= 2:
            try:
                return np.array([float(nums[0]), float(nums[1])], dtype=float)
            except ValueError:
                pass

        return None

    def _plan(
        self, start_xy: np.ndarray, goal_xy: np.ndarray,
    ) -> list[np.ndarray] | None:
        """Run A* and return a world-frame path, or None on failure."""
        from robot_agent.core.map_loader import plan_world_path
        from robot_agent.core.navigation import world_to_grid

        try:
            scene_dict = {
                "bounds": self._scene.bounds,
                "resolution": self._scene.resolution,
            }
            params_path = Path(__file__).resolve().parents[3] / "knowledge" / "robot_params.json"
            radius = 0.50
            endpoint_radius = 0.50
            minimum_radius = 0.10
            backoff_step = 0.05
            goal_bridge_max_distance = 0.80
            egress_bridge_max_distance = 1.00
            try:
                params = json.loads(params_path.read_text(encoding="utf-8"))
                move_params = params.get("move_skill", {})
                radius = max(0.0, min(float(
                    move_params.get("obstacle_inflation_radius", radius)
                ), 1.5))
                endpoint_radius = max(0.0, min(float(
                    move_params.get("endpoint_clearance_radius", endpoint_radius)
                ), 1.5))
                minimum_radius = max(0.0, min(float(
                    move_params.get("minimum_obstacle_inflation_radius", minimum_radius)
                ), radius))
                backoff_step = max(0.05, min(float(
                    move_params.get("inflation_backoff_step", backoff_step)
                ), 0.5))
                goal_bridge_max_distance = max(0.0, min(float(
                    move_params.get("goal_bridge_max_distance", goal_bridge_max_distance)
                ), 2.0))
                egress_bridge_max_distance = max(0.0, min(float(
                    move_params.get("egress_bridge_max_distance", egress_bridge_max_distance)
                ), 2.5))
            except Exception as exc:
                logger.warning("move-skill clearance config unavailable: %s", exc)

            def grid_with_clearance(
                candidate_radius: float,
                endpoints: tuple[np.ndarray, np.ndarray],
            ) -> np.ndarray:
                if candidate_radius <= 0.0:
                    planning_grid = self._grid.copy()
                else:
                    resolution = float(self._scene.resolution)
                    cells = max(1, int(np.ceil(candidate_radius / resolution)))
                    blocked = np.isin(self._grid, (1, 2))
                    horizontal = blocked.copy()
                    for offset in range(1, cells + 1):
                        horizontal[:, offset:] |= blocked[:, :-offset]
                        horizontal[:, :-offset] |= blocked[:, offset:]
                    inflated = horizontal.copy()
                    for offset in range(1, cells + 1):
                        inflated[offset:, :] |= horizontal[:-offset, :]
                        inflated[:-offset, :] |= horizontal[offset:, :]
                    planning_grid = self._grid.copy()
                    planning_grid[inflated] = 1
                    original_special = np.isin(self._grid, (3, 4))
                    planning_grid[original_special] = self._grid[original_special]

                # Approach and spawn points intentionally sit near stations.
                # Re-open only cells that were passable in the original map
                # around each endpoint so A* can enter/leave the safe corridor.
                resolution = float(self._scene.resolution)
                endpoint_cells = max(1, int(np.ceil(endpoint_radius / resolution)))
                for point in endpoints:
                    row, col = world_to_grid(
                        point[0], point[1], self._scene.bounds, resolution,
                    )
                    rows = slice(max(0, row - endpoint_cells), min(
                        planning_grid.shape[0], row + endpoint_cells + 1,
                    ))
                    cols = slice(max(0, col - endpoint_cells), min(
                        planning_grid.shape[1], col + endpoint_cells + 1,
                    ))
                    original = self._grid[rows, cols]
                    passable = np.isin(original, (0, 3, 4))
                    region = planning_grid[rows, cols]
                    region[passable] = original[passable]
                return planning_grid

            radii: list[float] = []
            candidate = radius
            while candidate > minimum_radius + 1e-9:
                radii.append(round(candidate, 6))
                candidate = max(minimum_radius, candidate - backoff_step)
            radii.append(round(minimum_radius, 6))
            if minimum_radius > 0.0:
                radii.append(0.0)

            last_error: Exception | None = None
            for candidate_radius in dict.fromkeys(radii):
                try:
                    path = plan_world_path(
                        scene_dict,
                        grid_with_clearance(candidate_radius, (start_xy, goal_xy)),
                        start_xy,
                        goal_xy,
                        min_spacing=self._path_spacing,
                    )
                    logger.info(
                        "A* selected obstacle inflation %.2fm (%d waypoints)",
                        candidate_radius,
                        len(path),
                    )
                    return path
                except RuntimeError as exc:
                    last_error = exc
                    logger.info(
                        "A* has no path with %.2fm inflation; backing off",
                        candidate_radius,
                    )
            # Auto-alignment may put the base inside a loading bay that the
            # occupancy grid intentionally disconnects from its public aisle.
            # Egress to the nearest official station approach, then plan the
            # long segment from that connected anchor.
            anchors: list[np.ndarray] = []
            for port_name in self._scene.all_port_names():
                try:
                    anchor = np.asarray(self._scene.approach_xy(port_name), dtype=float)
                except Exception:
                    continue
                distance = float(np.linalg.norm(anchor - start_xy))
                if 0.10 < distance <= 2.50:
                    anchors.append(anchor)
            anchors.sort(key=lambda point: float(np.linalg.norm(point - start_xy)))
            for anchor in anchors:
                for candidate_radius in dict.fromkeys(radii):
                    try:
                        anchored_path = plan_world_path(
                            scene_dict,
                            grid_with_clearance(candidate_radius, (anchor, goal_xy)),
                            anchor,
                            goal_xy,
                            min_spacing=self._path_spacing,
                        )
                        logger.warning(
                            "A* using station-egress anchor %s (%.2fm away), "
                            "inflation %.2fm",
                            np.round(anchor, 3).tolist(),
                            float(np.linalg.norm(anchor - start_xy)),
                            candidate_radius,
                        )
                        return [start_xy.copy(), anchor.copy(), *anchored_path[1:]]
                    except RuntimeError:
                        continue

            # A few generated maps encode an approach marker as a tiny island
            # inside collision geometry (notably L5 output_6).  Search the
            # safest inflated component reachable from a nearby station
            # anchor.  Bridge from the actual loading pose into that component,
            # but stop at its goal-side edge instead of driving the base into
            # the invalid marker.  PlaceDown aligns the carried object itself.
            from robot_agent.core.navigation import grid_to_world, nearest_passable_cell

            bridge_anchors = anchors or [start_xy]
            start_cell = world_to_grid(
                start_xy[0], start_xy[1], self._scene.bounds, self._scene.resolution,
            )
            goal_cell = world_to_grid(
                goal_xy[0], goal_xy[1], self._scene.bounds, self._scene.resolution,
            )
            for anchor in bridge_anchors:
                for candidate_radius in dict.fromkeys(radii):
                    candidate_grid = grid_with_clearance(
                        candidate_radius, (anchor, goal_xy),
                    )
                    passable = np.isin(candidate_grid, (0, 3, 4))
                    anchor_cell = world_to_grid(
                        anchor[0], anchor[1], self._scene.bounds, self._scene.resolution,
                    )
                    try:
                        anchor_cell = nearest_passable_cell(candidate_grid, anchor_cell)
                    except RuntimeError:
                        continue
                    visited = np.zeros(candidate_grid.shape, dtype=bool)
                    queue = deque([anchor_cell])
                    visited[anchor_cell] = True
                    best_goal_cell = anchor_cell
                    best_start_cell = anchor_cell
                    best_goal_distance_sq = float("inf")
                    best_start_distance_sq = float("inf")
                    while queue:
                        row, col = queue.popleft()
                        goal_distance_sq = (
                            (row - goal_cell[0]) ** 2 + (col - goal_cell[1]) ** 2
                        )
                        start_distance_sq = (
                            (row - start_cell[0]) ** 2 + (col - start_cell[1]) ** 2
                        )
                        if goal_distance_sq < best_goal_distance_sq:
                            best_goal_cell = (row, col)
                            best_goal_distance_sq = goal_distance_sq
                        if start_distance_sq < best_start_distance_sq:
                            best_start_cell = (row, col)
                            best_start_distance_sq = start_distance_sq
                        for drow, dcol in (
                            (-1, -1), (-1, 0), (-1, 1),
                            (0, -1), (0, 1),
                            (1, -1), (1, 0), (1, 1),
                        ):
                            nxt = (row + drow, col + dcol)
                            if not (
                                0 <= nxt[0] < candidate_grid.shape[0]
                                and 0 <= nxt[1] < candidate_grid.shape[1]
                            ):
                                continue
                            if visited[nxt] or not passable[nxt]:
                                continue
                            visited[nxt] = True
                            queue.append(nxt)
                    resolution = float(self._scene.resolution)
                    goal_gap = float(np.sqrt(best_goal_distance_sq)) * resolution
                    egress_gap = float(np.sqrt(best_start_distance_sq)) * resolution
                    if (
                        goal_gap > goal_bridge_max_distance
                        or egress_gap > egress_bridge_max_distance
                    ):
                        continue
                    egress_xy = grid_to_world(
                        *best_start_cell, self._scene.bounds, resolution,
                    )
                    bridge_xy = grid_to_world(
                        *best_goal_cell, self._scene.bounds, resolution,
                    )
                    try:
                        bridge_path = plan_world_path(
                            scene_dict,
                            candidate_grid,
                            egress_xy,
                            bridge_xy,
                            min_spacing=self._path_spacing,
                        )
                    except RuntimeError:
                        continue
                    logger.warning(
                        "A* safe bridge: egress %.3fm to %s, stop %.3fm before "
                        "goal at %s, inflation %.2fm",
                        egress_gap,
                        np.round(egress_xy, 3).tolist(),
                        goal_gap,
                        np.round(bridge_xy, 3).tolist(),
                        candidate_radius,
                    )
                    return [start_xy.copy(), egress_xy, *bridge_path[1:]]
            if last_error is not None:
                raise last_error
            return None
        except Exception:
            logger.exception("A* planning failed")
            return None
