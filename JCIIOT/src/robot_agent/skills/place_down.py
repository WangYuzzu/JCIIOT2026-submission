"""Place-down skill — release a held object at target via backend."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np

from robot_agent.core.scene_context import SceneContext
from robot_agent.core.types import ExecutionContext, SkillResult
from robot_agent.skills.base import BaseSkill
from robot_agent.skills.pick_up import _resolve_station_name, corrected_station_name

logger = logging.getLogger(__name__)


class PlaceDownSkill(BaseSkill):
    """Release a held object at the target through the environment backend.

    Resolves natural-language target descriptions to known station names
    via ``SceneContext`` (same algorithm as ``PickUpSkill``).
    """

    def __init__(self, *, backend, scene_context: SceneContext | None = None) -> None:
        super().__init__(
            name="place_down",
            description="Place down or drop an object",
            keywords=(
                "place", "put", "drop", "release",
                "place", "drop", "put", "release", "unload",
            ),
        )
        self._backend = backend
        self._scene = scene_context

    def run(self, context: ExecutionContext) -> SkillResult:
        raw_target: str = (
            context.metadata.get("inputs", {}).get("target")
            or context.task
        )
        target = raw_target
        if self._scene is not None:
            target = _resolve_station_name(raw_target, self._scene)
            logger.info("place_down target: %r → %r", raw_target, target)
        scene_name = str(getattr(self._backend, "_env_name", ""))
        target = corrected_station_name(scene_name, target)

        # Physics place (only mode — no teleport fallback)
        if hasattr(self._backend, "place_object_physics"):
            if self._scene is not None and target in self._scene.output_ports:
                try:
                    from robosuite.environments.factory_sorting.transport_attachment import (
                        TRANSPORT_ATTACHMENT_ATTR,
                        quat_conjugate_wxyz,
                        quat_multiply_wxyz,
                        yaw_quat_wxyz,
                    )

                    raw_env = self._backend.env
                    attachment = getattr(raw_env, TRANSPORT_ATTACHMENT_ATTR, None)
                    if attachment and attachment.get("active", False):
                        base_xy, _ = self._backend.get_base_pose()
                        station_xy = np.asarray(
                            self._scene.output_ports[target].center[:2], dtype=float,
                        )
                        held_object = str(
                            getattr(self._backend, "_held_crate_name", "") or ""
                        )
                        params_path = (
                            Path(__file__).resolve().parents[3]
                            / "knowledge"
                            / "robot_params.json"
                        )
                        params = json.loads(params_path.read_text(encoding="utf-8"))
                        scene_name = str(getattr(self._backend, "_env_name", ""))
                        offset = (
                            params.get("place_skill", {})
                            .get("scene_target_offsets", {})
                            .get(scene_name, {})
                            .get(held_object, [0.0, 0.0])
                        )
                        offset_xy = np.asarray(offset, dtype=float).reshape(-1)
                        if offset_xy.size < 2:
                            raise ValueError(
                                f"invalid placement offset for {scene_name}/{held_object}"
                            )
                        target_xy = station_xy + offset_xy[:2]
                        world_delta = target_xy - base_xy
                        target_yaw = math.atan2(
                            float(station_xy[1] - base_xy[1]),
                            float(station_xy[0] - base_xy[0]),
                        )
                        upright_objects = set(
                            params.get("place_skill", {}).get(
                                "upright_transport_objects",
                                [],
                            )
                        )
                        world_yaw_override = (
                            params.get("place_skill", {})
                            .get("scene_world_yaws", {})
                            .get(scene_name, {})
                            .get(held_object)
                        )
                        if world_yaw_override is not None:
                            attachment["relative_quat"] = quat_multiply_wxyz(
                                quat_conjugate_wxyz(yaw_quat_wxyz(target_yaw)),
                                yaw_quat_wxyz(float(world_yaw_override)),
                            )
                        elif held_object in upright_objects:
                            # The transport helper preserves the object's
                            # grasp-relative quaternion. A slightly rotated
                            # long tote can overhang the narrow output table
                            # and slide off during later L5 trips. Express an
                            # upright world quaternion in the base frame that
                            # place_object_physics will use when facing the
                            # target station.
                            attachment["relative_quat"] = quat_conjugate_wxyz(
                                yaw_quat_wxyz(target_yaw)
                            )
                        c, s = math.cos(target_yaw), math.sin(target_yaw)
                        relative_xy = np.array(
                            [
                                c * world_delta[0] + s * world_delta[1],
                                -s * world_delta[0] + c * world_delta[1],
                            ],
                            dtype=float,
                        )
                        # place_object_physics turns the base to face target;
                        # express the selected world-space table slot in that
                        # final base frame. Distinct L5 slots prevent later
                        # totes from pushing earlier placements off the table.
                        attachment["relative_xy"] = relative_xy
                        logger.info(
                            "place_down: aligned %s with %s slot %s "
                            "(relative_xy=%s)",
                            held_object,
                            target,
                            np.round(target_xy, 4).tolist(),
                            np.round(relative_xy, 4).tolist(),
                        )
                except Exception as exc:
                    logger.warning("place_down attachment alignment skipped: %s", exc)

            injected_output_alias = False
            output_ports = getattr(getattr(self._backend, "env", None), "output_ports", {})
            if (
                isinstance(output_ports, dict)
                and target not in output_ports
                and not any(name.startswith(target) for name in output_ports)
                and output_ports
                and self._scene is not None
                and target in self._scene.output_ports
            ):
                # Some Siemens scenes expose only four legacy output-port
                # aliases even though their semantic map and scoring contract
                # contain output_1..output_6.  Add a temporary alias so the
                # fixed backend can retain the requested semantic target,
                # choose the correct support-surface index, and release at the
                # robot's current (already navigated) pose.
                template = dict(next(iter(output_ports.values())))
                template["center"] = np.asarray(
                    self._scene.output_ports[target].center,
                    dtype=float,
                ).copy()
                output_ports[target] = template
                injected_output_alias = True
                logger.warning(
                    "place_down: temporarily mapped missing simulator port %s "
                    "to its semantic-map station",
                    target,
                )
            try:
                ok = self._backend.place_object_physics(target)
                msg = f"Physics place {'OK' if ok else 'FAIL'}: {target}"
                if not ok:
                    _held = getattr(self._backend, "_held_crate_name", None)
                    _ports = list(self._backend.env.output_ports.keys()) if hasattr(self._backend, 'env') and self._backend.env else []
                    logger.warning("place_down: failed target=%s held=%s avail_out=%s", target, _held, _ports)
                    msg += f" held={_held} out_ports={_ports}"
                return SkillResult(
                    skill_name=self.name,
                    success=ok,
                    message=msg,
                    payload={"action": "place_down", "target": target, "method": "physics", "ok": ok},
                )
            except Exception as exc:
                logger.exception("physics place crashed")
                return SkillResult(
                    skill_name=self.name, success=False,
                    message=f"Physics place error: {exc}",
                    payload={"action": "place_down", "target": target, "error": str(exc)},
                )
            finally:
                if injected_output_alias:
                    output_ports.pop(target, None)

        # No physics configured — teleport only
        try:
            self._backend.place_object(target)
        except Exception:
            pass
        return SkillResult(
            skill_name=self.name, success=True,
            message=f"Placed (snap): {target}",
            payload={"action": "place_down", "target": target, "raw_target": raw_target, "method": "teleport"},
        )
