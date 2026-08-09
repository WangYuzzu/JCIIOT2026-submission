"""Task conditioning for the single multi-task grasp policy."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


CONDITION_KEY = "bc_task_id"
TASK_OBJECTS = (
    "line_5_container_h01_near",
    "green_tote_b01_upper",
    "blue_tote_b01_near_right",
    "blue_container_h01_back_upper",
    "white_tote_b01_left_back",
    "white_tote_b01_left_center",
    "white_tote_b01_left_front",
)
TASK_INDEX = {object_name: index for index, object_name in enumerate(TASK_OBJECTS)}
# The August task correction replaces the former L3 orange tote with either
# of two geometrically equivalent blue totes on the auxiliary input table.
# They intentionally reuse the seventh-free L3 condition slot, keeping the
# deployed checkpoint observation width stable while new L3 data is collected
# and recovery-fine-tuned into that same task branch.
TASK_INDEX.update(
    {
        # Legacy pre-correction L3 dataset / checkpoint label.
        "orange_tote_b01_upper": 2,
        "blue_tote_b01_far_right": 2,
    }
)


def task_condition(object_name: str) -> np.ndarray:
    """Return the stable one-hot condition for one competition grasp target."""

    try:
        index = TASK_INDEX[object_name]
    except KeyError as exc:
        raise ValueError(f"unsupported task-conditioned BC object: {object_name}") from exc
    value = np.zeros(len(TASK_OBJECTS), dtype=np.float32)
    value[index] = 1.0
    return value


def checkpoint_uses_task_condition(checkpoint_dict: Mapping | None) -> bool:
    if not isinstance(checkpoint_dict, Mapping):
        return False
    shape_metadata = checkpoint_dict.get("shape_metadata")
    return (
        isinstance(shape_metadata, Mapping)
        and CONDITION_KEY in shape_metadata.get("all_shapes", {})
    )


class TaskConditionedPolicy:
    """Add a task one-hot to observations before calling a robomimic policy."""

    def __init__(self, policy, object_name: str) -> None:
        self.policy = policy
        self.object_name = object_name
        self.condition = task_condition(object_name)

    def start_episode(self) -> None:
        start_episode = getattr(self.policy, "start_episode", None)
        if callable(start_episode):
            start_episode()

    def __call__(self, *, ob):
        augmented = dict(ob)
        reference = np.asarray(augmented["timesteps"])
        leading_shape = reference.shape[:-1]
        augmented[CONDITION_KEY] = np.broadcast_to(
            self.condition,
            leading_shape + self.condition.shape,
        ).copy()
        return self.policy(ob=augmented)

    def __getattr__(self, name):
        return getattr(self.policy, name)


def maybe_condition_policy(policy, checkpoint_dict, object_name: str):
    """Wrap only task-conditioned checkpoints; leave legacy policies intact."""

    if not checkpoint_uses_task_condition(checkpoint_dict):
        return policy
    if isinstance(policy, TaskConditionedPolicy):
        if policy.object_name == object_name:
            return policy
        policy = policy.policy
    return TaskConditionedPolicy(policy, object_name)
