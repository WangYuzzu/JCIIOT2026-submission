"""Prepare one task-balanced BC model for all L1--L5 grasp targets.

The seven source datasets remain separate in the robomimic configuration.
``MetaDataset`` then supplies a weighted sampler, and
``normalize_weights_by_ds_size=True`` gives every target equal probability
despite their different episode lengths.  Only L1 needs a converted copy:
its RGB demonstrations predate the common ``timesteps`` observation used by
the other six policies.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


APP_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = APP_ROOT / "team_submission" / "training_artifacts"

LOW_DIM_KEYS = (
    "robot0_left_eef_pos",
    "robot0_left_eef_quat",
    "robot0_left_gripper_qpos",
    "robot0_right_eef_pos",
    "robot0_right_eef_quat",
    "robot0_right_gripper_qpos",
    "timesteps",
)

DATASETS = (
    (
        "L1",
        "line_5_container_h01_near",
        ARTIFACTS
        / "datasets_unified_v1"
        / "l1_line_5_container_h01_near_timestep_lowdim.hdf5",
    ),
    (
        "L2",
        "green_tote_b01_upper",
        ARTIFACTS
        / "datasets_l2_deployment_v1"
        / "l2_green_tote_b01_upper_timestep.hdf5",
    ),
    (
        "L3",
        "orange_tote_b01_upper",
        ARTIFACTS
        / "datasets_l3_collisionfree_v2"
        / "l3_orange_tote_b01_upper_timestep.hdf5",
    ),
    (
        "L4",
        "blue_container_h01_back_upper",
        ARTIFACTS
        / "datasets_l4_deployment_v1"
        / "l4_blue_container_h01_back_upper_timestep.hdf5",
    ),
    (
        "L5-back",
        "white_tote_b01_left_back",
        ARTIFACTS
        / "datasets_l5_back_collisionfree_v2"
        / "l5_white_tote_b01_left_back_timestep.hdf5",
    ),
    (
        "L5-center",
        "white_tote_b01_left_center",
        ARTIFACTS
        / "datasets_l5_center_grasp_lift_v2"
        / "l5_white_tote_b01_left_center_timestep.hdf5",
    ),
    (
        "L5-front",
        "white_tote_b01_left_front",
        ARTIFACTS
        / "datasets_l5_front_deployment_v1"
        / "l5_white_tote_b01_left_front_timestep.hdf5",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-config",
        type=Path,
        default=(
            ARTIFACTS
            / "scratch_lowdim_transformer_l3_collisionfree_v2_timestep.json"
        ),
    )
    parser.add_argument(
        "--l1-source",
        type=Path,
        default=(
            ARTIFACTS
            / "datasets_l1_deployment_v3"
            / "l1_line_5_container_h01_near.hdf5"
        ),
    )
    parser.add_argument(
        "--config-out",
        type=Path,
        default=ARTIFACTS / "unified_l1_l5_transformer_v1.json",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=ARTIFACTS / "datasets_unified_v1" / "manifest.json",
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--validation-steps", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260735)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _decode(values) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ]


def _copy_attrs(source, destination) -> None:
    for key, value in source.attrs.items():
        destination.attrs[key] = value


def _copy_l1_lowdim_with_timestep(
    source_path: Path,
    destination_path: Path,
    *,
    force: bool,
) -> None:
    if destination_path.exists() and not force:
        return
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_path.with_suffix(destination_path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()

    with (
        h5py.File(source_path, "r") as source,
        h5py.File(temporary, "w") as destination,
    ):
        _copy_attrs(source, destination)
        data_out = destination.create_group("data")
        _copy_attrs(source["data"], data_out)
        data_out.attrs["unified_conversion"] = (
            "low-dimensional observations plus absolute episode timestep"
        )

        for demo_name in sorted(
            source["data"],
            key=lambda name: int(name.split("_")[-1]),
        ):
            demo_in = source["data"][demo_name]
            demo_out = data_out.create_group(demo_name)
            _copy_attrs(demo_in, demo_out)
            for key in ("states", "actions"):
                demo_in.copy(key, demo_out)

            obs_in = demo_in["obs"]
            obs_out = demo_out.create_group("obs")
            length = int(demo_in["actions"].shape[0])
            for key in LOW_DIM_KEYS:
                if key == "timesteps":
                    obs_out.create_dataset(
                        key,
                        data=np.arange(length, dtype=np.float32).reshape(length, 1),
                    )
                    continue
                obs_in.copy(key, obs_out)

        if "mask" in source:
            source.copy("mask", destination)

    temporary.replace(destination_path)


def _audit_dataset(label: str, object_name: str, path: Path) -> dict:
    required = set(LOW_DIM_KEYS)
    with h5py.File(path, "r") as dataset:
        demos = sorted(
            dataset["data"],
            key=lambda name: int(name.split("_")[-1]),
        )
        if not demos:
            raise RuntimeError(f"{label}: no demonstrations in {path}")
        sample_count = 0
        action_shape = None
        for demo_name in demos:
            demo = dataset["data"][demo_name]
            obs_keys = set(demo["obs"])
            missing = sorted(required - obs_keys)
            if missing:
                raise RuntimeError(
                    f"{label}/{demo_name}: missing observations {missing}"
                )
            length = int(demo["actions"].shape[0])
            if int(demo["obs"]["timesteps"].shape[0]) != length:
                raise RuntimeError(f"{label}/{demo_name}: timestep length mismatch")
            current_shape = tuple(demo["actions"].shape[1:])
            if action_shape is None:
                action_shape = current_shape
            elif action_shape != current_shape:
                raise RuntimeError(f"{label}: inconsistent action shapes")
            sample_count += length

        if "mask" not in dataset:
            raise RuntimeError(f"{label}: train/valid masks are missing")
        train = _decode(dataset["mask"]["train"][:])
        valid = _decode(dataset["mask"]["valid"][:])
        if not train or not valid or set(train) & set(valid):
            raise RuntimeError(f"{label}: invalid train/valid split")

    return {
        "label": label,
        "object_name": object_name,
        "path": str(path.resolve()),
        "episodes": len(demos),
        "samples": sample_count,
        "train_episodes": len(train),
        "valid_episodes": len(valid),
        "action_shape": list(action_shape or ()),
        "observations": list(LOW_DIM_KEYS),
    }


def _write_config(args: argparse.Namespace, records: list[dict]) -> None:
    config = json.loads(args.base_config.read_text(encoding="utf-8"))
    config["experiment"]["name"] = "jciiot_unified_l1_l5_lowdim_bc_v1"
    config["experiment"]["ckpt_path"] = None
    config["experiment"]["validate"] = True
    config["experiment"]["rollout"]["enabled"] = False
    config["experiment"]["epoch_every_n_steps"] = args.steps_per_epoch
    config["experiment"]["validation_epoch_every_n_steps"] = (
        args.validation_steps
    )
    config["experiment"]["save"]["enabled"] = True
    config["experiment"]["save"]["every_n_epochs"] = 25
    config["experiment"]["save"]["on_best_validation"] = False
    config["experiment"]["save"]["on_best_rollout_return"] = False
    config["experiment"]["save"]["on_best_rollout_success_rate"] = False
    config["experiment"]["logging"]["terminal_output_to_txt"] = False
    config["experiment"]["logging"]["log_tb"] = False

    config["train"]["data"] = [
        {
            "path": record["path"],
            "eval": False,
            "key": record["label"].lower().replace("-", "_"),
            "weight": 1.0,
        }
        for record in records
    ]
    config["train"]["normalize_weights_by_ds_size"] = True
    # MetaDataset intentionally rejects cache_mode="all", because it first
    # computes one global action normalization across every child dataset.
    config["train"]["hdf5_cache_mode"] = "low_dim"
    config["train"]["hdf5_load_next_obs"] = False
    config["train"]["hdf5_normalize_obs"] = False
    config["train"]["num_data_workers"] = 0
    config["train"]["num_epochs"] = args.epochs
    config["train"]["batch_size"] = args.batch_size
    config["train"]["seed"] = args.seed
    config["train"]["hdf5_filter_key"] = "train"
    config["train"]["hdf5_validation_filter_key"] = "valid"
    config["train"]["frame_stack"] = 10
    config["train"]["seq_length"] = 1
    config["train"]["output_dir"] = str(
        (ARTIFACTS / "models").resolve()
    )

    config["observation"]["modalities"]["obs"] = {
        "low_dim": list(LOW_DIM_KEYS),
        "rgb": [],
        "depth": [],
        "scan": [],
    }

    transformer = config["algo"]["transformer"]
    transformer["enabled"] = True
    transformer["context_length"] = 10
    transformer["embed_dim"] = 256
    transformer["num_layers"] = 4
    transformer["num_heads"] = 8
    transformer["emb_dropout"] = 0.0
    transformer["attn_dropout"] = 0.0
    transformer["block_output_dropout"] = 0.0
    config["algo"]["actor_layer_dims"] = []
    config["algo"]["rnn"]["enabled"] = False
    config["algo"]["loss"]["l2_weight"] = 1.0
    config["algo"]["loss"]["l1_weight"] = 0.1
    config["algo"]["loss"]["cos_weight"] = 0.0

    policy_optim = config["algo"]["optim_params"]["policy"]
    policy_optim["optimizer_type"] = "adam"
    policy_optim["regularization"]["L2"] = 0.0
    policy_optim["learning_rate"]["initial"] = args.learning_rate
    policy_optim["learning_rate"]["scheduler_type"] = "linear"
    policy_optim["learning_rate"]["epoch_schedule"] = [args.epochs]
    policy_optim["learning_rate"]["decay_factor"] = 0.02
    policy_optim.pop("num_train_batches", None)
    policy_optim.pop("num_epochs", None)

    args.config_out.parent.mkdir(parents=True, exist_ok=True)
    args.config_out.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    l1_destination = DATASETS[0][2]
    _copy_l1_lowdim_with_timestep(
        args.l1_source.resolve(),
        l1_destination,
        force=args.force,
    )

    records = [
        _audit_dataset(label, object_name, path.resolve())
        for label, object_name, path in DATASETS
    ]
    action_shapes = {tuple(record["action_shape"]) for record in records}
    if action_shapes != {(20,)}:
        raise RuntimeError(f"expected one 20-D action schema, got {action_shapes}")

    _write_config(args, records)
    manifest = {
        "schema": 1,
        "purpose": "one task-balanced BC checkpoint for all L1-L5 grasp targets",
        "task_sampling": (
            "equal target probability via MetaDataset and "
            "normalize_weights_by_ds_size=true"
        ),
        "explicit_task_id": False,
        "task_identity_signal": (
            "world-frame EEF positions/quaternions plus episode timestep"
        ),
        "records": records,
        "totals": {
            "targets": len(records),
            "episodes": sum(record["episodes"] for record in records),
            "samples": sum(record["samples"] for record in records),
            "train_episodes": sum(
                record["train_episodes"] for record in records
            ),
            "valid_episodes": sum(
                record["valid_episodes"] for record in records
            ),
        },
        "config": str(args.config_out.resolve()),
    }
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.manifest_out.resolve())
    print(args.config_out.resolve())
    print(json.dumps(manifest["totals"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
