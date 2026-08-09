"""Build low-dimensional, task-conditioned data for corrected-L3 fine-tuning.

The six already validated branches are copied from their original expert
datasets.  The obsolete orange-tote L3 branch is replaced by newly collected
blue-tote demonstrations.  Generated HDF5 files live under ``training_work``
and are intentionally excluded from the submission repository.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from robot_agent.skills.bc_task_conditioning import CONDITION_KEY, TASK_OBJECTS


LOW_DIM_KEYS = (
    "robot0_left_eef_pos",
    "robot0_left_eef_quat",
    "robot0_left_gripper_qpos",
    "robot0_right_eef_pos",
    "robot0_right_eef_quat",
    "robot0_right_gripper_qpos",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-artifacts", type=Path, required=True)
    parser.add_argument("--l3-dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-out", type=Path, required=True)
    parser.add_argument("--model-output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int, default=500)
    parser.add_argument("--validation-steps", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--l3-weight", type=float, default=3.0)
    parser.add_argument("--save-every", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _sources(root: Path, l3_dataset: Path) -> tuple[tuple[str, str, Path], ...]:
    return (
        ("l1", TASK_OBJECTS[0], root / "datasets_l1_deployment_v3" / "l1_line_5_container_h01_near.hdf5"),
        ("l2", TASK_OBJECTS[1], root / "datasets_l2_deployment_v1" / "l2_green_tote_b01_upper.hdf5"),
        ("l3", TASK_OBJECTS[2], l3_dataset),
        ("l4", TASK_OBJECTS[3], root / "datasets_l4_deployment_v1" / "l4_blue_container_h01_back_upper.hdf5"),
        ("l5_back", TASK_OBJECTS[4], root / "datasets_l5_back_collisionfree_v2" / "l5_white_tote_b01_left_back.hdf5"),
        ("l5_center", TASK_OBJECTS[5], root / "datasets_l5_center_grasp_lift_v2" / "l5_white_tote_b01_left_center.hdf5"),
        ("l5_front", TASK_OBJECTS[6], root / "datasets_l5_front_deployment_v1" / "l5_white_tote_b01_left_front.hdf5"),
    )


def _copy_attrs(source, destination) -> None:
    for key, value in source.attrs.items():
        destination.attrs[key] = value


def _write_dataset(
    source_path: Path,
    destination_path: Path,
    *,
    task_index: int,
    seed: int,
    force: bool,
) -> dict:
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if destination_path.exists() and not force:
        raise FileExistsError(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_path.with_suffix(destination_path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    condition = np.zeros(len(TASK_OBJECTS), dtype=np.float32)
    condition[task_index] = 1.0

    with h5py.File(source_path, "r") as source, h5py.File(temporary, "w") as output:
        _copy_attrs(source, output)
        data_out = output.create_group("data")
        _copy_attrs(source["data"], data_out)
        data_out.attrs["bc_task_conditioning"] = (
            "7-D one-hot ordered by robot_agent.skills.bc_task_conditioning.TASK_OBJECTS"
        )
        demos = sorted(source["data"], key=lambda name: int(name.split("_")[-1]))
        samples = 0
        for demo_name in demos:
            demo_in = source["data"][demo_name]
            demo_out = data_out.create_group(demo_name)
            _copy_attrs(demo_in, demo_out)
            demo_in.copy("states", demo_out)
            demo_in.copy("actions", demo_out)
            obs_out = demo_out.create_group("obs")
            length = int(demo_in["actions"].shape[0])
            samples += length
            for key in LOW_DIM_KEYS:
                if key not in demo_in["obs"]:
                    raise RuntimeError(f"{source_path}/{demo_name}: missing {key}")
                demo_in["obs"].copy(key, obs_out)
            obs_out.create_dataset(
                "timesteps",
                data=np.arange(length, dtype=np.float32).reshape(length, 1),
            )
            obs_out.create_dataset(
                CONDITION_KEY,
                data=np.broadcast_to(condition, (length, len(condition))),
            )

        rng = np.random.default_rng(seed + task_index)
        shuffled = np.asarray(demos, dtype=object)
        rng.shuffle(shuffled)
        valid_count = max(1, int(round(0.10 * len(demos))))
        valid = sorted(shuffled[:valid_count].tolist(), key=lambda name: int(name.split("_")[-1]))
        train = sorted(shuffled[valid_count:].tolist(), key=lambda name: int(name.split("_")[-1]))
        if not train:
            raise RuntimeError(f"{source_path}: need at least two demonstrations")
        mask = output.create_group("mask")
        string_dtype = h5py.string_dtype(encoding="utf-8")
        mask.create_dataset("train", data=np.asarray(train, dtype=object), dtype=string_dtype)
        mask.create_dataset("valid", data=np.asarray(valid, dtype=object), dtype=string_dtype)

    temporary.replace(destination_path)
    return {
        "source": str(source_path.resolve()),
        "path": str(destination_path.resolve()),
        "episodes": len(demos),
        "samples": samples,
        "train_episodes": len(train),
        "valid_episodes": len(valid),
    }


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for task_index, (key, object_name, source) in enumerate(
        _sources(args.legacy_artifacts.resolve(), args.l3_dataset.resolve())
    ):
        destination = args.output_dir / f"{key}_task_conditioned.hdf5"
        audit = _write_dataset(
            source.resolve(),
            destination,
            task_index=task_index,
            seed=args.seed,
            force=args.force,
        )
        audit.update({"key": key, "object_name": object_name, "task_index": task_index})
        records.append(audit)

    checkpoint_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)
    raw_config = checkpoint_dict.get("config")
    config = json.loads(raw_config) if isinstance(raw_config, str) else dict(raw_config)
    config["experiment"]["name"] = "jciiot_unified_bc_v5_corrected_l3"
    config["experiment"]["ckpt_path"] = str(checkpoint)
    config["experiment"]["epoch_every_n_steps"] = args.steps_per_epoch
    config["experiment"]["validation_epoch_every_n_steps"] = args.validation_steps
    config["experiment"]["save"]["every_n_epochs"] = args.save_every
    config["experiment"]["save"]["epochs"] = []
    config["train"]["data"] = [
        {
            "path": record["path"],
            "eval": False,
            "key": record["key"],
            "weight": args.l3_weight if record["key"] == "l3" else 1.0,
        }
        for record in records
    ]
    config["train"]["num_epochs"] = args.epochs
    config["train"]["seed"] = args.seed
    config["train"]["cuda"] = True
    model_output_dir = (
        args.model_output_dir.resolve()
        if args.model_output_dir is not None
        else (args.output_dir.parent / "models").resolve()
    )
    model_output_dir.mkdir(parents=True, exist_ok=True)
    config["train"]["output_dir"] = str(model_output_dir)
    learning_rate = config["algo"]["optim_params"]["policy"]["learning_rate"]
    learning_rate["initial"] = args.learning_rate
    learning_rate["scheduler_type"] = "linear"
    learning_rate["epoch_schedule"] = [args.epochs]
    learning_rate["decay_factor"] = 0.1
    # Older local training runs serialized these two non-schema convenience
    # keys. Current robomimic rejects them while loading an external config;
    # the authoritative values are the experiment and train fields above.
    config["algo"]["optim_params"]["policy"].pop("num_train_batches", None)
    config["algo"]["optim_params"]["policy"].pop("num_epochs", None)

    args.config_out.parent.mkdir(parents=True, exist_ok=True)
    args.config_out.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema": 1,
        "purpose": "corrected L3 blue-tote unified BC fine-tune",
        "checkpoint": str(checkpoint),
        "task_order": list(TASK_OBJECTS),
        "records": records,
        "totals": {
            "episodes": sum(record["episodes"] for record in records),
            "samples": sum(record["samples"] for record in records),
        },
        "config": str(args.config_out.resolve()),
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["totals"]))
    print(args.config_out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
