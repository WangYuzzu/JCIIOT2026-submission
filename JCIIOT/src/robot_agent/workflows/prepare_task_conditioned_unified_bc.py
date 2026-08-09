"""Prepare a task-conditioned, task-balanced unified BC training run."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np

from robot_agent.skills.bc_task_conditioning import CONDITION_KEY, TASK_OBJECTS
from robot_agent.workflows.prepare_unified_bc import ARTIFACTS, DATASETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config",
        type=Path,
        default=ARTIFACTS / "unified_l1_l5_transformer_v1.json",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=ARTIFACTS / "datasets_unified_v2_task_conditioned",
    )
    parser.add_argument(
        "--config-out",
        type=Path,
        default=ARTIFACTS / "unified_l1_l5_transformer_v2_task_conditioned.json",
    )
    parser.add_argument("--epochs", type=int, default=175)
    parser.add_argument("--steps-per-epoch", type=int, default=300)
    parser.add_argument("--validation-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260736)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _conditioned_copy(
    source: Path,
    destination: Path,
    *,
    task_index: int,
    force: bool,
) -> None:
    if destination.exists() and not force:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    shutil.copy2(source, temporary)
    condition = np.zeros(len(TASK_OBJECTS), dtype=np.float32)
    condition[task_index] = 1.0
    with h5py.File(temporary, "r+") as dataset:
        dataset["data"].attrs["bc_task_conditioning"] = (
            "7-D one-hot ordered by bc_task_conditioning.TASK_OBJECTS"
        )
        for demo in dataset["data"].values():
            obs = demo["obs"]
            if CONDITION_KEY in obs:
                del obs[CONDITION_KEY]
            length = int(demo["actions"].shape[0])
            obs.create_dataset(
                CONDITION_KEY,
                data=np.broadcast_to(condition, (length, len(condition))),
            )
    temporary.replace(destination)


def main() -> int:
    args = parse_args()
    records = []
    for task_index, (label, object_name, source) in enumerate(DATASETS):
        if TASK_OBJECTS[task_index] != object_name:
            raise RuntimeError("task condition order does not match source datasets")
        destination = args.dataset_dir / f"{source.stem}_task_conditioned.hdf5"
        _conditioned_copy(
            source.resolve(),
            destination,
            task_index=task_index,
            force=args.force,
        )
        with h5py.File(destination, "r") as dataset:
            demos = list(dataset["data"])
            samples = sum(
                int(dataset["data"][demo]["actions"].shape[0]) for demo in demos
            )
            first = dataset["data"][demos[0]]["obs"][CONDITION_KEY]
            if tuple(first.shape[1:]) != (len(TASK_OBJECTS),):
                raise RuntimeError(f"invalid task condition in {destination}")
        records.append(
            {
                "label": label,
                "object_name": object_name,
                "task_index": task_index,
                "path": str(destination.resolve()),
                "episodes": len(demos),
                "samples": samples,
            }
        )

    config = json.loads(args.base_config.read_text(encoding="utf-8"))
    config["experiment"]["name"] = (
        "jciiot_unified_l1_l5_lowdim_bc_v2_task_conditioned"
    )
    config["experiment"]["epoch_every_n_steps"] = args.steps_per_epoch
    config["experiment"]["validation_epoch_every_n_steps"] = args.validation_steps
    config["experiment"]["save"]["every_n_epochs"] = 25
    config["train"]["data"] = [
        {
            "path": record["path"],
            "eval": False,
            "key": record["label"].lower().replace("-", "_"),
            "weight": 1.0,
        }
        for record in records
    ]
    config["train"]["num_epochs"] = args.epochs
    config["train"]["seed"] = args.seed
    # robomimic's MetaDataset does not implement global observation
    # normalization. Explicit task conditioning is sufficient to disambiguate
    # the seven policies without modifying third-party training code.
    config["train"]["hdf5_normalize_obs"] = False
    obs_keys = config["observation"]["modalities"]["obs"]["low_dim"]
    if CONDITION_KEY not in obs_keys:
        obs_keys.append(CONDITION_KEY)
    policy_optim = config["algo"]["optim_params"]["policy"]
    policy_optim["learning_rate"]["epoch_schedule"] = [args.epochs]

    args.config_out.parent.mkdir(parents=True, exist_ok=True)
    args.config_out.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": 2,
        "purpose": "one explicitly task-conditioned BC checkpoint for L1-L5",
        "task_condition_key": CONDITION_KEY,
        "task_order": list(TASK_OBJECTS),
        "task_sampling": "equal target probability",
        "observation_normalization": False,
        "records": records,
        "totals": {
            "targets": len(records),
            "episodes": sum(record["episodes"] for record in records),
            "samples": sum(record["samples"] for record in records),
        },
        "config": str(args.config_out.resolve()),
    }
    manifest_path = args.dataset_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(manifest_path.resolve())
    print(args.config_out.resolve())
    print(json.dumps(manifest["totals"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
