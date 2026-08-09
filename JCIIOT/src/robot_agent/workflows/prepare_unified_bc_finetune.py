"""Prepare a stability-focused fine-tune of the unified grasp policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = APP_ROOT / "team_submission" / "training_artifacts"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config",
        type=Path,
        default=ARTIFACTS / "unified_l1_l5_transformer_v2_task_conditioned.json",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="initialize a new policy instead of loading checkpoint weights",
    )
    parser.add_argument(
        "--task-heads",
        action="store_true",
        help="use seven task-conditioned action heads on one shared Transformer",
    )
    parser.add_argument(
        "--freeze-task-head",
        type=int,
        help="freeze the shared policy and train only this zero-based task head",
    )
    parser.add_argument(
        "--only-keys",
        nargs="+",
        help="retain only these dataset keys in the generated config",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ARTIFACTS
            / "unified_l1_l5_transformer_v3_stability_finetune.json"
        ),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=700)
    parser.add_argument("--validation-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument(
        "--experiment-name",
        default="jciiot_unified_l1_l5_lowdim_bc_v3_stability_finetune",
    )
    parser.add_argument("--focus-weight", type=float, default=3.0)
    parser.add_argument(
        "--focus-keys",
        nargs="+",
        default=["l5_center", "l5_front"],
        help="dataset keys that receive --focus-weight",
    )
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260737)
    args = parser.parse_args()

    if args.from_scratch:
        if args.checkpoint is not None:
            raise ValueError("--checkpoint and --from-scratch are mutually exclusive")
        checkpoint = None
    else:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required unless --from-scratch is used")
        checkpoint = args.checkpoint.resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    config = json.loads(args.base_config.read_text(encoding="utf-8"))
    config["experiment"]["name"] = args.experiment_name
    config["experiment"]["ckpt_path"] = (
        None if checkpoint is None else str(checkpoint)
    )
    config["experiment"]["epoch_every_n_steps"] = args.steps_per_epoch
    config["experiment"]["validation_epoch_every_n_steps"] = (
        args.validation_steps
    )
    config["experiment"]["save"]["every_n_epochs"] = args.save_every
    config["train"]["num_epochs"] = args.epochs
    config["train"]["seed"] = args.seed

    known_keys = {str(dataset["key"]) for dataset in config["train"]["data"]}
    unknown_focus = set(args.focus_keys) - known_keys
    if unknown_focus:
        raise ValueError(f"unknown focus dataset keys: {sorted(unknown_focus)}")
    if args.only_keys:
        unknown_only = set(args.only_keys) - known_keys
        if unknown_only:
            raise ValueError(f"unknown dataset keys: {sorted(unknown_only)}")
        config["train"]["data"] = [
            dataset
            for dataset in config["train"]["data"]
            if dataset["key"] in set(args.only_keys)
        ]

    # Non-focus targets remain present on every epoch to prevent forgetting;
    # selected corrective branches receive higher sampling probability.
    for dataset in config["train"]["data"]:
        dataset["weight"] = (
            args.focus_weight
            if dataset["key"] in set(args.focus_keys)
            else 1.0
        )

    learning_rate = config["algo"]["optim_params"]["policy"]["learning_rate"]
    learning_rate["initial"] = args.learning_rate
    learning_rate["scheduler_type"] = "linear"
    learning_rate["epoch_schedule"] = [args.epochs]
    learning_rate["decay_factor"] = 0.1
    if args.task_heads:
        if not args.from_scratch:
            raise ValueError("--task-heads currently requires --from-scratch")
        # Robomimic's existing schema field is used as an explicit architecture
        # marker; see BC_Transformer._create_networks. It is otherwise unused
        # by Transformer BC.
        config["algo"]["actor_layer_dims"] = [7]
    if args.freeze_task_head is not None:
        if args.from_scratch:
            raise ValueError("--freeze-task-head requires a checkpoint")
        if not 0 <= args.freeze_task_head < 7:
            raise ValueError("--freeze-task-head must be in [0, 6]")
        config["algo"]["actor_layer_dims"] = [7, args.freeze_task_head]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
