"""Prepare comparable robomimic BC fine-tune or scratch-training configs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[3]
for path in (APP_ROOT, APP_ROOT / "robosuite"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from robomimic.utils.file_utils import load_dict_from_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    artifacts = APP_ROOT / "team_submission" / "training_artifacts"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=APP_ROOT / "robosuite" / "robosuite" / "model_epoch_150.pth",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=(
            artifacts
            / "datasets_l1_diverse_v1"
            / "l1_line_5_container_h01_near.hdf5"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=artifacts / "models")
    parser.add_argument("--config-out", type=Path, default=artifacts / "finetune_config.json")
    parser.add_argument(
        "--initialization",
        choices=("fine_tune", "scratch"),
        default="fine_tune",
    )
    parser.add_argument("--name")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=50)
    parser.add_argument("--validation-steps", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--lowdim-only",
        action="store_true",
        help=(
            "Train without RGB observations. This is only architecture-safe "
            "with --initialization scratch."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    dataset = args.dataset.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    if args.lowdim_only and args.initialization != "scratch":
        raise ValueError("--lowdim-only requires --initialization scratch")

    checkpoint_data = load_dict_from_checkpoint(str(checkpoint))
    config = json.loads(checkpoint_data["config"])
    default_name = f"jciiot_l1_rgb_bc_{args.initialization}_diverse_v1"
    config["experiment"]["name"] = args.name or default_name
    config["experiment"]["ckpt_path"] = (
        str(checkpoint) if args.initialization == "fine_tune" else None
    )
    config["experiment"]["validate"] = True
    config["experiment"]["epoch_every_n_steps"] = args.steps_per_epoch
    config["experiment"]["validation_epoch_every_n_steps"] = args.validation_steps
    config["experiment"]["rollout"]["enabled"] = False
    config["experiment"]["render"] = False
    config["experiment"]["render_video"] = False
    config["experiment"]["logging"]["terminal_output_to_txt"] = False
    config["experiment"]["logging"]["log_tb"] = False
    config["experiment"]["save"]["enabled"] = True
    config["experiment"]["save"]["every_n_epochs"] = args.save_every
    # robomimic writes a new full checkpoint on every improvement. Validation
    # is still logged each epoch, while fixed-interval checkpoints keep disk
    # use bounded and are compared by real simulator rollouts afterwards.
    config["experiment"]["save"]["on_best_validation"] = False
    config["experiment"]["save"]["on_best_rollout_return"] = False
    config["experiment"]["save"]["on_best_rollout_success_rate"] = False

    config["train"]["data"] = [{"path": str(dataset), "eval": False}]
    config["train"]["output_dir"] = str(args.output_dir.resolve())
    config["train"]["num_epochs"] = args.epochs
    config["train"]["batch_size"] = args.batch_size
    config["train"]["num_data_workers"] = 0
    config["train"]["cuda"] = True
    config["train"]["seed"] = args.seed
    config["train"]["hdf5_filter_key"] = "train"
    config["train"]["hdf5_validation_filter_key"] = "valid"
    if args.lowdim_only:
        config["observation"]["modalities"]["obs"]["rgb"] = []

    policy_optim = config["algo"]["optim_params"]["policy"]
    policy_optim["learning_rate"]["initial"] = (
        args.learning_rate
        if args.learning_rate is not None
        else (2e-5 if args.initialization == "fine_tune" else 1e-4)
    )
    policy_optim["learning_rate"]["epoch_schedule"] = [args.epochs]

    # The supplied checkpoint was produced by a nearby robomimic revision
    # whose scheduler stored these two derived fields in optim_params. The
    # bundled trainer derives them from train.num_epochs and key-locks the
    # config, so remove only the incompatible cached fields.
    for optim_params in config.get("algo", {}).get("optim_params", {}).values():
        if isinstance(optim_params, dict):
            optim_params.pop("num_train_batches", None)
            optim_params.pop("num_epochs", None)

    args.config_out.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.config_out.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "config": str(args.config_out.resolve()),
                "initialization": args.initialization,
                "checkpoint": (
                    str(checkpoint)
                    if args.initialization == "fine_tune"
                    else None
                ),
                "dataset": str(dataset),
                "train_filter": "train",
                "validation_filter": "valid",
                "epochs": args.epochs,
                "steps_per_epoch": args.steps_per_epoch,
                "batch_size": args.batch_size,
                "learning_rate": policy_optim["learning_rate"]["initial"],
                "lowdim_only": args.lowdim_only,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
