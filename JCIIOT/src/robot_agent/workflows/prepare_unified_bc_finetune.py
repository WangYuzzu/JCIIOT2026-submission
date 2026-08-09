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
    parser.add_argument("--checkpoint", type=Path, required=True)
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
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260737)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config = json.loads(args.base_config.read_text(encoding="utf-8"))
    config["experiment"]["name"] = args.experiment_name
    config["experiment"]["ckpt_path"] = str(checkpoint)
    config["experiment"]["epoch_every_n_steps"] = args.steps_per_epoch
    config["experiment"]["validation_epoch_every_n_steps"] = (
        args.validation_steps
    )
    config["experiment"]["save"]["every_n_epochs"] = args.save_every
    config["train"]["num_epochs"] = args.epochs
    config["train"]["seed"] = args.seed

    # The five already-stable targets remain present on every epoch to prevent
    # forgetting. The two L5 targets that lifted but lost one gripper receive
    # three times the sampling probability during the stability fine-tune.
    for dataset in config["train"]["data"]:
        dataset["weight"] = (
            args.focus_weight
            if dataset["key"] in {"l5_center", "l5_front"}
            else 1.0
        )

    learning_rate = config["algo"]["optim_params"]["policy"]["learning_rate"]
    learning_rate["initial"] = args.learning_rate
    learning_rate["scheduler_type"] = "linear"
    learning_rate["epoch_schedule"] = [args.epochs]
    learning_rate["decay_factor"] = 0.1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
