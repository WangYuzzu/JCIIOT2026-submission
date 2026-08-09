"""Prepare unified BC fine-tuning with genuine L3 close-event oversampling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corrective-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int, default=500)
    parser.add_argument("--validation-steps", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--l3-weight", type=float, default=2.0)
    parser.add_argument("--corrective-weight", type=float, default=6.0)
    parser.add_argument("--save-every", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    corrective = args.corrective_dataset.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not corrective.is_file():
        raise FileNotFoundError(corrective)

    config = json.loads(args.base_config.read_text(encoding="utf-8"))
    config["experiment"]["name"] = "jciiot_unified_bc_v9b_l3_close_balanced"
    config["experiment"]["ckpt_path"] = str(checkpoint)
    config["experiment"]["epoch_every_n_steps"] = args.steps_per_epoch
    config["experiment"]["validation_epoch_every_n_steps"] = args.validation_steps
    config["experiment"]["save"]["every_n_epochs"] = args.save_every
    config["experiment"]["save"]["epochs"] = []
    config["train"]["num_epochs"] = args.epochs
    config["train"]["seed"] = args.seed
    config["train"]["output_dir"] = str(args.model_output_dir.resolve())

    for dataset in config["train"]["data"]:
        dataset["weight"] = args.l3_weight if dataset["key"] == "l3" else 1.0
    config["train"]["data"].append(
        {
            "path": str(corrective),
            "eval": False,
            "key": "l3_close_corrective",
            "weight": args.corrective_weight,
        }
    )

    learning_rate = config["algo"]["optim_params"]["policy"]["learning_rate"]
    learning_rate["initial"] = args.learning_rate
    learning_rate["scheduler_type"] = "linear"
    learning_rate["epoch_schedule"] = [args.epochs]
    learning_rate["decay_factor"] = 0.1
    config["algo"]["optim_params"]["policy"].pop("num_train_batches", None)
    config["algo"]["optim_params"]["policy"].pop("num_epochs", None)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.model_output_dir.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
