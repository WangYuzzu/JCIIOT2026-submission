"""Prepare a Transformer BC policy with an explicit episode timestep.

The runtime policy is still a robomimic Behavior Cloning policy.  The scripted
controller is used only to produce demonstrations.  An absolute episode-step
observation is added because the supplied two-arm demonstration contains
several phases with nearly identical end-effector poses but different intended
actions (approach, close, and hold).  A pre-existing training config can be
supplied for low-dimensional scene-specific policies.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np


APP_ROOT = Path(__file__).resolve().parents[3]
for path in (APP_ROOT, APP_ROOT / "robomimic", APP_ROOT / "robosuite"):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from robomimic.utils.file_utils import load_dict_from_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    artifacts = APP_ROOT / "team_submission" / "training_artifacts"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--template-checkpoint",
        type=Path,
        default=APP_ROOT / "robosuite" / "robosuite" / "model_epoch_150.pth",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=None,
        help=(
            "Optional JSON config to preserve. When omitted, the config embedded "
            "in --template-checkpoint is used."
        ),
    )
    parser.add_argument(
        "--source-dataset",
        type=Path,
        default=(
            artifacts
            / "datasets_l1_diverse_v1"
            / "l1_line_5_container_h01_near.hdf5"
        ),
    )
    parser.add_argument(
        "--dataset-out",
        type=Path,
        default=artifacts / "datasets_l1_diverse_v1" / "l1_timestep_rgb_bc.hdf5",
    )
    parser.add_argument(
        "--config-out",
        type=Path,
        default=artifacts / "timestep_rgb_bc_config.json",
    )
    parser.add_argument("--output-dir", type=Path, default=artifacts / "models")
    parser.add_argument("--name", default="jciiot_l1_timestep_rgb_bc_diverse_v1")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--steps-per-epoch", type=int, default=50)
    parser.add_argument("--validation-steps", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument(
        "--normalize-obs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "The bundled runtime normalizes before converting RGB HWC to CHW, "
            "so leave this disabled for deployable image checkpoints."
        ),
    )
    return parser.parse_args()


def _copy_with_timesteps(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    with h5py.File(destination, "a") as dataset:
        for demo in dataset["data"].values():
            length = int(demo["actions"].shape[0])
            obs = demo["obs"]
            if "timesteps" in obs:
                del obs["timesteps"]
            obs.create_dataset(
                "timesteps",
                data=np.arange(length, dtype=np.float32).reshape(length, 1),
            )


def main() -> int:
    args = parse_args()
    checkpoint = args.template_checkpoint.resolve()
    source = args.source_dataset.resolve()
    destination = args.dataset_out.resolve()
    if args.base_config is None and not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args.base_config is not None and not args.base_config.is_file():
        raise FileNotFoundError(args.base_config)
    if not source.is_file():
        raise FileNotFoundError(source)
    _copy_with_timesteps(source, destination)

    if args.base_config is not None:
        config = json.loads(args.base_config.read_text(encoding="utf-8"))
    else:
        checkpoint_data = load_dict_from_checkpoint(str(checkpoint))
        config = json.loads(checkpoint_data["config"])
    config["experiment"]["name"] = args.name
    config["experiment"]["ckpt_path"] = None
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
    config["experiment"]["save"]["on_best_validation"] = False
    config["experiment"]["save"]["on_best_rollout_return"] = False
    config["experiment"]["save"]["on_best_rollout_success_rate"] = False

    config["train"]["data"] = [{"path": str(destination), "eval": False}]
    config["train"]["output_dir"] = str(args.output_dir.resolve())
    config["train"]["num_epochs"] = args.epochs
    config["train"]["batch_size"] = args.batch_size
    config["train"]["num_data_workers"] = 0
    config["train"]["cuda"] = True
    config["train"]["seed"] = args.seed
    config["train"]["hdf5_filter_key"] = "train"
    config["train"]["hdf5_validation_filter_key"] = "valid"
    # Keep the same ten-frame Transformer interface used by the supplied
    # checkpoint. FrameStackWrapper also injects the runtime ``timesteps`` key.
    config["train"]["frame_stack"] = 10
    config["train"]["seq_length"] = 1
    config["train"]["hdf5_normalize_obs"] = args.normalize_obs
    if args.base_config is None:
        config["train"]["action_config"]["actions"]["normalization"] = "min_max"

    config["algo"]["actor_layer_dims"] = []
    config["algo"]["transformer"]["enabled"] = True
    config["algo"]["transformer"]["context_length"] = 10
    config["algo"]["transformer"]["embed_dim"] = 128
    config["algo"]["transformer"]["num_layers"] = 2
    config["algo"]["transformer"]["num_heads"] = 4
    config["algo"]["transformer"]["emb_dropout"] = 0.0
    config["algo"]["transformer"]["attn_dropout"] = 0.0
    config["algo"]["transformer"]["block_output_dropout"] = 0.0
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

    low_dim = config["observation"]["modalities"]["obs"]["low_dim"]
    if "timesteps" not in low_dim:
        low_dim.append("timesteps")
    # Existing modalities are intentionally preserved. The timestep removes
    # phase ambiguity in both low-dimensional and vision-based policies.

    args.config_out.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.config_out.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(destination)
    print(args.config_out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
