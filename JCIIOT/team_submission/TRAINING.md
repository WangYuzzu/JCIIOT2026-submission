# Unified BC training and data provenance

This document separates two reproducibility targets:

- **Evaluation reproduction is self-contained.** The deploy checkpoint,
  runtime code, five trajectories, hashes and offline verifier are committed.
- **From-scratch training regenerates data.** The 174,860-step expert HDF5
  corpus is not committed because it is generated from the included simulator
  and is substantially larger than the 12.9 MB deploy checkpoint.

The final checkpoint embeds the robomimic configuration, shape metadata,
environment metadata and action-normalization statistics. Its SHA-256 is:

```text
f8c7feb8047ad62f4e1e01f0e67886a0aa41f87781d486ae90e23164c37a7a5d
```

## Dataset

`src/robot_agent/workflows/collect_bc_demonstrations.py` runs a deterministic
MuJoCo expert controller. It varies the deployment base x/y/yaw, phase timing
and action values, then retains only episodes that pass bilateral contact,
physical lift and collision checks. The reported corpus uses seed `20260723`,
48 successful episodes for each of seven branches, and low-dimensional state:

| Branch | Episodes | Steps |
|---|---:|---:|
| L1 | 48 | 15,015 |
| L2 | 48 | 22,789 |
| corrected L3 | 48 | 23,819 |
| L4 | 48 | 24,277 |
| L5 back | 48 | 24,705 |
| L5 center | 48 | 32,596 |
| L5 front | 48 | 31,659 |
| **Total** | **336** | **174,860** |

An additional 144 contact-closing windows (16,752 steps) are derived from
successful expert episodes. They rebalance the short but critical finger-close
phase; they are not synthetic labels.

Representative regeneration command:

```bash
cd JCIIOT
PYTHONPATH=src:.:robomimic:robosuite/robosuite \
python -m robot_agent.workflows.collect_bc_demonstrations \
  --levels L1 L2 L3 L4 L5 \
  --rollouts 48 \
  --seed 20260723 \
  --pose-source deployment \
  --lowdim-only \
  --expert-lift-height 0.12 \
  --expert-lift-hold-steps 20 \
  --output-dir team_submission/training_artifacts/datasets
```

The collector writes an HDF5 file per object branch and a manifest containing
the seed, variation and success audit. L3 is the corrected blue tote at
`aux_input_1`; obsolete orange-tote data is not accepted into the final branch.

## Model and stages

The policy uses robomimic BC with a four-layer, 256-dimensional Transformer,
8 attention heads, context length 10 and 20-dimensional continuous actions.
The observation contains both EEF poses, both gripper states, a timestep and a
7-dimensional task id. Training loss is `L2 + 0.1 * L1`.

Training proceeded in two auditable stages:

1. train the shared Transformer and seven task heads from random initialization
   on the successful multi-task demonstrations;
2. freeze the shared trunk and unrelated heads, then calibrate only the L2,
   corrected-L3 and L5-center heads using their contact-closing data.

Relevant preparation and audit entry points are:

```text
src/robot_agent/workflows/prepare_corrected_l3_unified_bc.py
src/robot_agent/workflows/prepare_unified_bc_finetune.py
src/robot_agent/workflows/prepare_l3_gripper_close_corrections.py
src/robot_agent/workflows/evaluate_bc_grasp_and_lift.py
src/robot_agent/workflows/export_unified_bc_checkpoint.py
```

robomimic training is launched with its standard entry point after a preparation
script writes the JSON configuration:

```bash
PYTHONPATH=.:src python robomimic/scripts/train.py --config <generated-config.json>
```

Checkpoint selection is based on the intersection of seven strict physical
contact-and-lift evaluations, not validation loss alone. The deploy file is an
export of the selected checkpoint and contains only the weights and metadata
needed at runtime.

## Hardware and determinism

Collection and training were performed on Ubuntu/WSL2 with Python 3.11,
CUDA 12.x and an RTX 4060 Laptop GPU with 8 GB VRAM. Fixed seeds are present in
all collection/preparation/evaluation scripts. MuJoCo and CUDA execution can
still show small platform-dependent floating-point variation, so a retraining
run is expected to reproduce the method and acceptance tests rather than the
checkpoint byte-for-byte. The committed checkpoint and evidence hashes provide
byte-exact evaluation reproduction.
