# JCIIOT 2026 Submission

This directory is the reproducible submission package for the five official
FactorySorting tasks. It contains one unified task-conditioned BC checkpoint,
the SOP-generation and BC workflows, offline score verification, final
trajectory evidence, and the technical report.

The code is based on official commit `fa0eaef` (9 August 2026 task update).
The prohibited files remain unchanged relative to that official commit:
`app.py`, `src/robot_agent/core/`, `src/robot_agent/environments/`, and
`knowledge/task_config.json`.

## Result at a glance

Final official-condition replay audit: **100/100** (L1 10, L2 15, L3 20,
L4 25, L5 30), 2,263 saved frames and zero collision-marked frames. The
canonical evidence is under [`evidence/`](evidence/) and the five rendered
demonstrations are under [`demos/`](demos/).

Run the offline verifier after cloning:

```bash
cd JCIIOT
python team_submission/verify_submission.py
```

The verifier checks the model SHA-256, five trajectory schemas, successful
grasp events, source departure, final target distance, collision flags, and
the public `app.py` score conditions. See `TECHNICAL_REPORT.md` for the
per-level result and limitations.

## Repository layout

```text
team_submission/
├── README.md
├── VALIDATION.md                    # commands and clean-clone check record
├── TECHNICAL_REPORT.md              # comprehensive method and analysis
├── TECHNICAL_REPORT.pdf             # identical report for convenient review
├── verify_submission.py             # no MuJoCo/API required
├── run_all_levels.py                # full official execution runner
├── evidence/                        # canonical result/trajectory/score files
├── demos/                           # GIF/MP4 demonstrations
├── knowledge/                       # generated SOP Markdown + provenance
└── models/
    └── jciiot_unified_l1_l5_bc_v4_epoch10_deploy.pth
```

Training/data preparation code is under
`src/robot_agent/workflows/`; runtime changes are limited to
`src/robot_agent/skills/`, `src/robot_agent/task_subprocess_runner.py`, and
`knowledge/robot_params.json` plus generated knowledge Markdown.

## Environment setup

Tested on Ubuntu/WSL2, Python 3.11, CUDA 12.x, RTX 4060 Laptop 8 GB. CPU
execution is supported but slower.

```bash
git clone https://github.com/WangYuzzu/JCIIOT2026-submission.git
cd JCIIOT2026-submission/JCIIOT

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e ./robomimic
python -m pip install -e ./robosuite
python -m pip install -e .
```

Several optional official reference files are Git LFS pointers. They are not
used by the final runtime path; organizer links and exact roles are listed in
`ASSETS.md`. The submitted checkpoint itself is a normal Git blob.

The warnings about a private robosuite macro file and optional `mink` whole-
body IK do not affect these Tiago task controllers. They can be removed by
running the official macro setup script and installing `mink==0.0.5`, but
neither is required for the reproduced evidence.

## Configure the planner safely

Copy the example and export values in the shell used to launch Streamlit.
Do not commit an API key.

```bash
cp team_submission/.env.example team_submission/.env
set -a
source team_submission/.env
set +a
```

Any OpenAI-compatible text model that reliably emits the required JSON plan
schema can be used. The reported runs used `glm-5.2` through the endpoint shown
in `.env.example`. VLM access is only needed to regenerate image descriptions
from the DOCX SOPs; it is not needed for offline verification or trajectory
replay.

## Reproduce through the official UI

```bash
streamlit run app.py
```

In the sidebar choose **OpenAI API**, enter the same base URL, key, and model,
enable the knowledge base, then execute L1 through L5. Streamlit launches the
same isolated task runner used by `run_all_levels.py` and saves JSON under
`recordings/<scene>/`.

For a sequential command-line rerun:

```bash
python team_submission/run_all_levels.py
# or a subset
python team_submission/run_all_levels.py --levels L1 L3 L5
```

L5 requires three BC rollouts and is intentionally much slower than the other
levels. Each child process closes its MuJoCo environment before the next level
starts.

## Regenerate SOP knowledge

The submitted Markdown is not a blind copy of the provided reference files.
It is produced by `generate_sop_knowledge.py` from the five DOCX files, VLM
image descriptions, `task_config.json`, and semantic-map truth, then checked
for exact source/target/object consistency.

```bash
PYTHONPATH=src:. python -m robot_agent.workflows.generate_sop_knowledge \
  --help
```

The August correction is explicit: L3 uses `aux_input_1` and a blue tote;
L5 uses `aux_output_1`. The runtime also contains a narrow correction guard so
an obsolete alias emitted by a language model cannot silently send the robot
to the old station.

## Retrain the unified BC checkpoint

The final checkpoint is a low-dimensional Transformer BC trained with
robomimic from 336 successful episodes / 182,062 time steps covering seven
pre-correction grasp branches. All seven branches share one 7-D task condition
and one 3.17 M-parameter actor. Collection, timestep conversion, balanced
multi-task preparation, recovery fine-tuning, export, and strict grasp/lift
evaluation commands are preserved under `src/robot_agent/workflows/`.

The 9 August L3 asset changed orientation after training. Its old BC branch is
still invoked first, but cannot establish the official double-finger physical
contact because the corrected blue tote's authored grasp markers are rotated
relative to Tiago's finger closing axis. Only for this corrected branch, a
clearly recorded `transport_attachment_recovery` is enabled after BC/contact/
lift failure. L1, L2, L4, and all three L5 targets require and pass genuine BC
contact plus lift verification. This limitation is disclosed in the report and
trajectory event metadata.

## Integrity

```text
checkpoint bytes: 38,173,419
SHA-256: dd41174cdd1ed40d70f309024283326f0732de1aaeb0e3275b1573c13c824c5f
```

The submission checkpoint is stored as a normal Git blob (below GitHub's
100 MB limit) so evaluation does not depend on a separate LFS quota.
