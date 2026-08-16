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
L4 25, L5 30), 2,349 saved frames and zero collision-marked frames. The
canonical evidence is under [`evidence/`](evidence/) and the five rendered
demonstrations are under [`demos/`](demos/).

Run the offline verifier after cloning:

```bash
cd JCIIOT
python team_submission/verify_submission.py
```

The verifier checks the model SHA-256, five trajectory schemas, successful
grasp events, source departure, final target distance, collision flags, and
the public `app.py` score conditions. See `TECHNICAL_REPORT.md` or the concise
English paper `TECHNICAL_REPORT_EN.pdf` for per-level results and limitations;
the exact two-stage LLM input construction is documented in
[`PROMPT_DESIGN.md`](PROMPT_DESIGN.md).

## Repository layout

```text
team_submission/
├── README.md
├── VALIDATION.md                    # commands and clean-clone check record
├── TECHNICAL_REPORT.md              # comprehensive method and analysis
├── TECHNICAL_REPORT.pdf             # identical report for convenient review
├── TECHNICAL_REPORT_EN.pdf          # four-page English conference-style report
├── PROMPT_DESIGN.md                 # exact LLM/VLM prompts and runtime context
├── paper/                            # editable LaTeX, BibTeX, and vector figures
├── verify_submission.py             # no MuJoCo/API required
├── run_all_levels.py                # full official execution runner
├── evidence/                        # canonical result/trajectory/score files
├── demos/                           # five GIF demonstrations
├── TRAINING.md                      # data regeneration and training provenance
├── knowledge/                       # generated SOP Markdown + provenance
└── models/
    └── jciiot_unified_task_heads_v16_deploy.pth
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

Large optional official reference files are deliberately absent from the
submission tree. They are not used by the final runtime path; organizer links
and exact roles are listed in `ASSETS.md`. The submitted checkpoint itself is
a normal Git blob, and a normal clone requires no Git LFS download.

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

Prompt provenance matters: the online planner template under
`src/robot_agent/core/` is inherited unchanged from the official baseline.
Our contribution is the offline DOCX/VLM/text synthesis workflow, deterministic
critical-fact header, provenance manifest, live object binding, and narrow
errata guards. The precise assembly order, context budgets, API payload,
sanitized examples, and current limitations are recorded in
[`PROMPT_DESIGN.md`](PROMPT_DESIGN.md).

## Retrain the unified BC checkpoint

The final checkpoint is a low-dimensional Transformer BC trained with
robomimic. It has one shared 3.17 M-parameter Transformer and seven small
task-conditioned linear action heads (35,980 parameters total), all stored in
one 12.9 MB checkpoint. The shared model was trained on 336 successful full
demonstrations plus 144 contact-closing correction windows. L2, corrected L3,
and L5-center heads were then calibrated independently while the shared trunk
and other heads were frozen.

The 9 August L3 correction is fully included in the training data: 48/48
successful expert demonstrations use `blue_tote_b01_near_right` at the
collision-free deployment pose. The selected epoch passes genuine bilateral
finger contact and physical lift verification. No failed-grasp attachment
recovery exists in the final runtime. All seven branches pass the same strict
contact-and-lift evaluator, and all five final task trajectories have zero
collision-marked frames.

Exact data provenance, deterministic collection commands, the multi-stage
training rationale, and the boundary between deployment reproducibility and
from-scratch retraining are documented in [`TRAINING.md`](TRAINING.md).

## Integrity

```text
checkpoint bytes: 12,928,025
SHA-256: f8c7feb8047ad62f4e1e01f0e67886a0aa41f87781d486ae90e23164c37a7a5d
```

The submission checkpoint is stored as a normal Git blob (below GitHub's
100 MB limit) so evaluation does not depend on a separate LFS quota.
