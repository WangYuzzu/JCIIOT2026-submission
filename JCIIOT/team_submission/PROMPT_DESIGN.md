# LLM Prompt Design and Reproduction Notes

This document records the exact language-model boundary used by the submitted
system. It is intentionally more detailed than the technical report so that an
evaluator can distinguish inherited organizer code from contestant-authored
knowledge engineering.

## 1. Scope and provenance

There are two separate model calls:

1. **Offline SOP generation:** contestant code converts the five official DOCX
   files into auditable Markdown. This workflow and its prompts are implemented
   in `src/robot_agent/workflows/generate_sop_knowledge.py`.
2. **Online task planning:** the official planner converts one task request and
   bounded knowledge context into a JSON skill plan. Its prompt template lives
   in `src/robot_agent/core/planner.py` and remains unchanged from the official
   functional baseline `fa0eaef`.

Our LLM contribution is therefore not a claim that we invented the online
planner template. It is the reproducible DOCX-to-knowledge pipeline, provenance
tracking, deterministic critical-fact header, live object-name binding, and the
execution-side corrections needed to make the inherited planner robust to the
August task errata.

## 2. Stage A: DOCX to generated knowledge

### 2.1 Evidence extraction

For every official SOP DOCX, the workflow extracts:

- non-empty paragraphs;
- all table rows;
- de-duplicated embedded images;
- SHA-256 of the source file;
- verified execution facts derived from the read-only official task
  configuration and the current MuJoCo `material_metadata` inventory.

The generator never reads the official reference `knowledge/sop*.md` as an
input. This prevents a prewritten Markdown answer from being silently copied.

### 2.2 VLM prompt

Each embedded image is sent independently to the VLM with the following narrow
instruction (translated literally: objectively describe visible layout,
stations, tables, lines, major objects, and legible labels; do not infer hidden
facts; return only the final sentence):

```text
用一句话客观描述图中可见的工厂布局、工位、桌子、生产线、主要物体和清晰文字标签，
不要推测看不见的信息，只输出最终答案。
```

This prompt deliberately asks for observations rather than action planning.
The reported generation used `glm-5v-turbo`. The VLM is not called during final
trajectory replay or offline scoring.

### 2.3 Text synthesis prompt

The text model receives the source name and hash, verified execution facts as
JSON, extracted DOCX text, and VLM observations. Its instruction requires an
original concise planner document and enforces these semantic requirements:

1. preserve material, visible pick/place labels, and quantity;
2. state verified internal source, target, exact object names, and BC pose facts;
3. use only `move`, `pick_up`, and `place_down`;
4. emit a complete four-action cycle for every distinct object when quantity is
   greater than one;
5. permit transport attachment only after BC grasp and lift verification;
6. include navigation, grasp, placement, collision, drop, and anomaly rules;
7. distinguish DOCX observations from simulator/API metadata;
8. avoid unsupported claims; and
9. return Markdown only.

After synthesis, deterministic code prepends a **Planner-Critical Facts**
section. The model does not get to overwrite this header. It records the level,
quantity, exact semantic source and target, exact allowed object names, the
required four-step cycle, model identity, and source hash. The output files and
their provenance are listed in
`team_submission/knowledge/generated_sop_manifest.json`.

## 3. Stage B: online planning prompt

For one selected level, the inherited planner assembles a single user message
in the following order:

| Block | Runtime source | Bound/current behavior |
|---|---|---|
| Role | fixed template | Chinese robot task planner |
| Official knowledge | `knowledge/*.md` | at most 4,000 characters total; `sop_main.md` excluded from this block; each document snippet is at most 600 characters |
| Current coordinate row | `knowledge/sop_main.md` | only the selected L1--L5 row is appended |
| Team knowledge | `team_submission/knowledge/*.md` | a separate block of at most 4,000 characters |
| Live object map | MuJoCo `material_metadata`, merged with the current official task entry | every available station-to-exact-object mapping is listed |
| Skill interface | registry + fixed template | allowed names and exact `inputs` shapes |
| Few-shot example | fixed template | one four-step move/pick/move/place JSON example |
| Hard rules | fixed template | JSON-only output, exact object name, 300 s timeout, zero retry, complete cycle |
| User task | current UI task text | appended last |

The four general files `command_examples.md`, `pick_operation.md`,
`place_operation.md`, and `constraints.md` are ordinary official knowledge
documents. They are loaded by `KnowledgeManager` for every level, subject to
the same 4,000-character budget; they are not injected by
`skills/read_document.py`, and they are not exclusive to L1.

The live object mapping is built only after the MuJoCo scene is reset and its
`material_metadata` exists. The current task's official object list then acts
as an override/fallback for that station. For corrected L3, an explicit
preference orders `blue_tote_b01_near_right` before the other score-valid tote.
This mapping is what lets the prompt request an exact simulator object name
rather than a visual description such as “blue box.”

### 3.1 Sanitized L3 example

The effective high-salience portion of the L3 prompt is equivalent to:

```text
Current task row: L3 ... Pick=aux_input_1 ... Place=output_5
Current Scene Object Mapping:
- aux_input_1 -> [blue_tote_b01_near_right, blue_tote_b01_far_right]

Available skills: move, pick_up, place_down, ...
pick_up inputs = {"target": "<station>", "object_name": "<exact name>"}
Rules: exact mapped object name; complete move -> pick_up -> move -> place_down.
User task: move the corrected blue tote to the requested place station.
```

A conforming plan therefore uses `aux_input_1`,
`blue_tote_b01_near_right`, and `output_5`. The example above is explanatory;
the actual complete prompt is constructed by `_build_plan_prompt()`.

### 3.2 Sanitized L5 multi-object example

The L5 context states quantity three and exposes three distinct white-rim tote
identities at `input_1`. The generated SOP requires three complete cycles:

```text
move(input_1) -> pick_up(input_1, object_A) -> move(aux_output_1) -> place_down(aux_output_1)
move(input_1) -> pick_up(input_1, object_B) -> move(aux_output_1) -> place_down(aux_output_1)
move(input_1) -> pick_up(input_1, object_C) -> move(aux_output_1) -> place_down(aux_output_1)
```

Execution additionally selects the remaining back/center/front tote in a fixed
order so that a repeated or missing name from the LLM cannot collapse a
three-object task into repeated manipulation of one object.

## 4. API request and structured output

Reported online plans used Zhipu `glm-5.2` through the OpenAI-compatible chat
completions interface:

```json
{
  "messages": [{"role": "user", "content": "<assembled prompt>"}],
  "temperature": 0.1,
  "max_tokens": 4096,
  "stream": false,
  "response_format": {"type": "json_object"},
  "thinking": {"type": "disabled"}
}
```

`thinking` is disabled because only the auditable plan JSON is needed; it does
not mean GLM-5.2 lacks reasoning ability. If a provider rejects JSON mode, the
client retries once without `response_format`. The parser then attempts direct
JSON decoding, repairs common fences/quotes/trailing commas or extracts an
embedded object, and can issue a short “valid JSON only” retry. The default
planner retry budget is two.

API keys are read from environment variables or entered in the UI and are not
stored in this repository. “OpenAI API” in the UI denotes protocol
compatibility, not an OpenAI-hosted model.

## 5. Validation and execution boundary

It is important not to overstate what the planner validator does:

- `normalize_planner_output()` normalizes the JSON envelope and validates that
  every `skill_name` exists in the registry.
- Missing `object_name` values can be filled from the live mapping.
- Narrow August-errata aliases are corrected at execution time (L3 legacy
  sources map to `aux_input_1`; L5 legacy destination maps to
  `aux_output_1`).
- L5 execution maintains a fixed list of remaining distinct objects.
- Navigation, grasp contact/lift checks, collision checks, and score-compatible
  event serialization are deterministic execution responsibilities.

The current schema validator does **not** independently prove that every
generated source, target, object, and multiplicity field is legal before
execution. Correctness comes from redundant prompt evidence plus runtime
binding and physical checks, not from a universal formal plan verifier.

## 6. Known limitations

- Knowledge retrieval is bounded but not task-ranked: the official and team
  blocks can contain snippets from other levels before truncation. The current
  SOP coordinate row and live object map are appended separately to keep the
  active task salient.
- The full `task_config.json` and complete semantic-map coordinate dump are not
  placed in the online prompt. They support offline knowledge generation and
  deterministic runtime behavior.
- When knowledge is enabled, the current implementation does not insert the
  `SceneContext` prose summary into the prompt; base navigation still consumes
  the semantic map and occupancy grid directly.
- Planning latency and availability depend on the external API. Saved evidence,
  GIF generation, and offline score verification do not.

Task-scoped retrieval and full schema-level semantic validation are sensible
future improvements, but they are not claimed as part of the reported 100/100
executions.

## 7. Reproduction

Inspect the exact templates:

```bash
sed -n '1,140p' src/robot_agent/workflows/generate_sop_knowledge.py
sed -n '70,250p' src/robot_agent/core/planner.py
sed -n '40,110p' src/robot_agent/core/openai_client.py
```

Regenerate the SOP knowledge after configuring LLM/VLM environment variables:

```bash
PYTHONPATH=src:. python -m robot_agent.workflows.generate_sop_knowledge --help
```

Run the official UI or sequential runner:

```bash
streamlit run app.py
python team_submission/run_all_levels.py
```

The exact generated Markdown and manifest used by the submitted evidence are
already included, so offline verification does not require either model API.
