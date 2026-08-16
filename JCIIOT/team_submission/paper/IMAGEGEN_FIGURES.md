# Image-generated paper figures

Figures 1 and 2 use raster infographics generated with the built-in OpenAI
image-generation tool and then embedded without textual post-editing. The
original LaTeX captions remain authoritative. The generated images contain no
experimental measurements; they only visualize architecture already described
in the paper.

## Figure 1: system overview

Final asset:
`figures/generated/system_overview_imagegen_v2.png` (1798 x 875 RGB PNG).

The first draft was inspected and rejected because it incorrectly routed the
task request through offline knowledge generation. The accepted edit used this
prompt:

```text
Use case: precise-object-edit
Asset type: corrected publication-ready academic workflow infographic.
Input image: the supplied Figure 1 is the edit target.
Primary request: Preserve the exact visual style, white background, wide
landscape ratio, palette, typography, icons, short labels, three horizontal
bands, and overall polish, but correct the data-flow topology and reposition
cards as needed so every connector is semantically accurate and collision-free.

Required corrected layout and flow:
1. In the top band "OFFLINE KNOWLEDGE", show exactly three source cards:
   "DOCX SOP", "Task Config", and "Map + Grid". These three flow into
   "Grounded Knowledge". Remove "Task Request" from the offline band.
2. In the middle band "SYMBOLIC PLANNING", add a compact "Task Request" card
   to the left of "GLM-5.2 Planner". Draw a direct arrow from Task Request to
   GLM-5.2 Planner. Draw a separate arrow from Grounded Knowledge to GLM-5.2
   Planner. GLM-5.2 Planner flows down to Runtime Binding.
3. In the bottom band "AUDITABLE EXECUTION", place "Transformer BC" on the
   left and "A* Navigation" on the right. Runtime Binding splits cleanly into
   those two branches. Transformer BC flows to "Physical Gate"; Physical Gate
   and A* Navigation both flow into centered "Execution"; Execution flows to
   "Evidence".
4. Add a thin dashed teal data arrow from "Map + Grid" to "A* Navigation",
   routed along the far right margin so it crosses no card, text, icon, or
   other connector.

Text (verbatim, each exactly once): "OFFLINE KNOWLEDGE"; "DOCX SOP";
"Task Config"; "Map + Grid"; "Grounded Knowledge"; "SYMBOLIC PLANNING";
"Task Request"; "GLM-5.2 Planner"; "Runtime Binding";
"AUDITABLE EXECUTION"; "Transformer BC"; "Physical Gate";
"A* Navigation"; "Execution"; "Evidence".

Constraints: all arrowheads terminate at card edges; no connector passes
through a card, icon, label, or another connector; no duplicated labels; keep
every word correctly spelled; generous whitespace; no watermark, title, or
caption inside the image.
```

## Figure 2: unified BC architecture

Final asset:
`figures/generated/unified_bc_imagegen_v1.png` (1608 x 978 RGB PNG).

```text
Use case: scientific-educational
Asset type: polished single-column academic paper architecture figure,
landscape, legible at 8.5 cm printed width.
Primary request: Create a clean flat-vector neural policy architecture diagram
for a unified task-conditioned behavior-cloning controller used by a bimanual
mobile robot.

Composition: 1.65:1 landscape ratio and a top-to-bottom hierarchy. The top row
has four compact icon cards labeled "EEF State", "Grippers", "Timestep", and
"Task ID". All arrows converge into "Shared Embedding", followed by
"Temporal Transformer", a highlighted "7 Task Heads" card with seven colored
head indicators, and "20-D Action" with two robot-arm icons. A separate
conditioning arrow connects Task ID to 7 Task Heads. Add the badge
"4 layers · d=256" beside the Transformer.

Style: modern academic flat-vector infographic; white background; rounded
cards; thin navy outlines; restrained pale blue, teal, amber, and coral fills;
consistent line icons; high-contrast Source Sans/Inter-like typography.

Text (verbatim): "EEF State"; "Grippers"; "Timestep"; "Task ID";
"Shared Embedding"; "Temporal Transformer"; "4 layers · d=256";
"7 Task Heads"; "20-D Action".

Constraints: every label exactly once; no body copy; large text; arrows end at
card edges and do not cross text, icons, or other connectors; no watermark,
logo, gradient, 3D, or photorealism.
```

Both final images were inspected at native resolution and in the compiled
four-page PDF. All visible labels match the architecture described by
`main.tex`.
