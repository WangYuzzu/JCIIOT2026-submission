# JCIIOT 2026 Task Matrix and Common SOP

Use this table as the authoritative natural-language-to-simulator mapping. It
is synchronized with `knowledge/task_config.json` and the regenerated semantic
maps from the official August 2026 release.

| Level | Scene | Pick station | Source | Allowed object(s) | Place station | Target |
|---|---|---|---|---|---|---|
| L1 | FactorySorting1 | Pick Station 2 | `input_5` | `line_5_container_h01_near` or `line_5_container_h01_far` | Place Station 3 | `output_4` |
| L2 | FactorySorting3 | Pick Station 1 | `input_6` | `green_tote_b01_upper` or `green_tote_b01_lower` | Place Station 3 | `output_4` |
| L3 | FactorySorting5 | Pick Station 1 | `aux_input_1` | `blue_tote_b01_far_right` or `blue_tote_b01_near_right` | Place Station 2 | `output_5` |
| L4 | FactorySorting7 | Pick Station 5 | `input_2` | `blue_container_h01_back_upper` or `blue_container_h01_back_lower` | Place Station 2 | `output_5` |
| L5 | FactorySorting9 | Pick Station 6 | `input_1` | three distinct `white_tote_b01_left_*` objects | Place Station 1 | `aux_output_1` |

## Common execution contract

For one object, emit exactly:

1. `move(target=<source>, object_name=<exact object>)`
2. `pick_up(target=<source>, object_name=<exact object>)`
3. `move(target=<target>, object_name=<same object>)`
4. `place_down(target=<target>, object_name=<same object>)`

For L5, repeat the four-step cycle three times using back, center, and front
objects exactly once each. Use exact semantic station names and exact object
names; do not substitute legacy aliases.

## Safety and completion

- Plan on the occupancy grid and maintain obstacle clearance.
- Confirm grasp before transport and stable support after release.
- Do not collide, drop, duplicate, or silently change the selected object.
- August correction: L3 is not `input_6`/orange; L5 is not `output_6`.
