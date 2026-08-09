# L3 SOP Knowledge — Case 5 (August 2026 correction)

## Task

Move one blue material-transfer tote from **Pick Station 1** to **Place Station 2**.

## Authoritative simulator mapping

- Scene: `FactorySorting5_3FO3ERTPXEUT`
- Pick Station 1: `aux_input_1`, center `(0.144, 8.473)`, approach `(0.110, 7.550)`
- Place Station 2: `output_5`, center `(4.872, -7.261)`, approach `(4.020, -7.261)`
- Allowed objects, in preferred order: `blue_tote_b01_far_right`, `blue_tote_b01_near_right`
- Required quantity: one

The August 2026 official task correction supersedes the former mapping to
`input_6` and the former orange tote. Do not use those obsolete values.

## Required plan

1. `move(target="aux_input_1", object_name=<one allowed blue tote>)`
2. `pick_up(target="aux_input_1", object_name=<the same tote>)`
3. `move(target="output_5", object_name=<the same tote>)`
4. `place_down(target="output_5", object_name=<the same tote>)`

Keep the exact object name unchanged across all four steps. Follow the
collision-free map route and confirm grasp, departure from source, and stable
placement on the destination table.
