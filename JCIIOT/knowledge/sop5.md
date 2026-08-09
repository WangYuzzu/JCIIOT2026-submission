# L5 SOP Knowledge — Case 9 (August 2026 correction)

## Task

Move all three white-rimmed storage totes from **Pick Station 6** to
**Place Station 1**.

## Authoritative simulator mapping

- Scene: `FactorySorting9_3FO3ERT2C5FP`
- Pick Station 6: `input_1`, center `(-14.544, 5.010)`
- Place Station 1: `aux_output_1`, center `(0.144, 8.473)`, approach `(0.110, 7.550)`
- Objects: `white_tote_b01_left_back`, `white_tote_b01_left_center`,
  `white_tote_b01_left_front`
- Required quantity: three distinct objects

The August 2026 official task correction supersedes the former destination
`output_6`. Do not use that obsolete destination.

## Required plan

For each distinct object, execute one complete cycle before starting the next:

1. `move(target="input_1", object_name=<current tote>)`
2. `pick_up(target="input_1", object_name=<current tote>)`
3. `move(target="aux_output_1", object_name=<current tote>)`
4. `place_down(target="aux_output_1", object_name=<current tote>)`

Preferred order: back, center, front. Place the three totes in separate slots
on the auxiliary output table. Never reuse a previously transported object.
