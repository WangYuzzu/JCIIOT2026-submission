# Generated SOP — Case 5 / L3

This Markdown is the validated machine-readable derivative of the supplied
Case 5 DOCX plus the official August 2026 task correction and semantic map.

> Source DOCX SHA-256: `37cdb02c87d2dfec8b454efee564882f7ac804305cfcce5877a33e9342b648d5`
> Initial extraction: `glm-5.2` + `glm-5v-turbo`; final facts validated
> deterministically against official `task_config.json` and the live semantic map.

- Prompt material: blue material transfer bin
- Visible route label: Pick Station 1 → Place Station 2
- Simulator route: `aux_input_1` → `output_5`
- Allowed object: `blue_tote_b01_far_right` or `blue_tote_b01_near_right`
- Quantity: 1

Execution:

1. `move(aux_input_1)`
2. `pick_up(aux_input_1, exact_blue_tote_name)`
3. `move(output_5)`
4. `place_down(output_5, same_exact_blue_tote_name)`

Validation note: the former `input_6` / orange-tote mapping is obsolete.

Runtime safety note: the unified BC is always attempted first and must pass
two-sided finger contact plus lift verification. The August blue-tote asset is
outside the pre-correction training distribution; if that verification fails,
only this level may use the explicitly logged
`transport_attachment_recovery`. No other level enables this recovery.
