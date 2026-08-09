# AI-Generated SOP — Case 3

## Planner-Critical Facts

- `L2`; quantity **1**; exact object: `green_tote_b01_upper`
- Required BC cycle: `move(input_6) → pick_up(target=input_6, object_name=green_tote_b01_upper) → move(output_4) → place_down(output_4)`
- `pick_up` uses the supplied BC policy; transport is valid only after BC grasp and lift both succeed.

> Generated automatically from `JCIIOT 2026 case 3 SOP.docx`; this is contestant output, not a copied reference SOP.
> Source SHA256: `6fe5f27bd234065eb5686b60919203431920338e83f8a3f0ef66d0e57cdf1178` · Text model: `glm-5.2` · Vision model: `glm-5v-turbo`

## Execution Facts

- Level: `L2`; environment: `FactorySorting3_3FO3ERRPH7X9`
- Task quantity: **1**
- Internal source: `input_6`; internal target: `output_4`
- Configured target object: `green_tote_b01_upper`
- Runtime object port: `input_6`
- Distinct eligible objects at that port: `green_tote_b01_upper`
- Grasp implementation: supplied **BC policy**, followed by lift verification
- Transport attachment rule: enable only after both BC grasp and lift verification succeed

# JCIIOT 2026 Case 3 SOP: Electronics Factory Material Handling

## 1. DOCX Evidence: Task Material Information
- **Material Name:** Green-rimmed storage bin
- **Starting Location:** Pick Station 1
- **Target Location:** Place Station 3
- **Quantity to Transport:** 1
- **Document Number:** SOP-ELC-001-2026-IND (Version 1.0, Release Date: April 23, 2026)

## 2. Verified Simulator/API Metadata
- **Environment:** FactorySorting3_3FO3ERRPH7X9
- **Semantic Source:** input_6
- **Semantic Target:** output_4
- **Configured Target Object:** green_tote_b01_upper
- **Runtime Object Port:** input_6
- **Source BC Pose:** pos [6.0, 4.8, 0.0], yaw -3.139453
- **Runtime Object BC Pose:** pos [6.0, 4.8, 0.0], yaw -3.139453
- **Objects at Runtime Port:** green_tote_b01_upper

## 3. Executable Task Planner Sequence
*Note: `runtime_object_port` matches `semantic_source`. The sequence executes the complete four-action cycle once for the single distinct object.*

1. `move(target=input_6)` — Navigate to Pick Station 1 at the verified BC pose.
2. `pick_up(target=green_tote_b01_upper)` — Execute grasp using the supplied BC policy.
3. `move(target=output_4)` — Transport material to Place Station 3.
4. `place_down(target=output_4)` — Lower and release the material at the placement point.

## 4. BC Policy and Transport Constraints
- `pick_up` is performed exclusively by the supplied BC policy.
- Transport attachment is valid **only** after BC grasp and lift verification both succeed.
- If grasp or lift fails, abort transport and trigger anomaly handling.

## 5. Operational Constraints
- **Navigation:** Maintain safe driving speed; follow planned path; avoid restricted areas.
- **Grasp:** Apply appropriate clamping force to avoid material deformation; confirm successful picking signal.
- **Placement:** Lower at steady speed; ensure full contact with placement surface; material must not be tilted or exceed boundaries.
- **Collision:** Zero tolerance for collisions; immediate stop required upon impact.
- **Drop:** If material drops, suspend task; assess if safe re-grasping is possible; otherwise await manual handling.
- **Anomaly:** On any SOP discrepancy, collision, or obstruction, stop operations immediately and report to the site supervisor. Do not unilaterally change procedures.

## 6. Validation Checklist
- [ ] Task order confirmed: 1x Green-rimmed storage bin from Pick Station 1 to Place Station 3.
- [ ] Runtime port and semantic source verified as `input_6`.
- [ ] Robot navigated to BC pose [6.0, 4.8, 0.0] with yaw -3.139453.
- [ ] BC policy grasp executed on `green_tote_b01_upper`.
- [ ] Grasp and lift verification both succeeded before transport.
- [ ] Material transported to `output_4` with zero collisions.
- [ ] Material placed stably at Place Station 3 without tilt or boundary violation.
- [ ] No repeated transport requests for the already transported object.
- [ ] Anomaly report filled if any collision, drop, or obstruction occurred.
