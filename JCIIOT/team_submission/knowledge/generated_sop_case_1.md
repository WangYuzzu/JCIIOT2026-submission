# AI-Generated SOP — Case 1

## Planner-Critical Facts

- `L1`; quantity **1**; exact object: `line_5_container_h01_near`
- Required BC cycle: `move(input_5) → pick_up(target=input_5, object_name=line_5_container_h01_near) → move(output_4) → place_down(output_4)`
- `pick_up` uses the supplied BC policy; transport is valid only after BC grasp and lift both succeed.

> Generated automatically from `JCIIOT 2026 case 1 SOP.docx`; this is contestant output, not a copied reference SOP.
> Source SHA256: `32a446a8395b03b46c9581d3e4978bd84bd90f31096e675da77366fd1cdc9c1c` · Text model: `glm-5.2` · Vision model: `glm-5v-turbo`

## Execution Facts

- Level: `L1`; environment: `FactorySorting1_3FO3ERFHISEM`
- Task quantity: **1**
- Internal source: `input_5`; internal target: `output_4`
- Configured target object: `line_5_container_h01_near`
- Runtime object port: `input_5`
- Distinct eligible objects at that port: `line_5_container_h01_near`
- Grasp implementation: supplied **BC policy**, followed by lift verification
- Transport attachment rule: enable only after both BC grasp and lift verification succeed

# JCIIOT 2026 Case 1 SOP — Contestant Knowledge Base

## 1. DOCX Evidence (SOP-ELC-001-2026-GEN)
- **Source Document:** JCIIOT 2026 case 1 SOP.docx
- **SHA256:** 32a446a8395b03b46c9581d3e4978bd84bd90f31096e675da77366fd1cdc9c1c
- **Task Prompt:** Transport a blue, hollow plastic box from Pick Station 2 to Place Station 3.
- **Material:** Blue, hollow plastic box (turnover box).
- **Visible Pick Station:** Pick Station 2 (starting point).
- **Visible Place Station:** Place Station 3 (end point).
- **Quantity:** 1 distinct object.
- **Key SOP Steps:**
  1. Navigate to Pick Station; confirm area is clear.
  2. Identify target material; plan grasp path; grasp smoothly and securely.
  3. Navigate to Place Station via shortest safe path; avoid obstacles; keep material stable.
  4. Place material precisely in designated area; release only after secure placement.
  5. Repeat if multiple materials required.
  6. Return to standby or prepare for next task; record completion status.
- **Anomaly Handling:** Stop on collision; replan or report. Re-grasp dropped material if safe; otherwise report. Stop if SOP differs from reality. Safety first.

## 2. Verified Simulator/API Metadata
- **Level:** L1
- **Environment:** FactorySorting1_3FO3ERFHISEM
- **Semantic Source:** input_5
- **Semantic Target:** output_4
- **Configured Target Object:** line_5_container_h01_near
- **Runtime Object Port:** input_5
- **Source BC Pose:** pos [8.0, 4.6, 0.0], yaw -3.139453
- **Runtime Object BC Pose:** pos [8.0, 4.6, 0.0], yaw -3.139453
- **Same-Kind Objects at Runtime Port:** line_5_container_h01_near
- **Quantity at Runtime Port:** 1
- **BC Policy:** pick_up is performed by the supplied BC policy. Transport attachment is valid only after BC grasp and lift verification both succeed.

## 3. Executable Sequence
Object: line_5_container_h01_near

1. `move(input_5)`
2. `pick_up(target=input_5, object_name=line_5_container_h01_near)`
3. `move(output_4)`
4. `place_down(output_4)`

## 4. Constraints
- **Navigation:** Use shortest safe path; avoid hazardous areas; dynamically avoid obstacles.
- **Grasp:** Plan grasp path to avoid touching other materials or equipment; grasp smoothly and securely.
- **Placement:** Confirm placement area is clear; place within designated range; avoid tipping or overshoot.
- **Collision:** Stop immediately on collision; assess; replan or report.
- **Drop:** If material drops, attempt re-grasp; if unsafe, report anomaly.
- **Anomaly:** Stop if SOP and environment differ; await instructions.
- **Transport Validity:** Attachment valid only after BC grasp and lift verification both succeed.

## 5. Validation Checklist
- [ ] Pick Station 2 identified as input_5.
- [ ] Place Station 3 identified as output_4.
- [ ] Object line_5_container_h01_near present at input_5.
- [ ] Source port is resolved from the current simulator map.
- [ ] No duplicate transport of same object.
- [ ] pick_up executed by BC policy.
- [ ] Grasp and lift verification succeeded before transport.
- [ ] Placement within designated area at output_4.
- [ ] Collision, drop, and anomaly handling ready.
- [ ] Task completion recorded.
