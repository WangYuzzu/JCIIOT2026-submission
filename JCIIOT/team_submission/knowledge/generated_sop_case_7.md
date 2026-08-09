# AI-Generated SOP — Case 7

## Planner-Critical Facts

- `L4`; quantity **1**; exact object: `blue_container_h01_back_upper`
- Required BC cycle: `move(input_2) → pick_up(target=input_2, object_name=blue_container_h01_back_upper) → move(output_5) → place_down(output_5)`
- `pick_up` uses the supplied BC policy; transport is valid only after BC grasp and lift both succeed.

> Generated automatically from `JCIIOT 2026 case 7 SOP.docx`; this is contestant output, not a copied reference SOP.
> Source SHA256: `df637fcb1e558cbcacfc895d38b435008aa9eb15e1f85744631cea12b137e568` · Text model: `glm-5.2` · Vision model: `glm-5v-turbo`

## Execution Facts

- Level: `L4`; environment: `FactorySorting7_3FO3ERFKY9RN`
- Task quantity: **1**
- Internal source: `input_2`; internal target: `output_5`
- Configured target object: `blue_container_h01_back_upper`
- Runtime object port: `input_2`
- Exact eligible object at that port: `blue_container_h01_back_upper`
- Grasp implementation: supplied **BC policy**, followed by lift verification
- Transport attachment rule: enable only after both BC grasp and lift verification succeed

# JCIIOT 2026 Case 7: Material Handling Task Knowledge-Base

## 1. Evidence Distinction

### DOCX Evidence (SOP-ELC-001-2026-GEN)
- **Source Document:** JCIIOT 2026 case 7 SOP.docx
- **Source SHA256:** `df637fcb1e558cbcacfc895d38b435008aa9eb15e1f85744631cea12b137e568`
- **Material:** A blue, hollow plastic box.
- **Pick Station:** Pick Station 5.
- **Place Station:** Place Station 2.
- **Quantity:** One target object.

### Verified Simulator/API Metadata
- **Level:** L4
- **Environment:** FactorySorting7_3FO3ERFKY9RN
- **Semantic Source:** input_2
- **Semantic Target:** output_5
- **Runtime Object Port:** input_2
- **Runtime Navigation Port:** `input_2` (resolved by the simulator map at execution time)
- **Configured Target Object:** `blue_container_h01_back_upper`
- **Only Eligible Object:** `blue_container_h01_back_upper`

## 2. Task Constraints & Policies
- **BC Policy Execution:** The `pick_up` action is performed entirely by the supplied Behavior Cloning (BC) policy.
- **Transport Validation:** Transport attachment is valid *only* after BC grasp and lift verification both succeed. The system must not initiate transport if stability is uncertain.
- **Navigation Constraints:** The route from Pick Station to Place Station must be clear. If path blockage occurs, slow down, stop, or replan.
- **Collision Constraints:** Stop all motion immediately if a collision occurs. Assess safe recovery before replanning. The gripper path must not cause unnecessary interference.
- **Placement Constraints:** Drop-off area must be verified as unoccupied. Lower the material gradually to prevent tipping, offset, or overhang.
- **Drop/Anomaly Constraints:** If the material is dropped, stop immediately. Reattempt pickup only if safe. If site conditions differ from the SOP, halt and wait for approval.
- **Repetition Rule:** For quantity > 1, repeat the complete four-action cycle once per distinct object. Never request an already transported object again.

## 3. Executable Action Sequence
The following sequence executes the complete four-action cycle exactly once.

### Transport `blue_container_h01_back_upper`
1. `move(input_2)`
   - Navigate to Pick Station 5 using the current simulator map. Verify operating area is clear.
2. `pick_up(target=input_2, object_name=blue_container_h01_back_upper)`
   - Execute BC policy to grasp the blue, hollow plastic box. Await grasp and lift verification before proceeding.
3. `move(output_5)`
   - Navigate safely to Place Station 2 (verified target: `output_5`). Monitor material stability during transport.
4. `place_down(output_5)`
   - Inspect drop-off area. Lower gradually and release. Verify final placement stability.

## 4. Validation Checklist
- [ ] **Target Verification:** Confirmed target material is a blue, hollow plastic box.
- [ ] **Station Verification:** Navigated to Pick Station 5 (`input_2`) and Place Station 2 (`output_5`).
- [ ] **BC Grasp Success:** `pick_up` executed via BC policy; grasp and lift verification succeeded.
- [ ] **Collision Free:** No collisions occurred during approach, transport, or placement.
- [ ] **Drop Prevention:** Material remained stable throughout transport; no drops detected.
- [ ] **Placement Accuracy:** Material positioned within designated zone at Place Station 2 without tipping or overhang.
- [ ] **Quantity Completion:** The single configured object `blue_container_h01_back_upper` was transferred exactly once.
- [ ] **Anomaly Handling:** Any exceptions (collisions, drops, blockages) documented; system returned to standby ready status.
