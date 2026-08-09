# Trajectory demonstrations

These GIFs are rendered directly from the canonical trajectory JSON files in
`../evidence/`. Each visualization uniformly samples 180 saved simulator
states, restores the robot base pose, all 27 recorded joints, and movable
object poses, then renders the official scene with the bird-view camera.

| Level | Demonstration | Frames in canonical trajectory |
|---|---|---:|
| L1 | [bird view](l1_birdview.gif) | 304 |
| L2 | [bird view](l2_birdview.gif) | 344 |
| L3 | [bird view](l3_birdview.gif) | 360 |
| L4 | [bird view](l4_birdview.gif) | 367 |
| L5 | [bird view](l5_birdview.gif) | 974 |

Regenerate with:

```bash
cd JCIIOT
PYTHONPATH=src:. python team_submission/generate_demos.py
```
