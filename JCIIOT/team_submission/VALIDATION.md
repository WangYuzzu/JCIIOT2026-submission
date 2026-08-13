# Submission validation record

Validated on 2026-08-13 against official functional base commit `fa0eaef`.
Official `master` at audit time was `01032e8`; the intervening commits only
changed the root leaderboard README.

| Check | Result |
|---|---|
| Canonical trajectory/offline score audit | 100/100 PASS |
| Checkpoint SHA-256 | PASS |
| L1–L5 trajectory schema (base, 27 joints, objects) | PASS |
| Collision-marked frames | 0 / 2,349 |
| Seven strict BC grasp/lift branches | 7 / 7 PASS |
| Failed-grasp attachment recovery | absent |
| `test_skill_pipeline.py` | PASS |
| Contestant Python compilation | PASS |
| `run_all_levels.py --dry-run` | PASS |
| Streamlit `/_stcore/health` on port 8512 | `ok` |
| Protected official-file diff against `fa0eaef` | empty |
| Staged secret-pattern scan | no hits |
| Fresh public clone with LFS smudge disabled | 100/100 PASS |
| Current-tree Git LFS dependencies | none |
| Canonical result portability | repository-relative paths |
| GitHub Actions offline verification | configured |

Primary offline command:

```bash
cd JCIIOT
python team_submission/verify_submission.py
```

Expected final line:

```text
TOTAL: 100/100 PASS
```

The machine-readable record is [`evidence/verification.json`](evidence/verification.json).
