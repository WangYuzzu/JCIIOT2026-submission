# External asset inventory

The official repository represents several reference files as Git LFS
objects. They are deliberately not redistributed in the submission tree, and
none is required by the final runtime path: the submitted unified checkpoint
is tracked directly in Git, the extracted MuJoCo assets used by the five tasks
remain in the repository, and the official sample HDF5 is only a format
example. This avoids an unnecessary roughly 1.1 GB LFS download for reviewers.

If the evaluator wants the optional official reference files, use the organizer
links below:

| Optional official asset | Organizer URL / destination |
|---|---|
| Reference epoch-150 BC | [download](https://github.com/JCIIOT2026/JCIIOT2026/raw/refs/heads/master/JCIIOT/robosuite/robosuite/model_epoch_150.pth) → `robosuite/robosuite/model_epoch_150.pth` |
| Sample HDF5 | [download](https://github.com/JCIIOT2026/JCIIOT2026/raw/refs/heads/master/JCIIOT/robosuite/dataset/table_setup_from_dishwasher_sample.hdf5) → `robosuite/dataset/` |
| Five USD archives | Organizer repository `competition description/USD/` |
| Four lowered-table mesh archives | Organizer repository under `robosuite/.../meshes/parts/` |

The final contestant checkpoint is:

```text
team_submission/models/jciiot_unified_task_heads_v16_deploy.pth
bytes: 12,928,025
sha256: f8c7feb8047ad62f4e1e01f0e67886a0aa41f87781d486ae90e23164c37a7a5d
```

It is below GitHub's 100 MB per-file limit and is deliberately excluded from
Git LFS, so a normal clone of the submission repository receives it directly.
