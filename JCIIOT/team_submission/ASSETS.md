# External asset inventory

The official repository represents several reference files as Git LFS
pointers. They are not modified or redistributed by this submission, and none
of them is required by the final runtime path: the submitted unified checkpoint
is tracked directly in Git, the MuJoCo scene assets used by the five tasks are
already present in the repository, and the official sample HDF5 is only a
training-format example.

If the evaluator wants the optional official reference files, use the organizer
links below (or `git lfs pull` after the organizer restores its LFS quota):

| Optional official asset | Organizer URL / destination |
|---|---|
| Reference epoch-150 BC | [download](https://github.com/JCIIOT2026/JCIIOT2026/raw/refs/heads/master/JCIIOT/robosuite/robosuite/model_epoch_150.pth) → `robosuite/robosuite/model_epoch_150.pth` |
| Sample HDF5 | [download](https://github.com/JCIIOT2026/JCIIOT2026/raw/refs/heads/master/JCIIOT/robosuite/dataset/table_setup_from_dishwasher_sample.hdf5) → `robosuite/dataset/` |
| Five USD archives | Organizer repository `competition description/USD/` |
| Four lowered-table mesh archives | Organizer repository under `robosuite/.../meshes/parts/` |

The final contestant checkpoint is:

```text
team_submission/models/jciiot_unified_l1_l5_bc_v4_epoch10_deploy.pth
bytes: 38,173,419
sha256: dd41174cdd1ed40d70f309024283326f0732de1aaeb0e3275b1573c13c824c5f
```

It is below GitHub's 100 MB per-file limit and is deliberately excluded from
Git LFS, so a normal clone of the submission repository receives it directly.
