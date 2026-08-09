"""Create leakage-safe corrective windows around the L3 gripper close event.

The unified data are strongly imbalanced in time: navigation and arm approach
occupy roughly 430 frames while bilateral gripper closure occupies about 70.
This workflow does not synthesize or edit actions. It crops several overlapping
windows from each successful expert demonstration so the original close and
hold transitions are sampled more often during BC fine-tuning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pre-close-frames",
        type=int,
        nargs="+",
        default=[80, 40, 16],
        help="expert context retained before the first bilateral close action",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _copy_attrs(source, destination) -> None:
    for key, value in source.attrs.items():
        destination.attrs[key] = value


def _decode_mask(values) -> set[str]:
    return {
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    }


def _slice_group(source: h5py.Group, destination: h5py.Group, start: int) -> None:
    _copy_attrs(source, destination)
    for key, item in source.items():
        if isinstance(item, h5py.Dataset):
            destination.create_dataset(key, data=item[start:])
        elif isinstance(item, h5py.Group):
            child = destination.create_group(key)
            _slice_group(item, child, start)
        else:  # pragma: no cover - h5py groups contain datasets or groups here
            raise TypeError(f"unsupported HDF5 item: {item}")


def main() -> int:
    args = parse_args()
    source_path = args.input.resolve()
    output_path = args.output.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if output_path.exists() and not args.force:
        raise FileExistsError(output_path)
    if any(frames < 10 for frames in args.pre_close_frames):
        raise ValueError("all corrective windows need at least 10 pre-close frames")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    audit: list[dict] = []

    with h5py.File(source_path, "r") as source, h5py.File(temporary, "w") as output:
        _copy_attrs(source, output)
        data_out = output.create_group("data")
        _copy_attrs(source["data"], data_out)
        data_out.attrs["corrective_sampling"] = (
            "unaltered expert windows cropped around bilateral gripper closure"
        )
        train_source = _decode_mask(source["mask/train"][:])
        valid_source = _decode_mask(source["mask/valid"][:])
        train_out: list[str] = []
        valid_out: list[str] = []
        output_index = 0

        demos = sorted(source["data"], key=lambda name: int(name.split("_")[-1]))
        for demo_name in demos:
            demo = source["data"][demo_name]
            actions = demo["actions"][:]
            bilateral_close = np.flatnonzero(
                (actions[:, 18] > 0.0) & (actions[:, 19] > 0.0)
            )
            if bilateral_close.size == 0:
                raise RuntimeError(f"{demo_name}: no bilateral gripper close event")
            onset = int(bilateral_close[0])
            split = (
                "train" if demo_name in train_source
                else "valid" if demo_name in valid_source
                else None
            )
            if split is None:
                raise RuntimeError(f"{demo_name}: absent from source train/valid masks")

            for pre_close in args.pre_close_frames:
                start = max(0, onset - pre_close)
                output_index += 1
                new_name = f"demo_{output_index}"
                new_demo = data_out.create_group(new_name)
                _slice_group(demo, new_demo, start)
                new_demo.attrs["num_samples"] = int(actions.shape[0] - start)
                new_demo.attrs["source_demo"] = demo_name
                new_demo.attrs["source_start"] = start
                new_demo.attrs["close_onset_in_source"] = onset
                new_demo.attrs["pre_close_frames"] = pre_close
                (train_out if split == "train" else valid_out).append(new_name)
                audit.append(
                    {
                        "demo": new_name,
                        "source_demo": demo_name,
                        "split": split,
                        "start": start,
                        "length": int(actions.shape[0] - start),
                        "close_onset": onset,
                        "pre_close_frames": pre_close,
                    }
                )

        mask = output.create_group("mask")
        string_dtype = h5py.string_dtype(encoding="utf-8")
        mask.create_dataset(
            "train", data=np.asarray(train_out, dtype=object), dtype=string_dtype
        )
        mask.create_dataset(
            "valid", data=np.asarray(valid_out, dtype=object), dtype=string_dtype
        )
        data_out.attrs["total"] = sum(item["length"] for item in audit)

    temporary.replace(output_path)
    manifest = {
        "schema": 1,
        "source": str(source_path),
        "output": str(output_path),
        "method": "unaltered overlapping expert windows around bilateral close",
        "pre_close_frames": args.pre_close_frames,
        "episodes": len(audit),
        "samples": sum(item["length"] for item in audit),
        "train_episodes": sum(item["split"] == "train" for item in audit),
        "valid_episodes": sum(item["split"] == "valid" for item in audit),
        "windows": audit,
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("episodes", "samples", "train_episodes", "valid_episodes")}))
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
