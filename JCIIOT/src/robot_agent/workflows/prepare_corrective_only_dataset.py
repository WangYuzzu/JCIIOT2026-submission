"""Build a BC aggregate from original demos and successful recovery segments.

The corrective collector stores two regions per episode:

1. DAgger query labels for states visited by the current policy.
2. A smooth scripted-expert recovery that ends in a verified bilateral grasp.

This utility can discard the query region while preserving the physically
successful recovery trajectories. It is useful when an early query policy
produced overly aggressive labels, while the recovery controller itself
remained valid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--corrective", type=Path, required=True)
    parser.add_argument("--recovery-out", type=Path, required=True)
    parser.add_argument("--aggregate-out", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _decode(values) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ]


def _write_mask(group, key: str, values: list[str]) -> None:
    dtype = h5py.string_dtype(encoding="utf-8")
    group.create_dataset(key, data=np.asarray(values, dtype=object), dtype=dtype)


def _copy_attrs(source, target) -> None:
    for key, value in source.attrs.items():
        target.attrs[key] = value


def write_recovery_only(corrective: Path, output: Path) -> None:
    temporary = output.with_suffix(output.suffix + ".tmp")
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(corrective, "r") as source, h5py.File(temporary, "w") as out:
        data = out.create_group("data")
        _copy_attrs(source["data"], data)
        data.attrs["purpose"] = "smooth successful expert recovery segments only"
        data.attrs["source_corrective_dataset"] = str(corrective.resolve())
        total = 0
        for name in sorted(source["data"], key=lambda x: int(x.split("_")[-1])):
            old = source["data"][name]
            metadata = json.loads(old.attrs["corrective_metadata"])
            start = int(metadata["query_samples"])
            stop = int(old["actions"].shape[0])
            if not 0 <= start < stop:
                raise RuntimeError(f"{name}: invalid recovery slice {start}:{stop}")

            demo = data.create_group(name)
            _copy_attrs(old, demo)
            demo.attrs["num_samples"] = stop - start
            demo.attrs["recovery_slice_start"] = start
            for key in ("states", "actions"):
                demo.create_dataset(key, data=old[key][start:stop])
            obs = demo.create_group("obs")
            for key, dataset in old["obs"].items():
                options = (
                    {"compression": "gzip", "compression_opts": 4}
                    if key.endswith("_image")
                    else {}
                )
                obs.create_dataset(key, data=dataset[start:stop], **options)
            total += stop - start
        data.attrs["total"] = total
        data.attrs["num_successful_demos"] = len(data)
    temporary.replace(output)


def merge(source: Path, recovery: Path, output: Path) -> None:
    temporary = output.with_suffix(output.suffix + ".tmp")
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        h5py.File(source, "r") as original,
        h5py.File(recovery, "r") as corrections,
        h5py.File(temporary, "w") as out,
    ):
        data = out.create_group("data")
        _copy_attrs(original["data"], data)
        name_map: dict[tuple[str, str], str] = {}
        next_index = 1
        for origin, dataset in (("source", original), ("recovery", corrections)):
            for old_name in sorted(
                dataset["data"], key=lambda x: int(x.split("_")[-1])
            ):
                new_name = f"demo_{next_index}"
                dataset.copy(dataset["data"][old_name], data, name=new_name)
                data[new_name].attrs["aggregate_source"] = origin
                name_map[(origin, old_name)] = new_name
                next_index += 1
        data.attrs["num_successful_demos"] = next_index - 1
        data.attrs["aggregate_source_dataset"] = str(source.resolve())
        data.attrs["aggregate_recovery_dataset"] = str(recovery.resolve())

        masks = out.create_group("mask")
        original_train = (
            _decode(original["mask"]["train"][:])
            if "mask" in original and "train" in original["mask"]
            else list(original["data"])
        )
        original_valid = (
            _decode(original["mask"]["valid"][:])
            if "mask" in original and "valid" in original["mask"]
            else []
        )
        train = [name_map[("source", name)] for name in original_train]
        train.extend(
            name_map[("recovery", name)] for name in corrections["data"]
        )
        valid = [name_map[("source", name)] for name in original_valid]
        _write_mask(masks, "train", train)
        _write_mask(masks, "valid", valid)
    temporary.replace(output)


def main() -> int:
    args = parse_args()
    for path in (args.source, args.corrective):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (args.recovery_out, args.aggregate_out):
        if path.exists() and not args.force:
            raise FileExistsError(f"{path} exists; pass --force to replace it")
    write_recovery_only(args.corrective, args.recovery_out)
    merge(args.source, args.recovery_out, args.aggregate_out)
    print(args.recovery_out.resolve())
    print(args.aggregate_out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
