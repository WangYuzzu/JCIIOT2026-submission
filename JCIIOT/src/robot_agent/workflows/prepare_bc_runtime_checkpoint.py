"""Export BC checkpoints with image normalization stats in runtime HWC layout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for source in args.checkpoints:
        source = source.resolve()
        data = torch.load(source, map_location="cpu", weights_only=False)
        stats = data.get("obs_normalization_stats")
        if not isinstance(stats, dict):
            raise RuntimeError(f"{source}: no observation normalization stats")
        transposed = []
        for key, key_stats in stats.items():
            if not key.endswith("_image"):
                continue
            for field in ("offset", "scale"):
                value = np.asarray(key_stats[field])
                if value.ndim != 4 or value.shape[1] not in (1, 3, 4):
                    raise RuntimeError(
                        f"{source}: unexpected {key}.{field} shape {value.shape}"
                    )
                key_stats[field] = np.transpose(value, (0, 2, 3, 1)).copy()
            transposed.append(key)
        if not transposed:
            raise RuntimeError(f"{source}: no image statistics transposed")

        destination = args.output_dir / source.name
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(data, temporary)
        temporary.replace(destination)
        record = {
            "source": str(source),
            "source_sha256": _sha256(source),
            "output": str(destination.resolve()),
            "output_sha256": _sha256(destination),
            "transposed_image_stats": transposed,
        }
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    manifest = args.output_dir / "runtime_checkpoint_manifest.json"
    manifest.write_text(json.dumps({"records": records}, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
