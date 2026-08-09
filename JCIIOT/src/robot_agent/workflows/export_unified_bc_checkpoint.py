"""Export a robomimic multi-dataset checkpoint for the single-env runtime.

robomimic's ``MetaDataset`` stores one copy of ``env_metadata`` and
``shape_metadata`` per source dataset. The competition runtime loads a policy
with ``policy_from_checkpoint``, which expects one shape-metadata dictionary.
All unified datasets use the same observation and action schema, so the list
can be losslessly collapsed to its first entry for deployment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _all_equal(values: list[object]) -> bool:
    return all(value == values[0] for value in values[1:])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)

    shape_metadata = checkpoint.get("shape_metadata")
    if isinstance(shape_metadata, list):
        if not shape_metadata:
            raise ValueError("expected non-empty shape_metadata list")
        if not _all_equal(shape_metadata):
            raise ValueError("source datasets do not share one observation/action schema")
        checkpoint["shape_metadata"] = shape_metadata[0]
    elif not isinstance(shape_metadata, dict):
        raise ValueError("expected shape_metadata dictionary or non-empty list")

    # Environment creation in the competition path is driven by its explicit
    # scene argument, not checkpoint metadata. Retaining one representative
    # dictionary also keeps generic robomimic checkpoint tooling compatible.
    env_metadata = checkpoint.get("env_metadata")
    if isinstance(env_metadata, list):
        if not env_metadata:
            raise ValueError("expected non-empty env_metadata list")
        checkpoint["env_metadata"] = env_metadata[0]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(args.output)
    print(f"exported deployment checkpoint: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
