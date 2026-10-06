#!/usr/bin/env python3
"""Filter QA samples that contain multiple supporting contexts.

The input and output files use the project's existing mapping format:
``sample_id -> sample``.  Sample IDs and sample contents are preserved; only
samples with at least ``min_contexts`` contexts are written to the output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SPLITS = ("train", "dev", "test")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Create multihop train/dev/test files from QA JSON files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=project_root / "dataset" / "QA",
        help="Directory containing <split>_data.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "dataset" / "QA",
        help="Directory where <split>_multihop.json files will be written.",
    )
    parser.add_argument(
        "--min-contexts",
        type=int,
        default=2,
        help="Minimum number of contexts required to keep a sample (default: 2).",
    )
    return parser.parse_args()


def load_samples(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        samples = json.load(file)

    if not isinstance(samples, dict):
        raise ValueError(f"Expected a JSON object in {path}, got {type(samples).__name__}")
    return samples


def filter_samples(samples: dict[str, Any], min_contexts: int) -> dict[str, Any]:
    filtered: dict[str, Any] = {}
    for sample_id, sample in samples.items():
        if not isinstance(sample, dict):
            raise ValueError(f"Sample {sample_id!r} must be a JSON object")
        contexts = sample.get("contexts")
        if not isinstance(contexts, dict):
            raise ValueError(f"Sample {sample_id!r} must contain a contexts object")
        if len(contexts) >= min_contexts:
            filtered[sample_id] = sample
    return filtered


def write_samples(path: Path, samples: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(samples, file, ensure_ascii=False, indent=2)
        file.write("\n")


def main() -> None:
    args = parse_args()
    if args.min_contexts < 1:
        raise ValueError("--min-contexts must be at least 1")

    for split in SPLITS:
        input_path = args.input_dir / f"{split}_data.json"
        output_path = args.output_dir / f"{split}_multihop.json"
        samples = load_samples(input_path)
        filtered = filter_samples(samples, args.min_contexts)
        write_samples(output_path, filtered)
        print(
            f"{split}: {len(samples):,} -> {len(filtered):,} samples "
            f"(minimum contexts: {args.min_contexts})"
        )


if __name__ == "__main__":
    main()
