#!/usr/bin/env python3
"""Audit shared experiment settings and summarize extractive QA metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=root / "runs/extractive")
    parser.add_argument("--track", choices=("all_matched", "true_multihop"), default="true_multihop")
    parser.add_argument("--methods", nargs="+", default=["cog", "dfgn", "hgn", "qanet"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--splits", nargs="+", choices=("dev", "test"), default=["dev"])
    parser.add_argument("--expected-encoder")
    parser.add_argument("--expected-epochs", type=int)
    parser.add_argument("--expected-max-steps", type=int)
    parser.add_argument("--expected-max-train-examples", type=int)
    parser.add_argument("--expected-batch-size", type=int)
    parser.add_argument("--expected-gradient-accumulation", type=int)
    parser.add_argument("--expected-learning-rate", type=float)
    parser.add_argument("--expected-passage-loss-weight", type=float)
    parser.add_argument("--expected-max-passages", type=int)
    parser.add_argument("--expected-max-length", type=int)
    parser.add_argument("--expected-save-steps", type=int)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing experiment artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    run_args_by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    metrics_by_split: dict[str, dict[str, dict[int, dict[str, Any]]]] = {
        split: {method: {} for method in args.methods} for split in args.splits
    }
    checked_data_roots: set[Path] = set()
    checked_code_hashes = False
    code_root = Path(__file__).parent

    for seed in args.seeds:
        run_args_by_seed[seed] = {}
        for method in args.methods:
            run_dir = args.output_dir / f"{args.track}_{method}_seed{seed}"
            if not (run_dir / "COMPLETE").is_file():
                raise RuntimeError(f"Experiment is not complete: {run_dir}")
            run_args = load_json(run_dir / "run_args.json")
            if run_args.get("method") != method or int(run_args.get("seed", -1)) != seed:
                raise ValueError(f"Run metadata does not match its path: {run_dir}")
            if not checked_code_hashes:
                code_hashes = run_args.get("code_sha256", {})
                if not code_hashes:
                    raise ValueError(f"Run lacks code fingerprints: {run_dir}")
                for name, expected_hash in code_hashes.items():
                    path = code_root / name
                    if not path.is_file() or sha256_file(path) != expected_hash:
                        raise ValueError(f"Code changed since training: {path}")
                checked_code_hashes = True
            data_root = Path(run_args["data_dir"])
            if data_root not in checked_data_roots:
                hash_paths = {
                    "train": data_root / args.track / "train.jsonl",
                    "dev": data_root / args.track / "dev.jsonl",
                    "test": data_root / args.track / "test.jsonl",
                    "manifest": data_root / "manifest.json",
                }
                for name, path in hash_paths.items():
                    if not path.is_file() or sha256_file(path) != run_args["data_sha256"].get(name):
                        raise ValueError(f"Data changed or is missing since training: {path}")
                checked_data_roots.add(data_root)
            run_args_by_seed[seed][method] = run_args
            for split in args.splits:
                metric = load_json(run_dir / f"{split}_metrics.json")
                if (
                    metric.get("method") != method
                    or metric.get("encoder") != run_args.get("encoder")
                    or metric.get("encoder_revision") != run_args.get("encoder_revision")
                ):
                    raise ValueError(
                        f"Evaluation metadata does not match training for {run_dir}/{split}"
                    )
                metrics_by_split[split][method][seed] = metric

    excluded = {"method", "seed", "trainable_parameters"}
    normalized_runs = {
        (seed, method): {key: value for key, value in run_args.items() if key not in excluded}
        for seed, method_args in run_args_by_seed.items()
        for method, run_args in method_args.items()
    }
    reference_key = next(iter(normalized_runs))
    reference_config = normalized_runs[reference_key]
    mismatches = {
        f"seed={seed}/{method}": config
        for (seed, method), config in normalized_runs.items()
        if config != reference_config
    }
    if mismatches:
        raise ValueError(
            "Model runs do not share identical data/training settings; "
            f"reference={reference_key}, mismatches={list(mismatches)}"
        )

    requested = {
        "encoder": args.expected_encoder,
        "epochs": args.expected_epochs,
        "max_steps": args.expected_max_steps,
        "max_train_examples": args.expected_max_train_examples,
        "batch_size": args.expected_batch_size,
        "gradient_accumulation": args.expected_gradient_accumulation,
        "learning_rate": args.expected_learning_rate,
        "passage_loss_weight": args.expected_passage_loss_weight,
        "max_passages": args.expected_max_passages,
        "max_length": args.expected_max_length,
        "save_steps": args.expected_save_steps,
    }
    requested = {key: value for key, value in requested.items() if value is not None}
    changed_from_request = {
        key: (reference_config.get(key), value)
        for key, value in requested.items()
        if reference_config.get(key) != value
    }
    if changed_from_request:
        raise ValueError(
            "Existing runs differ from the requested launcher settings; use a new "
            f"OUTPUT_ROOT to avoid mixing experiments: {changed_from_request}"
        )

    summaries: dict[str, dict[str, Any]] = {}
    for split, methods in metrics_by_split.items():
        split_summary = {}
        reference_count = None
        reference_data_path = None
        for method, by_seed in methods.items():
            counts = {int(metric["num_examples"]) for metric in by_seed.values()}
            data_paths = {metric["data"] for metric in by_seed.values()}
            if len(counts) != 1 or len(data_paths) != 1:
                raise ValueError(
                    f"{split} evaluation differs between seeds for {method}: "
                    f"counts={counts}, data={data_paths}"
                )
            count = counts.pop()
            data_path = data_paths.pop()
            if reference_count is None:
                reference_count = count
            elif count != reference_count:
                raise ValueError(f"Different {split} sample counts across methods")
            if reference_data_path is None:
                reference_data_path = data_path
            elif data_path != reference_data_path:
                raise ValueError(f"Different {split} data paths across methods")
            metric_names = set.intersection(
                *(set(metric["metrics"]) for metric in by_seed.values())
            )
            values = {
                name: [float(by_seed[seed]["metrics"][name]) for seed in args.seeds]
                for name in sorted(metric_names)
                if all(by_seed[seed]["metrics"][name] is not None for seed in args.seeds)
            }
            split_summary[method] = {
                "num_examples": count,
                "data": data_path,
                "seeds": {
                    name: {
                        "values": scores,
                        "mean": statistics.mean(scores),
                        "std": statistics.stdev(scores) if len(scores) > 1 else None,
                    }
                    for name, scores in values.items()
                },
            }
        summaries[split] = split_summary

    result = {
        "track": args.track,
        "methods": args.methods,
        "seeds": args.seeds,
        "trainable_parameters": {
            method: {
                str(seed): run_args_by_seed[seed][method].get("trainable_parameters")
                for seed in args.seeds
            }
            for method in args.methods
        },
        "shared_run_configuration": reference_config,
        "metrics": summaries,
    }
    output_path = args.output_dir / f"{args.track}_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Fairness audit passed: {len(args.methods)} models × {len(args.seeds)} seed(s)")
    print(f"Shared config + data hashes: identical; summary: {output_path}")
    for method, by_seed in result["trainable_parameters"].items():
        print(f"{method} trainable parameters: {by_seed}")
    for split, method_summary in summaries.items():
        print(f"[{split}: n={next(iter(method_summary.values()))['num_examples']}]")
        for method, details in method_summary.items():
            scores = details["seeds"]
            rendered = "  ".join(
                f"{name}={value['mean']:.4f}"
                + (f" ± {value['std']:.4f}" if value["std"] is not None else "")
                for name, value in scores.items()
            )
            print(f"{method}: {rendered}")


if __name__ == "__main__":
    main()
