"""Audit comparability and summarize paired IR runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=root / "runs/ir")
    parser.add_argument("--track", choices=("all_matched", "true_multihop"), default="true_multihop")
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--splits", nargs="+", choices=("dev", "test"), default=["dev"])
    parser.add_argument("--search-depth", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    dense_family = {"dense", "mdr", "m3", "baleen", "mopo"}
    audited_models = [method for method in args.methods if method in dense_family]
    if "hybrid" in args.methods and "dense" not in audited_models:
        audited_models.append("dense")
    shared_train_fields = (
        "track", "encoder", "seed", "epochs", "max_steps", "max_train_examples",
        "batch_size", "gradient_accumulation", "learning_rate", "max_length",
        "temperature", "resolved_optimizer_steps", "num_train_queries",
        "train_queries_sha256", "train_qrels_sha256", "corpus_sha256",
    )
    for seed in args.seeds:
        train_configs = []
        for method in audited_models:
            config_path = args.output_root / "models" / f"{args.track}_{method}_seed{seed}" / "run_config.json"
            if not config_path.is_file():
                raise FileNotFoundError(f"Missing training audit config: {config_path}")
            with config_path.open("r", encoding="utf-8") as stream:
                train_configs.append((method, json.load(stream)))
        for field in shared_train_fields:
            values = {config.get(field) for _method, config in train_configs}
            if len(values) > 1:
                raise ValueError(f"Dense-family methods do not share {field} for seed {seed}: {values}")

    for split in args.splits:
        compared = []
        for seed in args.seeds:
            for method in args.methods:
                path = args.output_root / "runs" / f"{args.track}_{method}_seed{seed}" / f"{split}_metrics.json"
                if not path.is_file():
                    raise FileNotFoundError(f"Missing result metrics: {path}")
                with path.open("r", encoding="utf-8") as stream:
                    metrics = json.load(stream)
                expected = {"method": method, "seed": seed, "track": args.track, "split": split,
                            "search_depth": args.search_depth}
                mismatch = {key: (metrics.get(key), value) for key, value in expected.items()
                            if metrics.get(key) != value}
                if mismatch:
                    raise ValueError(f"Run metadata mismatch in {path}: {mismatch}")
                compared.append(metrics)
                rows.append(metrics)
        invariants = ("corpus_sha256", "qrels_sha256", "queries_sha256", "search_depth", "track", "split")
        for field in invariants:
            values = {entry.get(field) for entry in compared}
            if len(values) > 1:
                raise ValueError(f"Methods do not share {field} for {split}: {values}")

    output_json = args.output_root / f"{args.track}_comparison.json"
    output_tsv = args.output_root / f"{args.track}_comparison.tsv"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    keys = ["split", "method", "seed", "num_queries", "recall@1", "recall@5", "recall@10",
            "recall@20", "recall@100", "mrr@10", "ndcg@10", "map@10", "complete_evidence@100",
            "results_per_query_mean"]
    with output_tsv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output_json}")
    print(f"wrote {output_tsv}")
    for row in rows:
        print(f"{row['split']:>4}  {row['method']:<8} seed={row['seed']}  "
              f"R@10={row.get('recall@10', 0):.4f}  MRR@10={row.get('mrr@10', 0):.4f}  "
              f"nDCG@10={row.get('ndcg@10', 0):.4f}  Complete@100={row.get('complete_evidence@100', 0):.4f}")


if __name__ == "__main__":
    main()
