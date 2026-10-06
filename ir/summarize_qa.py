"""Audit and summarize downstream QA metrics for a fixed reader."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=root / "runs/ir")
    parser.add_argument("--track", choices=("all_matched", "true_multihop"), default="true_multihop")
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--splits", nargs="+", choices=("dev", "test"), default=["dev"])
    parser.add_argument("--qa-top-k", type=int, default=10)
    args = parser.parse_args()
    results = []
    for split in args.splits:
        split_results = []
        for seed in args.seeds:
            for method in args.methods:
                path = (args.output_root / "runs" / f"{args.track}_{method}_seed{seed}"
                        / f"{split}_qa_metrics_top{args.qa_top_k}.json")
                if not path.is_file():
                    raise FileNotFoundError(f"Missing downstream QA metrics: {path}")
                record = json.loads(path.read_text(encoding="utf-8"))
                expected = {"retrieval_method": method, "retrieval_seed": seed, "split": split,
                            "track": args.track, "top_k_contexts": args.qa_top_k}
                mismatch = {key: (record.get(key), value) for key, value in expected.items()
                            if record.get(key) != value}
                if mismatch:
                    raise ValueError(f"Metadata mismatch in {path}: {mismatch}")
                split_results.append(record)
                results.append(record)
        for field in ("reader_checkpoint", "reader_method", "reader_model_name", "queries_sha256",
                      "num_examples", "max_passage_tokens"):
            values = {record.get(field) for record in split_results}
            if len(values) > 1:
                raise ValueError(f"Downstream QA methods do not share {field} for {split}: {values}")

    json_path = args.output_root / f"{args.track}_qa_comparison.json"
    tsv_path = args.output_root / f"{args.track}_qa_comparison.tsv"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = ["split", "retrieval_method", "retrieval_seed", "num_examples", "top_k_contexts",
              "mean_retrieved_contexts", "rougeL", "meteor", "bertscore_f1"]
    with tsv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for record in results:
            writer.writerow({**record, **record.get("metrics", {})})
    print(f"wrote {json_path}")
    print(f"wrote {tsv_path}")
    for record in results:
        metrics = record["metrics"]
        print(f"{record['split']:>4} {record['retrieval_method']:<8} seed={record['retrieval_seed']} "
              f"ROUGE-L={metrics.get('rougeL', 0):.4f} METEOR={metrics.get('meteor', 0):.4f} "
              f"BERTScore={metrics.get('bertscore_f1')}")


if __name__ == "__main__":
    main()
