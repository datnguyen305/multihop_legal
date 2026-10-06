"""Re-score saved IR rankings against the official split qrels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import read_jsonl, read_qrels, retrieval_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rankings = {
        str(row["query_id"]): [str(result["doc_id"]) for result in row.get("results", [])]
        for row in read_jsonl(args.rankings)
    }
    metrics = retrieval_metrics(read_qrels(args.qrels), rankings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
