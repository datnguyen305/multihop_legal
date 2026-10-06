#!/usr/bin/env python3
"""Prepare answer-generation data from structured multihop IR outputs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from .data import read_jsonl, write_jsonl


SPLITS = ("train", "dev", "test")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--structured-ir-dir",
        type=Path,
        default=root / "dataset" / "IR" / "structured",
        help="Directory containing structured IR samples and corpora.",
    )
    parser.add_argument(
        "--qa-dir",
        type=Path,
        default=root / "dataset" / "QA",
        help="Directory containing <split>_multihop.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "dataset" / "QA" / "abstractive",
        help="Output directory for prepared QA tracks.",
    )
    parser.add_argument(
        "--with-distractors",
        action="store_true",
        help="Add BM25 hard negatives to candidate_doc_ids.",
    )
    parser.add_argument("--num-distractors", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_qa_answers(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected an object in {path}")
    return data


def bm25_candidates(
    query: str,
    corpus_records: list[dict[str, Any]],
    positive_ids: set[str],
    number: int,
    seed: int,
) -> list[str]:
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        raise RuntimeError(
            "--with-distractors requires rank-bm25; install requirements-qa.txt"
        ) from exc

    tokenized = [
        (str(record.get("title", "")) + " " + str(record.get("text", "")))
        .lower()
        .split()[:8192]
        for record in corpus_records
    ]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(
        zip(scores, corpus_records), key=lambda item: (-item[0], item[1]["doc_id"])
    )
    selected = [
        record["doc_id"]
        for _, record in ranked
        if record["doc_id"] not in positive_ids
    ][:number]
    if len(selected) < number:
        remaining = [
            record["doc_id"]
            for record in corpus_records
            if record["doc_id"] not in positive_ids and record["doc_id"] not in selected
        ]
        random.Random(seed).shuffle(remaining)
        selected.extend(remaining[: number - len(selected)])
    return selected


def prepare_split(args: argparse.Namespace, split: str) -> dict[str, int]:
    qa_answers = load_qa_answers(args.qa_dir / f"{split}_multihop.json")
    source_samples = read_jsonl(args.structured_ir_dir / f"{split}_samples.jsonl")
    corpus = read_jsonl(args.structured_ir_dir / f"{split}_corpus.jsonl")
    corpus_by_id = {record["doc_id"]: record for record in corpus}

    all_records: list[dict[str, Any]] = []
    true_multihop_records: list[dict[str, Any]] = []
    for sample in source_samples:
        source_qa_id = str(sample["source_qa_id"])
        if source_qa_id not in qa_answers:
            raise KeyError(f"QA sample {source_qa_id!r} is missing from {split}_multihop.json")
        positive_ids = list(dict.fromkeys(sample["positive_doc_ids"]))
        contexts = [
            {"doc_id": doc_id, "hop_index": index}
            for index, doc_id in enumerate(positive_ids)
        ]
        record: dict[str, Any] = {
            "query_id": sample["query_id"],
            "source_qa_id": source_qa_id,
            "question": qa_answers[source_qa_id]["question"],
            "answer": qa_answers[source_qa_id]["answer"],
            "contexts": contexts,
            "positive_doc_ids": positive_ids,
            "source_context_count": sample["source_context_count"],
            "structured_context_count": sample["structured_context_count"],
        }
        if args.with_distractors:
            distractors = bm25_candidates(
                record["question"],
                corpus,
                set(positive_ids),
                args.num_distractors,
                args.seed,
            )
            record["candidate_doc_ids"] = positive_ids + distractors
        all_records.append(record)
        if len(positive_ids) >= 2:
            true_multihop_records.append(record)

    write_jsonl(args.output_dir / "all_matched" / f"{split}.jsonl", all_records)
    write_jsonl(
        args.output_dir / "true_multihop" / f"{split}.jsonl", true_multihop_records
    )
    return {
        "all_matched": len(all_records),
        "true_multihop": len(true_multihop_records),
        "documents": len(corpus),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "source": str(args.structured_ir_dir),
        "with_distractors": args.with_distractors,
        "num_distractors": args.num_distractors if args.with_distractors else 0,
        "seed": args.seed,
        "splits": {},
    }
    for split in SPLITS:
        manifest["splits"][split] = prepare_split(args, split)
        print(split, manifest["splits"][split])
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
