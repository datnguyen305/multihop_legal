#!/usr/bin/env python3
"""Prepare weakly labelled extractive QA data from structured passages."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .data import read_jsonl, write_jsonl
from .extractive_data import build_passages, find_answer_span


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structured-ir-dir", type=Path, default=root / "dataset/IR/structured")
    parser.add_argument(
        "--structured-data-dir", type=Path, default=root / "dataset/IR/structured_data"
    )
    parser.add_argument("--qa-dir", type=Path, default=root / "dataset/QA")
    parser.add_argument("--output-dir", type=Path, default=root / "dataset/QA/extractive")
    parser.add_argument("--num-distractors", type=int, default=0)
    parser.add_argument("--max-passages", type=int, default=6)
    parser.add_argument("--passage-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_answers(path: Path) -> dict[str, dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected object in {path}")
    return value


def select_related_article(
    context: dict[str, Any],
    question: str,
    structured_data_dir: Path,
    document_cache: dict[Path, dict[str, Any]],
) -> dict[str, Any]:
    """For full-document fallbacks, select one article using the question only."""
    if not str(context.get("doc_id", "")).endswith("#document"):
        return context
    source_file = context.get("source_file")
    if not source_file:
        return context
    path = structured_data_dir / source_file
    if not path.is_file():
        return context
    if path not in document_cache:
        document_cache[path] = json.loads(path.read_text(encoding="utf-8"))
    document = document_cache[path]
    articles = document.get("Điều", [])
    if not isinstance(articles, list) or not articles:
        return context

    stopwords = {
        "là", "và", "của", "cho", "trong", "với", "theo", "được", "có",
        "những", "nào", "gì", "thế", "nào", "như", "để", "khi", "thì",
        "phải", "được", "thực", "hiện", "quy", "định", "các", "một",
    }
    query_terms = {
        token
        for token in re.findall(r"\w+", question.casefold(), flags=re.UNICODE)
        if len(token) > 1 and token not in stopwords
    }
    if not query_terms:
        return context

    best_article = None
    best_score = 0.0
    for article in articles:
        if not isinstance(article, dict):
            continue
        body = str(article.get("title", ""))
        heading = body.splitlines()[0] if body else ""
        body_terms = set(re.findall(r"\w+", body.casefold(), flags=re.UNICODE))
        heading_terms = set(re.findall(r"\w+", heading.casefold(), flags=re.UNICODE))
        score = len(query_terms & body_terms) + 2.0 * len(query_terms & heading_terms)
        if score > best_score:
            best_article = article
            best_score = score
    if best_article is None:
        return context

    result = dict(context)
    number = str(best_article.get("number", ""))
    title = str(best_article.get("title", "")).strip()
    result["doc_id"] = f"{str(context['doc_id']).removesuffix('#document')}#dieu={number}"
    result["section_type"] = "article"
    result["section_number"] = number
    heading = title.splitlines()[0].strip() if title else ""
    result["title"] = f"Điều {number}. {heading}".strip()
    result["text"] = f"Điều {number}. {title}".strip()
    return result


def build_lexical_index(corpus: list[dict[str, Any]]) -> dict[str, set[int]]:
    index: dict[str, set[int]] = defaultdict(set)
    for record_index, record in enumerate(corpus):
        text = (str(record.get("title", "")) + " " + str(record.get("text", ""))).lower()
        for token in set(re.findall(r"\w+", text, flags=re.UNICODE)):
            index[token].add(record_index)
    return index


def lexical_distractors(
    query: str,
    corpus: list[dict[str, Any]],
    positive_ids: set[str],
    number: int,
    seed: int,
    index: dict[str, set[int]],
) -> list[str]:
    if number <= 0:
        return []
    scores: Counter[int] = Counter()
    for token in set(re.findall(r"\w+", query.lower(), flags=re.UNICODE)):
        for record_index in index.get(token, set()):
            scores[record_index] += 1
    ranked = sorted(
        ((score, corpus[record_index]) for record_index, score in scores.items()),
        key=lambda item: (-item[0], item[1]["doc_id"]),
    )
    selected = [
        record["doc_id"] for _, record in ranked if record["doc_id"] not in positive_ids
    ][:number]
    if len(selected) < number:
        remaining = [
            record["doc_id"]
            for record in corpus
            if record["doc_id"] not in positive_ids and record["doc_id"] not in selected
        ]
        random.Random(seed).shuffle(remaining)
        selected.extend(remaining[: number - len(selected)])
    return selected


def prepare_split(args: argparse.Namespace, split: str) -> dict[str, Any]:
    answers = load_answers(args.qa_dir / f"{split}_multihop.json")
    samples = read_jsonl(args.structured_ir_dir / f"{split}_samples.jsonl")
    corpus = read_jsonl(args.structured_ir_dir / f"{split}_corpus.jsonl")
    corpus_by_id = {record["doc_id"]: record for record in corpus}
    lexical_index = build_lexical_index(corpus) if args.num_distractors > 0 else {}
    document_cache: dict[Path, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    label_counts: dict[str, int] = {}
    zero_overlap = 0
    document_fallbacks_trimmed = 0
    single_input_context = 0

    for sample in samples:
        source_id = str(sample["source_qa_id"])
        qa_sample = answers[source_id]
        positive_ids = list(dict.fromkeys(sample["positive_doc_ids"]))
        distractor_ids = lexical_distractors(
            qa_sample["question"],
            corpus,
            set(positive_ids),
            args.num_distractors,
            args.seed + int(hashlib.sha1(source_id.encode()).hexdigest()[:8], 16),
            lexical_index,
        )
        candidate_ids = positive_ids + distractor_ids
        contexts = []
        for index, doc_id in enumerate(candidate_ids):
            if doc_id not in corpus_by_id:
                raise KeyError(f"Missing corpus document {doc_id}")
            context = dict(corpus_by_id[doc_id])
            context["is_positive"] = doc_id in set(positive_ids)
            context["candidate_index"] = index
            original_is_document = str(context.get("doc_id", "")).endswith("#document")
            context = select_related_article(
                context,
                qa_sample["question"],
                args.structured_data_dir,
                document_cache,
            )
            document_fallbacks_trimmed += int(
                original_is_document and context.get("section_type") == "article"
            )
            contexts.append(context)

        # Allocate one chunk per source context before adding second chunks,
        # ensuring a long first document cannot consume the entire token budget.
        # Optional lexical negatives are appended after all positive contexts.
        passages, flat_tokens = build_passages(
            contexts,
            max_passages=args.max_passages,
            passage_tokens=args.passage_tokens,
        )
        span_boundaries = [
            (int(passage["global_start"]), int(passage["global_end"]))
            for passage in passages
        ]
        span = find_answer_span(
            qa_sample["answer"], flat_tokens, span_boundaries=span_boundaries
        )
        label_counts[span["method"]] = label_counts.get(span["method"], 0) + 1
        zero_overlap += int(span["overlap_score"] == 0.0)
        input_context_count = len(
            {
                passage["source_context_index"]
                for passage in passages
                if passage["is_positive"]
            }
        )
        single_input_context += int(input_context_count < 2)
        for passage in passages:
            overlaps_span = (
                passage["global_start"] <= span["end"]
                and passage["global_end"] >= span["start"]
            )
            passage["is_answer_passage"] = bool(
                span["overlap_score"] > 0 and overlaps_span
            )
        records.append(
            {
                "query_id": sample["query_id"],
                "source_qa_id": source_id,
                "question": qa_sample["question"],
                "answer": qa_sample["answer"],
                "passages": passages,
                "flat_tokens": flat_tokens,
                "start_position": span["start"],
                "end_position": span["end"],
                "label_method": span["method"],
                "overlap_score": span["overlap_score"],
                "positive_doc_ids": positive_ids,
                "source_context_count": sample["source_context_count"],
                "structured_context_count": sample["structured_context_count"],
                "input_context_count": input_context_count,
            }
        )
        if len(records) % 1000 == 0:
            print(f"{split}: prepared {len(records):,}/{len(samples):,}", flush=True)

    multihop_records = [
        record
        for record in records
        if len(record["positive_doc_ids"]) >= 2 and record["input_context_count"] >= 2
    ]
    write_jsonl(args.output_dir / "all_matched" / f"{split}.jsonl", records)
    write_jsonl(
        args.output_dir / "true_multihop" / f"{split}.jsonl", multihop_records
    )
    return {
        "all_matched": {
            "samples": len(records),
            "label_methods": label_counts,
            "zero_overlap": zero_overlap,
            "samples_with_fewer_than_two_input_contexts": single_input_context,
        },
        "true_multihop": {
            "samples": len(multihop_records),
            "label_methods": {
                method: sum(
                    record["label_method"] == method for record in multihop_records
                )
                for method in label_counts
            },
            "zero_overlap": sum(
                record["overlap_score"] == 0 for record in multihop_records
            ),
            "samples_with_fewer_than_two_input_contexts": sum(
                record["input_context_count"] < 2 for record in multihop_records
            ),
        },
        "num_distractors": args.num_distractors,
        "document_fallbacks_trimmed": document_fallbacks_trimmed,
    }


def main() -> None:
    args = parse_args()
    if args.max_passages < 2 or args.passage_tokens < 1:
        raise ValueError("Multihop extractive data requires >=2 passages and positive passage size")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "source": str(args.structured_ir_dir),
        "max_passages": args.max_passages,
        "passage_tokens": args.passage_tokens,
        "num_distractors": args.num_distractors,
        "document_fallback_selection": "question-only term overlap",
        "passage_context_policy": "round-robin source contexts; answer labels stay within one passage",
        "seed": args.seed,
        "splits": {},
    }
    for split in ("train", "dev", "test"):
        manifest["splits"][split] = prepare_split(args, split)
        print(split, manifest["splits"][split])
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
