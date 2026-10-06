"""Shared data loading, validation, and retrieval metrics."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            records.append(value)
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 4:
                raise ValueError(f"Expected 4 qrels columns at {path}:{line_number}")
            query_id, _iteration, doc_id, relevance = fields
            qrels.setdefault(query_id, {})[doc_id] = int(relevance)
    return qrels


def write_qrels(path: Path, qrels: dict[str, dict[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for query_id in sorted(qrels):
            for doc_id, relevance in sorted(qrels[query_id].items()):
                stream.write(f"{query_id}\t0\t{doc_id}\t{relevance}\n")


def validate_queries_and_qrels(
    queries: list[dict[str, Any]], qrels: dict[str, dict[str, int]], corpus_ids: set[str]
) -> None:
    query_ids = {str(row["query_id"]) for row in queries}
    if query_ids != set(qrels):
        missing = sorted(query_ids - set(qrels))[:5]
        extra = sorted(set(qrels) - query_ids)[:5]
        raise ValueError(f"Query/qrels IDs differ; missing={missing}, extra={extra}")
    absent = sorted({doc for rels in qrels.values() for doc, score in rels.items()
                     if score > 0 and doc not in corpus_ids})
    if absent:
        raise ValueError(f"{len(absent)} relevant doc IDs are absent from corpus: {absent[:5]}")
    no_positive = [qid for qid, rels in qrels.items() if not any(v > 0 for v in rels.values())]
    if no_positive:
        raise ValueError(f"{len(no_positive)} queries have no positive qrels: {no_positive[:5]}")


def retrieval_metrics(
    qrels: dict[str, dict[str, int]], rankings: dict[str, list[str]], cutoffs=(1, 5, 10, 20, 100)
) -> dict[str, float | int]:
    """Compute macro Recall, MRR, nDCG, MAP and complete-evidence recall."""
    query_ids = sorted(qrels)
    result: dict[str, float | int] = {"num_queries": len(query_ids)}
    if not query_ids:
        return result
    for k in cutoffs:
        recalls, reciprocal_ranks, ndcgs, average_precisions, complete = [], [], [], [], []
        for query_id in query_ids:
            relevant = {doc for doc, gain in qrels[query_id].items() if gain > 0}
            ranked = list(dict.fromkeys(rankings.get(query_id, [])))[:k]
            hits = [doc for doc in ranked if doc in relevant]
            recalls.append(len(hits) / len(relevant) if relevant else 0.0)
            reciprocal_ranks.append(1.0 / (ranked.index(hits[0]) + 1) if hits else 0.0)
            dcg = sum((2 ** (qrels[query_id].get(doc, 0)) - 1) / math.log2(i + 2)
                      for i, doc in enumerate(ranked))
            ideal = sorted((gain for gain in qrels[query_id].values() if gain > 0), reverse=True)[:k]
            idcg = sum((2 ** gain - 1) / math.log2(i + 2) for i, gain in enumerate(ideal))
            ndcgs.append(dcg / idcg if idcg else 0.0)
            precision_sum = 0.0
            found = 0
            for rank, doc in enumerate(ranked, 1):
                if doc in relevant:
                    found += 1
                    precision_sum += found / rank
            average_precisions.append(precision_sum / min(k, len(relevant)) if relevant else 0.0)
            complete.append(float(bool(relevant) and relevant.issubset(set(ranked))))
        result[f"recall@{k}"] = sum(recalls) / len(recalls)
        result[f"mrr@{k}"] = sum(reciprocal_ranks) / len(reciprocal_ranks)
        result[f"ndcg@{k}"] = sum(ndcgs) / len(ndcgs)
        result[f"map@{k}"] = sum(average_precisions) / len(average_precisions)
        result[f"complete_evidence@{k}"] = sum(complete) / len(complete)
    return result
