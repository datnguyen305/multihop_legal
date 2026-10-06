"""Data loading and method-specific input serialization for QA experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


METHODS = (
    "ask_to_understand",
    "generative_context_pair_selection",
    "msg",
    "pointer_generator",
    "pathfid",
)

METHOD_INSTRUCTIONS = {
    "ask_to_understand": (
        "Answer the legal question by asking yourself focused sub-questions "
        "about each passage before producing the final answer."
    ),
    "generative_context_pair_selection": (
        "Select the compatible legal passages that jointly answer the question "
        "and synthesize one answer from them."
    ),
    "msg": (
        "Select the relevant legal facts from the passages and write a concise, "
        "complete answer to the question."
    ),
    "pointer_generator": (
        "Answer using the legal passages. Preserve exact legal terms, article "
        "numbers, conditions, and enumerations when they are relevant."
    ),
    "pathfid": (
        "Follow the legal reasoning path across the passages in order and then "
        "generate the final answer."
    ),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            records.append(record)
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_corpus(path: Path) -> dict[str, dict[str, Any]]:
    return {record["doc_id"]: record for record in read_jsonl(path)}


def enrich_candidate_contexts(
    examples: list[dict[str, Any]], corpus: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    for example in examples:
        if example.get("contexts") and not example["contexts"][0].get("text"):
            expanded_contexts = []
            for context in example["contexts"]:
                doc_id = context["doc_id"]
                if doc_id not in corpus:
                    raise KeyError(f"Context document {doc_id!r} is missing from corpus")
                expanded = dict(corpus[doc_id])
                expanded["hop_index"] = context.get("hop_index")
                expanded_contexts.append(expanded)
            example["contexts"] = expanded_contexts
        candidate_ids = example.get("candidate_doc_ids")
        if not candidate_ids:
            continue
        contexts: list[dict[str, Any]] = []
        positive_ids = set(example.get("positive_doc_ids", []))
        for doc_id in candidate_ids:
            if doc_id not in corpus:
                raise KeyError(f"Candidate document {doc_id!r} is missing from corpus")
            context = dict(corpus[doc_id])
            context["is_positive"] = doc_id in positive_ids
            contexts.append(context)
        example["candidate_contexts"] = contexts
    return examples


def load_prepared_examples(
    path: Path, corpus_path: Path | None = None
) -> list[dict[str, Any]]:
    examples = read_jsonl(path)
    if corpus_path is not None:
        enrich_candidate_contexts(examples, load_corpus(corpus_path))
    return examples


def _context_label(context: dict[str, Any], index: int) -> str:
    title = context.get("title") or context.get("doc_id") or f"passage-{index}"
    return f"[PASSAGE_{index}] {title}"


def build_source(
    example: dict[str, Any],
    method: str,
    max_contexts: int | None = None,
) -> str:
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {METHODS}")

    contexts = example.get("candidate_contexts") or example.get("contexts", [])
    if max_contexts is not None:
        contexts = contexts[:max_contexts]

    question = str(example.get("question", "")).strip()
    instruction = METHOD_INSTRUCTIONS[method]
    blocks = [f"instruction: {instruction}", f"question: {question}"]

    for index, context in enumerate(contexts, start=1):
        label = _context_label(context, index)
        text = str(context.get("text", "")).strip()
        if method == "pathfid":
            blocks.append(f"<title-{index}> {label}\n<context-{index}> {text}")
        elif method == "generative_context_pair_selection":
            blocks.append(f"<candidate-{index}> {label}\n{text}")
        elif method == "msg":
            blocks.append(f"<source-{index}> {label}\n{text}")
        else:
            blocks.append(f"{label}\n{text}")

    return "\n\n".join(blocks)


def build_target(example: dict[str, Any]) -> str:
    answer = str(example.get("answer", "")).strip()
    if not answer:
        raise ValueError(f"Example {example.get('query_id')} has an empty answer")
    return answer
