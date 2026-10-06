#!/usr/bin/env python3
"""Create IR samples, per-split corpora, and qrels from multihop QA data.

Each QA context contains a ``content`` reference such as
``context_136938.json``. The referenced structured JSON file is joined into
the IR corpus, with its ``passage`` used as the retrievable text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SPLITS = ("train", "dev", "test")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Create IR samples and qrels from *_multihop.json files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=project_root / "dataset" / "QA",
        help="Directory containing <split>_multihop.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "dataset" / "IR",
        help="Directory where IR files will be written.",
    )
    parser.add_argument(
        "--contexts-dir",
        type=Path,
        default=project_root / "dataset" / "IR" / "contexts",
        help="Directory containing structured context JSON files.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def get_doc_id(context: dict[str, Any]) -> str:
    content_ref = context.get("content")
    if isinstance(content_ref, str) and content_ref.strip():
        return content_ref.strip()

    serialized = json.dumps(context, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:16]
    return f"context_{digest}"


def load_structured_context(
    content_ref: str, contexts_dir: Path
) -> dict[str, Any]:
    path = contexts_dir / content_ref
    if not path.is_file():
        raise FileNotFoundError(
            f"Structured context {content_ref!r} referenced by QA data was not found at {path}"
        )
    with path.open("r", encoding="utf-8") as file:
        structured = json.load(file)
    if not isinstance(structured, dict):
        raise ValueError(f"Structured context {path} must contain a JSON object")
    return structured


def corpus_record(
    doc_id: str,
    qa_context: dict[str, Any],
    structured_context: dict[str, Any],
) -> dict[str, Any]:
    passage = structured_context.get("passage", "")
    if not isinstance(passage, str):
        raise ValueError(f"Structured context {doc_id} has a non-string passage")

    qa_metadata = {
        key: value for key, value in qa_context.items() if key != "content"
    }
    structured_metadata = {
        key: value
        for key, value in structured_context.items()
        if key != "passage"
    }
    return {
        "doc_id": doc_id,
        "text": passage,
        "title": structured_context.get("name", qa_context.get("document", "")),
        "url": structured_context.get("link", qa_context.get("link", "")),
        "content_ref": qa_context.get("content", ""),
        "structured_metadata": structured_metadata,
        "qa_metadata": qa_metadata,
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def create_split(
    split: str,
    input_dir: Path,
    output_dir: Path,
    contexts_dir: Path,
) -> tuple[int, int, int, int]:
    qa_samples = load_json(input_dir / f"{split}_multihop.json")
    ir_samples: list[dict[str, Any]] = []
    corpus: dict[str, dict[str, Any]] = {}
    qrels: list[tuple[str, str]] = []

    for source_qa_id, sample in qa_samples.items():
        if not isinstance(sample, dict):
            raise ValueError(f"Sample {source_qa_id!r} in {split} is not an object")
        contexts = sample.get("contexts")
        if not isinstance(contexts, dict) or len(contexts) < 2:
            raise ValueError(
                f"Sample {source_qa_id!r} in {split} must contain at least 2 contexts"
            )

        query_id = f"{split}_{source_qa_id}"
        positive_doc_ids: list[str] = []
        for context in contexts.values():
            if not isinstance(context, dict):
                raise ValueError(f"Invalid context in sample {source_qa_id!r}")
            doc_id = get_doc_id(context)
            if doc_id not in corpus:
                content_ref = context.get("content")
                if not isinstance(content_ref, str) or not content_ref.strip():
                    raise ValueError(
                        f"Sample {source_qa_id!r} has a context without a content reference"
                    )
                structured_context = load_structured_context(
                    content_ref.strip(), contexts_dir
                )
                corpus[doc_id] = corpus_record(
                    doc_id, context, structured_context
                )
            if doc_id not in positive_doc_ids:
                positive_doc_ids.append(doc_id)

        ir_samples.append(
            {
                "query_id": query_id,
                "query": sample.get("question", ""),
                "positive_doc_ids": positive_doc_ids,
                "source_qa_id": source_qa_id,
            }
        )
        qrels.extend((query_id, doc_id) for doc_id in positive_doc_ids)

    write_jsonl(output_dir / f"{split}_samples.jsonl", ir_samples)
    write_jsonl(output_dir / f"{split}_corpus.jsonl", list(corpus.values()))
    with (output_dir / f"{split}_qrels.tsv").open("w", encoding="utf-8") as file:
        for query_id, doc_id in qrels:
            file.write(f"{query_id}\t0\t{doc_id}\t1\n")

    text_count = sum(bool(record["text"]) for record in corpus.values())
    return len(ir_samples), len(corpus), len(qrels), text_count


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        sample_count, corpus_count, qrel_count, text_count = create_split(
            split, args.input_dir, args.output_dir, args.contexts_dir
        )
        print(
            f"{split}: queries={sample_count:,}, documents={corpus_count:,}, "
            f"qrels={qrel_count:,}, documents_with_text={text_count:,}"
        )


if __name__ == "__main__":
    main()
