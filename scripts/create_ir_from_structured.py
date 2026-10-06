#!/usr/bin/env python3
"""Build IR data by joining multihop QA contexts to structured legal data.

QA contexts identify their source document through the numeric suffix in the
source URL. Structured files are named with the same suffix when available.
The matching article (``Điều``) is used as the IR passage; documents without a
matching article are represented by a flattened full-document passage.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
import urllib.parse
from collections import defaultdict
from pathlib import Path
from typing import Any


SPLITS = ("train", "dev", "test")
STRUCTURE_KEYS = ("Phần", "Chương", "Mục", "Tiểu Mục", "Điều")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Create IR samples from multihop QA and structured legal data."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=root / "dataset" / "QA",
        help="Directory containing <split>_multihop.json files.",
    )
    parser.add_argument(
        "--structured-dir",
        type=Path,
        default=root / "dataset" / "IR" / "structured_data",
        help="Directory containing structured legal-document JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "dataset" / "IR" / "structured",
        help="Directory where structured IR files will be written.",
    )
    return parser.parse_args()


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", "", value)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def build_structured_index(
    structured_dir: Path,
) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    by_numeric_id: dict[str, list[Path]] = defaultdict(list)
    by_prefix: dict[str, list[Path]] = defaultdict(list)
    for path in structured_dir.rglob("*.json"):
        stem = normalize(path.stem)
        match = re.search(r"-(\d+)\.json$", path.name)
        if match:
            by_numeric_id[match.group(1)].append(path)
        by_prefix[stem[:50]].append(path)
    return by_numeric_id, by_prefix


def url_stem_and_id(url: str) -> tuple[str, str | None]:
    basename = urllib.parse.urlsplit(url).path.rsplit("/", 1)[-1]
    stem = re.sub(r"\.(?:html|aspx)$", "", basename, flags=re.IGNORECASE)
    match = re.search(r"-(\d+)$", stem)
    return stem, match.group(1) if match else None


def resolve_structured_file(
    url: str,
    by_numeric_id: dict[str, list[Path]],
    by_prefix: dict[str, list[Path]],
) -> Path | None:
    stem, numeric_id = url_stem_and_id(url)
    normalized_stem = normalize(stem)
    candidates = by_numeric_id.get(numeric_id, []) if numeric_id else []

    if len(candidates) == 1:
        return candidates[0]

    if candidates:
        slug_matches = [
            path
            for path in candidates
            if normalize(path.stem).startswith(normalized_stem)
            or normalized_stem.startswith(normalize(path.stem))
        ]
        if len(slug_matches) == 1:
            return slug_matches[0]

    prefix_candidates = [
        path
        for path in by_prefix.get(normalized_stem[:50], [])
        if normalize(path.stem).startswith(normalized_stem)
        or normalized_stem.startswith(normalize(path.stem))
    ]
    return prefix_candidates[0] if len(prefix_candidates) == 1 else None


def format_section(label: str, value: Any) -> str:
    if isinstance(value, str):
        return f"{label}\n{value}"
    if isinstance(value, dict):
        number = value.get("number")
        title = value.get("title", "")
        heading = f"{label} {number}" if number is not None else label
        return f"{heading}. {title}".strip()
    return f"{label}\n{value}"


def flatten_document(structured: dict[str, Any]) -> str:
    parts: list[str] = []
    header = structured.get("Header")
    if isinstance(header, str) and header.strip():
        parts.append(header.strip())
    for key in STRUCTURE_KEYS:
        values = structured.get(key, [])
        if not isinstance(values, list):
            continue
        for value in values:
            text = format_section(key, value).strip()
            if text:
                parts.append(text)
    return "\n\n".join(parts)


def find_article(
    structured: dict[str, Any], article_number: str | None
) -> tuple[str, dict[str, Any]] | None:
    if article_number is None:
        return None
    articles = structured.get("Điều", [])
    if not isinstance(articles, list):
        return None
    for article in articles:
        if not isinstance(article, dict):
            continue
        if str(article.get("number")) != str(article_number):
            continue
        title = article.get("title", "")
        if not isinstance(title, str):
            title = str(title)
        text = f"Điều {article_number}. {title}".strip()
        metadata = {
            key: value for key, value in article.items() if key != "title"
        }
        return text, metadata
    return None


def load_structured(path: Path) -> dict[str, Any]:
    value = load_json(path)
    if "Header" not in value and not any(key in value for key in STRUCTURE_KEYS):
        raise ValueError(f"Structured file has no recognized fields: {path}")
    return value


def make_corpus_record(
    doc_id: str,
    source_path: Path,
    structured: dict[str, Any],
    qa_context: dict[str, Any],
    article_number: str | None,
) -> dict[str, Any]:
    article = find_article(structured, article_number)
    if article is None:
        text = flatten_document(structured)
        section_type = "document"
        section_metadata: dict[str, Any] = {}
    else:
        text, section_metadata = article
        section_type = "article"

    return {
        "doc_id": doc_id,
        "text": text,
        "title": qa_context.get("document", source_path.stem),
        "url": qa_context.get("link", ""),
        "source_file": source_path.as_posix(),
        "section_type": section_type,
        "section_number": article_number,
        "structured_metadata": section_metadata,
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def create_split(
    split: str,
    input_dir: Path,
    structured_dir: Path,
    output_dir: Path,
    by_numeric_id: dict[str, list[Path]],
    by_prefix: dict[str, list[Path]],
) -> tuple[int, int, int, int, int]:
    qa_samples = load_json(input_dir / f"{split}_multihop.json")
    samples: list[dict[str, Any]] = []
    corpus: dict[str, dict[str, Any]] = {}
    qrels: list[tuple[str, str]] = []
    dropped: list[dict[str, Any]] = []
    structured_cache: dict[Path, dict[str, Any]] = {}
    matched_contexts = 0

    for source_qa_id, sample in qa_samples.items():
        contexts = sample.get("contexts") if isinstance(sample, dict) else None
        if not isinstance(contexts, dict) or len(contexts) < 2:
            raise ValueError(f"Sample {source_qa_id!r} is not a valid multihop sample")

        resolved: list[tuple[dict[str, Any], Path]] = []
        missing: list[str] = []
        for context in contexts.values():
            if not isinstance(context, dict):
                missing.append("<invalid-context>")
                continue
            path = resolve_structured_file(
                str(context.get("link", "")), by_numeric_id, by_prefix
            )
            if path is None:
                missing.append(str(context.get("link", "")))
            else:
                resolved.append((context, path))

        if missing:
            dropped.append(
                {
                    "source_qa_id": source_qa_id,
                    "reason": "missing_structured_context",
                    "missing_links": missing,
                }
            )
            continue

        query_id = f"{split}_{source_qa_id}"
        positive_doc_ids: list[str] = []
        source_refs: list[str] = []
        for context, path in resolved:
            if path not in structured_cache:
                structured_cache[path] = load_structured(path)
            article_number = context.get("điều")
            article_number = str(article_number) if article_number is not None else None
            relative_path = path.relative_to(structured_dir).as_posix()
            suffix = f"#dieu={article_number}" if article_number else "#document"
            doc_id = f"{relative_path}{suffix}"
            if doc_id not in corpus:
                corpus[doc_id] = make_corpus_record(
                    doc_id,
                    path.relative_to(structured_dir),
                    structured_cache[path],
                    context,
                    article_number,
                )
            if doc_id not in positive_doc_ids:
                positive_doc_ids.append(doc_id)
            content_ref = context.get("content")
            if content_ref not in source_refs:
                source_refs.append(content_ref)
            matched_contexts += 1

        samples.append(
            {
                "query_id": query_id,
                "query": sample.get("question", ""),
                "positive_doc_ids": positive_doc_ids,
                "source_qa_id": source_qa_id,
                "source_context_count": len(contexts),
                "structured_context_count": len(resolved),
                "source_context_refs": source_refs,
            }
        )
        qrels.extend((query_id, doc_id) for doc_id in positive_doc_ids)

    write_jsonl(output_dir / f"{split}_samples.jsonl", samples)
    write_jsonl(output_dir / f"{split}_corpus.jsonl", list(corpus.values()))
    write_jsonl(output_dir / f"{split}_dropped.jsonl", dropped)
    with (output_dir / f"{split}_qrels.tsv").open("w", encoding="utf-8") as file:
        for query_id, doc_id in qrels:
            file.write(f"{query_id}\t0\t{doc_id}\t1\n")

    return len(samples), len(corpus), len(qrels), len(dropped), matched_contexts


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    by_numeric_id, by_prefix = build_structured_index(args.structured_dir)
    print(
        f"indexed {sum(len(v) for v in by_numeric_id.values()):,} structured files "
        f"with numeric IDs"
    )
    for split in SPLITS:
        sample_count, corpus_count, qrel_count, dropped_count, matched_contexts = create_split(
            split,
            args.input_dir,
            args.structured_dir,
            args.output_dir,
            by_numeric_id,
            by_prefix,
        )
        print(
            f"{split}: queries={sample_count:,}, documents={corpus_count:,}, "
            f"qrels={qrel_count:,}, dropped={dropped_count:,}, "
            f"matched_contexts={matched_contexts:,}"
        )


if __name__ == "__main__":
    main()
