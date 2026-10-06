"""Build a shared passage-level IR corpus and matched query/qrels tracks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.create_ir_from_structured import (
    find_article,
    flatten_document,
    load_structured,
)
from .data import read_jsonl, read_qrels, validate_queries_and_qrels, write_jsonl, write_qrels


SPLITS = ("train", "dev", "test")


def source_paths_from_corpora(ir_dir: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        for record in read_jsonl(ir_dir / f"{split}_corpus.jsonl"):
            records.setdefault(record["source_file"], {})
            # Preserve metadata and original title/url for positive passages.
            prior = records[record["source_file"]]
            if record.get("title"):
                prior.setdefault("title", record["title"])
            if record.get("url"):
                prior.setdefault("url", record["url"])
    return records


def gold_document_records(ir_dir: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for split in SPLITS:
        for record in read_jsonl(ir_dir / f"{split}_corpus.jsonl"):
            result.setdefault(record["doc_id"], record)
    return result


def make_global_corpus(structured_dir: Path, ir_dir: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    source_metadata = source_paths_from_corpora(ir_dir)
    gold_records = gold_document_records(ir_dir)
    corpus: dict[str, dict[str, Any]] = {}
    counts = {"source_documents": 0, "article_passages": 0, "full_documents": 0}
    for source_file in sorted(structured_dir.rglob("*.json")):
        relative = source_file.relative_to(structured_dir).as_posix()
        structured = load_structured(source_file)
        metadata = source_metadata.get(relative, {})
        title = str(metadata.get("title") or structured.get("Header") or source_file.stem)
        url = str(metadata.get("url") or "")
        counts["source_documents"] += 1
        articles = structured.get("Điều", [])
        if isinstance(articles, list):
            for article in articles:
                if not isinstance(article, dict) or article.get("number") is None:
                    continue
                number = str(article["number"])
                doc_id = f"{relative}#dieu={number}"
                found = find_article(structured, number)
                if found is None:
                    continue
                text, article_metadata = found
                record = {
                    "doc_id": doc_id,
                    "text": text,
                    "title": title,
                    "url": url,
                    "source_file": relative,
                    "section_type": "article",
                    "section_number": number,
                    "structured_metadata": article_metadata,
                }
                corpus.setdefault(doc_id, gold_records.get(doc_id, record))
                counts["article_passages"] += 1

        # Full-document records are retained only where they are gold in this
        # benchmark (or the source has no article-level structure). This avoids
        # duplicating every long source document beside all its short sections.
        document_id = f"{relative}#document"
        if document_id in gold_records or not (isinstance(articles, list) and any(
            isinstance(a, dict) and a.get("number") is not None for a in articles
        )):
            text = flatten_document(structured)
            if text.strip():
                record = {
                    "doc_id": document_id,
                    "text": text,
                    "title": title,
                    "url": url,
                    "source_file": relative,
                    "section_type": "document",
                    "section_number": None,
                    "structured_metadata": {},
                }
                corpus.setdefault(document_id, gold_records.get(document_id, record))
                counts["full_documents"] += 1

    # Existing qrels are authoritative: sometimes a context names an article
    # number absent from the parsed ``Điều`` array, and the original join
    # correctly stored a full-document fallback under that requested ID.
    for doc_id, record in gold_records.items():
        corpus.setdefault(doc_id, record)
    rows = list(corpus.values())
    counts["corpus_passages"] = len(rows)
    counts["gold_passages"] = len(gold_records)
    return rows, counts


def filter_track(
    samples: list[dict[str, Any]], qrels: dict[str, dict[str, int]], track: str
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    if track == "all_matched":
        selected = samples
    elif track == "true_multihop":
        selected = [
            row for row in samples
            if sum(1 for gain in qrels[str(row["query_id"])].values() if gain > 0) >= 2
        ]
    else:
        raise ValueError(f"Unknown track {track}")
    ids = {str(row["query_id"]) for row in selected}
    return selected, {qid: qrels[qid] for qid in qrels if qid in ids}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ir-dir", type=Path, default=root / "dataset/IR/structured")
    parser.add_argument("--structured-dir", type=Path, default=root / "dataset/IR/structured_data")
    parser.add_argument("--output-dir", type=Path, default=root / "dataset/IR/experiments")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    corpus, corpus_counts = make_global_corpus(args.structured_dir, args.ir_dir)
    corpus_path = args.output_dir / "corpus.jsonl"
    write_jsonl(corpus_path, corpus)
    corpus_ids = {row["doc_id"] for row in corpus}
    manifest: dict[str, Any] = {
        "corpus_scope": "all numbered articles in structured_data, with only qrels-required full-document passages",
        "corpus_path": corpus_path.name,
        "corpus_sha256": sha256(corpus_path),
        "corpus_counts": corpus_counts,
        "tracks": {},
    }
    for track in ("all_matched", "true_multihop"):
        target = args.output_dir / track
        manifest["tracks"][track] = {}
        for split in SPLITS:
            samples = read_jsonl(args.ir_dir / f"{split}_samples.jsonl")
            qrels = read_qrels(args.ir_dir / f"{split}_qrels.tsv")
            selected, selected_qrels = filter_track(samples, qrels, track)
            validate_queries_and_qrels(selected, selected_qrels, corpus_ids)
            write_jsonl(target / f"{split}_queries.jsonl", selected)
            write_qrels(target / f"{split}_qrels.tsv", selected_qrels)
            manifest["tracks"][track][split] = {
                "queries": len(selected),
                "qrels": sum(len(rels) for rels in selected_qrels.values()),
                "queries_with_2plus_positives": sum(
                    sum(1 for gain in rels.values() if gain > 0) >= 2 for rels in selected_qrels.values()
                ),
            }
            print(f"{track}/{split}: queries={len(selected):,}, qrels={manifest['tracks'][track][split]['qrels']:,}")
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"corpus: {len(corpus):,} passages, sha256={manifest['corpus_sha256']}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
