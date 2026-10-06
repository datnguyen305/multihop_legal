"""Measure retrieval's downstream QA quality with one fixed QA reader."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from qa.data import METHODS as QA_METHODS, build_source
from qa.metrics import compute_basic_metrics, compute_bertscore
from qa.modeling import configure_tokenizer, generation_kwargs
from .data import read_jsonl as read_ir_jsonl


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_examples(
    split: str,
    track_dir: Path,
    corpus_path: Path,
    rankings_path: Path,
    qa_dir: Path,
    top_k: int,
) -> list[dict[str, Any]]:
    query_rows = read_ir_jsonl(track_dir / f"{split}_queries.jsonl")
    ranking_rows = read_ir_jsonl(rankings_path)
    rankings = {
        str(row["query_id"]): [str(result["doc_id"]) for result in row.get("results", [])]
        for row in ranking_rows
    }
    requested_ids = {
        doc_id for row in query_rows for doc_id in rankings.get(str(row["query_id"]), [])[:top_k]
    }
    corpus = {}
    if requested_ids:
        with corpus_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                doc_id = record.get("doc_id")
                if doc_id in requested_ids:
                    corpus[doc_id] = record
                    if len(corpus) == len(requested_ids):
                        break
    missing_docs = requested_ids - set(corpus)
    if missing_docs:
        raise KeyError(f"{len(missing_docs)} retrieved passages are missing from corpus: {sorted(missing_docs)[:5]}")
    qa_path = qa_dir / f"{split}_multihop.json"
    with qa_path.open("r", encoding="utf-8") as stream:
        qa_data = json.load(stream)
    examples = []
    for row in query_rows:
        query_id = str(row["query_id"])
        source_id = str(row["source_qa_id"])
        if query_id not in rankings:
            raise KeyError(f"Ranking is missing query {query_id}")
        if source_id not in qa_data:
            raise KeyError(f"QA answer is missing source sample {source_id} in {qa_path}")
        answer_record = qa_data[source_id]
        contexts = []
        for doc_id in rankings[query_id][:top_k]:
            if doc_id not in corpus:
                raise KeyError(f"Retrieved doc {doc_id!r} is missing from shared corpus")
            contexts.append(dict(corpus[doc_id]))
        examples.append({
            "query_id": query_id,
            "question": str(answer_record["question"]),
            "answer": str(answer_record["answer"]),
            "candidate_contexts": contexts,
        })
    return examples


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--retrieval-method", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--track", choices=("all_matched", "true_multihop"), default="true_multihop")
    parser.add_argument("--data-dir", type=Path, default=root / "dataset/IR/experiments")
    parser.add_argument("--corpus", type=Path, default=root / "dataset/IR/experiments/corpus.jsonl")
    parser.add_argument("--qa-dir", type=Path, default=root / "dataset/QA")
    parser.add_argument("--reader-checkpoint", type=Path, required=True)
    parser.add_argument("--reader-method", choices=QA_METHODS, default="pathfid")
    parser.add_argument("--reader-model-name", default="")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-source-length", type=int, default=2048)
    parser.add_argument("--max-target-length", type=int, default=1024)
    parser.add_argument("--max-passage-tokens", type=int, default=256)
    parser.add_argument("--skip-bertscore", action="store_true")
    parser.add_argument("--bertscore-model", default="xlm-roberta-large")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_name = args.reader_model_name
    run_args_path = args.reader_checkpoint / "run_args.json"
    if not model_name and run_args_path.is_file():
        model_name = json.loads(run_args_path.read_text(encoding="utf-8"))["model_name"]
    model_name = model_name or args.reader_checkpoint.name
    tokenizer = configure_tokenizer(AutoTokenizer.from_pretrained(args.reader_checkpoint), model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.reader_checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    examples = make_examples(
        args.split, args.data_dir / args.track, args.corpus, args.rankings, args.qa_dir, args.top_k
    )
    if args.max_source_length < 256 or args.max_passage_tokens < 1:
        raise ValueError("--max-source-length must be >=256 and --max-passage-tokens must be positive")
    available = max(1, (args.max_source_length - 256) // args.top_k - 32)
    passage_budget = min(args.max_passage_tokens, available)
    predictions, references, records = [], [], []
    for number, example in enumerate(examples, 1):
        for context in example["candidate_contexts"]:
            passage_ids = tokenizer.encode(str(context.get("text", "")), add_special_tokens=False,
                                           truncation=True, max_length=passage_budget)
            context["text"] = tokenizer.decode(passage_ids, skip_special_tokens=True)
        source = build_source(example, args.reader_method, max_contexts=args.top_k)
        inputs = tokenizer(source, return_tensors="pt", truncation=True,
                           max_length=args.max_source_length).to(device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs, max_new_tokens=args.max_target_length,
                **generation_kwargs(tokenizer, model_name),
            )
        prediction = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        reference = example["answer"].strip()
        predictions.append(prediction)
        references.append(reference)
        records.append({"query_id": example["query_id"], "question": example["question"],
                        "prediction": prediction, "reference": reference})
        if number % 100 == 0:
            print(f"downstream QA: {number}/{len(examples)}")

    qa_metrics = compute_basic_metrics(predictions, references)
    if not args.skip_bertscore:
        qa_metrics["bertscore_f1"] = compute_bertscore(
            predictions, references, args.bertscore_model
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output.with_suffix(".predictions.jsonl")
    with prediction_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    metadata = {
        "retrieval_method": args.retrieval_method,
        "retrieval_seed": args.seed,
        "reader_checkpoint": str(args.reader_checkpoint.resolve()),
        "reader_method": args.reader_method,
        "reader_model_name": model_name,
        "split": args.split,
        "track": args.track,
        "top_k_contexts": args.top_k,
        "max_passage_tokens": passage_budget,
        "num_examples": len(examples),
        "mean_retrieved_contexts": round(
            sum(len(example["candidate_contexts"]) for example in examples) / max(1, len(examples)), 4
        ),
        "rankings_sha256": sha256(args.rankings),
        "queries_sha256": sha256(args.data_dir / args.track / f"{args.split}_queries.jsonl"),
        "metrics": qa_metrics,
    }
    args.output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(qa_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
