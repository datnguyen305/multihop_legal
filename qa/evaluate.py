#!/usr/bin/env python3
"""Generate answers and evaluate a QA checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .data import METHODS, build_source, load_prepared_examples
from .metrics import compute_basic_metrics, compute_bertscore
from .modeling import configure_tokenizer, generation_kwargs


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument(
        "--model-name",
        default="",
        help="Original model name; inferred from run_args.json when omitted.",
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-source-length", type=int, default=2048)
    parser.add_argument("--max-target-length", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-bertscore", action="store_true")
    parser.add_argument("--bertscore-model", default="xlm-roberta-large")
    return parser.parse_args()


def main():
    args = parse_args()
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_name = args.model_name
    run_args_path = args.checkpoint / "run_args.json"
    if not model_name and run_args_path.is_file():
        model_name = json.loads(run_args_path.read_text(encoding="utf-8"))["model_name"]
    model_name = model_name or args.checkpoint.name
    tokenizer = configure_tokenizer(
        AutoTokenizer.from_pretrained(args.checkpoint), model_name
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    examples = load_prepared_examples(args.data, args.corpus)
    if args.limit:
        examples = examples[: args.limit]

    predictions = []
    references = []
    records = []
    for example in examples:
        source = build_source(example, args.method)
        inputs = tokenizer(
            source,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_source_length,
        ).to(device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_target_length,
                **generation_kwargs(tokenizer, model_name),
            )
        prediction = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        reference = str(example["answer"]).strip()
        predictions.append(prediction)
        references.append(reference)
        records.append(
            {
                "query_id": example["query_id"],
                "question": example["question"],
                "prediction": prediction,
                "reference": reference,
            }
        )

    metrics = compute_basic_metrics(predictions, references)
    if not args.skip_bertscore:
        metrics["bertscore_f1"] = compute_bertscore(
            predictions, references, args.bertscore_model
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output.with_suffix(".predictions.jsonl")
    with prediction_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    args.output.write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "method": args.method,
                "model_name": model_name,
                "data": str(args.data),
                "num_examples": len(examples),
                "metrics": metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
