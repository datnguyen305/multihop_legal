#!/usr/bin/env python3
"""Evaluate extractive QA spans with the same answer metrics as abstractive QA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import read_jsonl
from .extractive_data import ExtractiveCollator, detokenize_tokens
from .extractive_models import MODEL_CLASSES, build_extractive_model
from .metrics import compute_basic_metrics, compute_bertscore
from .train_extractive import ExtractiveDataset


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--method", choices=sorted(MODEL_CLASSES), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-bertscore", action="store_true")
    parser.add_argument("--bertscore-model", default="xlm-roberta-large")
    return parser.parse_args()


def predict_span(start_logits, end_logits, global_positions, flat_tokens, max_answer_length=512):
    valid = global_positions >= 0
    start_values = start_logits.masked_fill(~valid, -1e4)
    end_values = end_logits.masked_fill(~valid, -1e4)
    start_candidates = torch.topk(start_values.flatten(), k=min(20, start_values.numel())).indices
    end_candidates = torch.topk(end_values.flatten(), k=min(20, end_values.numel())).indices
    best = (0, 0, -1e9)
    positions = global_positions.flatten()
    tokens_per_passage = start_logits.shape[-1]
    for start_index in start_candidates.tolist():
        start_global = int(positions[start_index])
        if start_global < 0:
            continue
        for end_index in end_candidates.tolist():
            if start_index // tokens_per_passage != end_index // tokens_per_passage:
                continue
            end_global = int(positions[end_index])
            if end_global < start_global or end_global < 0:
                continue
            if end_global - start_global + 1 > max_answer_length:
                continue
            score = float(start_values.flatten()[start_index] + end_values.flatten()[end_index])
            if score > best[2]:
                best = (start_global, end_global, score)
    start, end = best[:2]
    if not flat_tokens:
        return "", start, end
    return detokenize_tokens(flat_tokens[start : end + 1]), start, end


def main():
    args = parse_args()
    run_args = json.loads((args.checkpoint / "run_args.json").read_text(encoding="utf-8"))
    if run_args.get("method") != args.method:
        raise ValueError(
            f"Checkpoint method is {run_args.get('method')!r}, not {args.method!r}"
        )
    encoder = run_args["encoder"]
    max_passages = int(run_args.get("max_passages", 6))
    max_length = int(run_args.get("max_length", 512))
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)
    model = build_extractive_model(args.method, encoder)
    state_path = args.checkpoint / "model_state.pt"
    if not state_path.is_file():
        raise FileNotFoundError(f"Missing final model state: {state_path}")
    model.load_state_dict(torch.load(state_path, map_location="cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    examples = read_jsonl(args.data)
    if args.limit:
        examples = examples[: args.limit]
    loader = DataLoader(
        ExtractiveDataset(examples),
        batch_size=1,
        shuffle=False,
        collate_fn=ExtractiveCollator(tokenizer, max_passages, max_length),
    )
    predictions = []
    references = []
    records = []
    with torch.no_grad():
        for batch in loader:
            metadata = batch.pop("metadata")
            tensor_batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**tensor_batch)
            prediction, start, end = predict_span(
                outputs["start_logits"][0],
                outputs["end_logits"][0],
                batch["global_positions"][0],
                metadata[0]["flat_tokens"],
            )
            reference = metadata[0]["answer"]
            predictions.append(prediction)
            references.append(reference)
            records.append(
                {
                    "query_id": metadata[0]["query_id"],
                    "prediction": prediction,
                    "reference": reference,
                    "predicted_start": start,
                    "predicted_end": end,
                }
            )

    metrics = compute_basic_metrics(predictions, references)
    if not args.skip_bertscore:
        metrics["bertscore_f1"] = compute_bertscore(
            predictions, references, args.bertscore_model
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".predictions.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    args.output.write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "method": args.method,
                "encoder": encoder,
                "encoder_revision": run_args.get("encoder_revision"),
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
