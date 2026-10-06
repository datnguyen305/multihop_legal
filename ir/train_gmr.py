"""Train a GMR-inspired constrained document-ID generator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import torch

from .data import read_jsonl, read_qrels


DOC_SEPARATOR = "<DOC>"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=("all_matched", "true_multihop"), default="true_multihop")
    parser.add_argument("--data-dir", type=Path, default=root / "dataset/IR/experiments")
    parser.add_argument("--output-root", type=Path, default=root / "runs/ir")
    parser.add_argument("--model-name", default="google/mt5-small")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-source-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=256)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    split_dir = args.data_dir / args.track
    train_file = split_dir / "train_queries.jsonl"
    qrels_file = split_dir / "train_qrels.tsv"
    rows = read_jsonl(train_file)
    if args.max_train_examples:
        rows = rows[:args.max_train_examples]
    qrels = read_qrels(qrels_file)
    examples = []
    for row in rows:
        ids = [doc for doc, gain in qrels[str(row["query_id"])].items() if gain > 0]
        if ids:
            examples.append({"query": str(row["query"]), "target": f" {DOC_SEPARATOR} ".join(ids)})

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    added = tokenizer.add_special_tokens({"additional_special_tokens": [DOC_SEPARATOR]})
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    if added:
        model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    run_dir = args.output_root / "models" / f"{args.track}_gmr_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "COMPLETE").is_file():
        raise FileExistsError(f"Completed GMR run exists at {run_dir}; use a new output root")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    steps_per_epoch = math.ceil(len(examples) / args.batch_size / args.gradient_accumulation)
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, int(total_steps * 0.06)), total_steps)
    start_epoch, start_batch, global_step = 0, 0, 0
    pointer = run_dir / "latest_checkpoint.txt"
    if pointer.is_file():
        state_path = run_dir / pointer.read_text(encoding="utf-8").strip() / "trainer_state.pt"
        if state_path.is_file():
            try:
                state = torch.load(state_path, map_location="cpu", weights_only=False)
            except TypeError:
                state = torch.load(state_path, map_location="cpu")
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            if "scheduler" in state:
                scheduler.load_state_dict(state["scheduler"])
            for opt_state in optimizer.state.values():
                for key, value in opt_state.items():
                    if torch.is_tensor(value):
                        opt_state[key] = value.to(device)
            start_epoch, start_batch, global_step = state["epoch"], state["next_batch"], state["step"]
            torch.set_rng_state(state["torch_rng"])
            random.setstate(state["python_rng"])
            if device.type == "cuda" and state.get("cuda_rng"):
                torch.cuda.set_rng_state_all(state["cuda_rng"])
            print(f"resuming GMR from step {global_step}")

    config = {
        "method": "gmr",
        "track": args.track,
        "model_name": args.model_name,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "max_train_examples": args.max_train_examples,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "max_source_length": args.max_source_length,
        "max_target_length": args.max_target_length,
        "seed": args.seed,
        "train_queries_sha256": file_hash(train_file),
        "train_qrels_sha256": file_hash(qrels_file),
        "adaptation": "GMR-style generative retrieval, emitting a constrained sequence of corpus doc_ids rather than full passage text",
    }
    (run_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    for epoch in range(start_epoch, args.epochs):
        order = torch.randperm(len(examples), generator=torch.Generator().manual_seed(args.seed + epoch)).tolist()
        shuffled = [examples[index] for index in order]
        batches = [shuffled[i:i + args.batch_size] for i in range(0, len(shuffled), args.batch_size)]
        begin = start_batch if epoch == start_epoch else 0
        for batch_index in range(begin, len(batches)):
            batch = batches[batch_index]
            model_inputs = tokenizer([item["query"] for item in batch], padding=True, truncation=True,
                                     max_length=args.max_source_length, return_tensors="pt").to(device)
            targets = tokenizer(text_target=[item["target"] for item in batch], padding=True, truncation=True,
                                max_length=args.max_target_length, return_tensors="pt").input_ids.to(device)
            targets[targets == tokenizer.pad_token_id] = -100
            loss = model(**model_inputs, labels=targets).loss
            (loss / args.gradient_accumulation).backward()
            micro_step += 1
            update = micro_step % args.gradient_accumulation == 0 or batch_index + 1 == len(batches)
            if update:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % args.save_steps == 0:
                    path = run_dir / "checkpoints" / f"step-{global_step:08d}"
                    path.mkdir(parents=True, exist_ok=True)
                    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(),
                                "epoch": epoch, "next_batch": batch_index + 1, "step": global_step,
                                "torch_rng": torch.get_rng_state(),
                                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                                "python_rng": random.getstate()}, path / "trainer_state.pt")
                    (run_dir / "latest_checkpoint.txt").write_text(f"checkpoints/{path.name}\n", encoding="utf-8")
                if global_step % 20 == 0:
                    print(f"GMR epoch={epoch + 1} step={global_step}/{total_steps} loss={loss.item():.4f}")
                if args.max_steps and global_step >= args.max_steps:
                    break
        if args.max_steps and global_step >= args.max_steps:
            break
        start_batch = 0
        micro_step = 0

    model.save_pretrained(run_dir)
    tokenizer.save_pretrained(run_dir)
    (run_dir / "COMPLETE").write_text(f"steps={global_step}\n", encoding="utf-8")
    print(f"completed GMR: {run_dir} (steps={global_step})")


if __name__ == "__main__":
    main()
