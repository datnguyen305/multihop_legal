#!/usr/bin/env python3
"""Train a paper-inspired extractive QA model with weak span labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .data import read_jsonl
from .extractive_data import ExtractiveCollator
from .extractive_models import build_extractive_model, MODEL_CLASSES


class ExtractiveDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class EpochRandomSampler(Sampler[int]):
    """Same seeded shuffle per epoch across methods; safe to reconstruct on resume."""

    def __init__(self, data_source, seed: int):
        self.data_source = data_source
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(len(self.data_source), generator=generator).tolist())

    def __len__(self):
        return len(self.data_source)


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=sorted(MODEL_CLASSES), required=True)
    parser.add_argument("--encoder", default="xlm-roberta-base")
    parser.add_argument("--track", choices=("all_matched", "true_multihop"), default="true_multihop")
    parser.add_argument("--data-dir", type=Path, default=root / "dataset/QA/extractive")
    parser.add_argument("--output-dir", type=Path, default=root / "runs/extractive")
    parser.add_argument("--max-passages", type=int, default=6)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--passage-loss-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(model, tokenizer, optimizer, scheduler, run_dir, epoch, next_batch, step):
    checkpoint_dir = run_dir / "checkpoints" / f"step-{step:08d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "next_batch": next_batch,
            "step": step,
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        checkpoint_dir / "trainer_state.pt",
    )
    tokenizer.save_pretrained(checkpoint_dir)
    (run_dir / "latest_checkpoint.txt").write_text(
        f"checkpoints/{checkpoint_dir.name}\n", encoding="utf-8"
    )
    return checkpoint_dir


def load_state(path: Path):
    try:
        return torch.load(path / "trainer_state.pt", map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path / "trainer_state.pt", map_location="cpu")


def move_optimizer_state(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def compute_loss(outputs, batch, passage_loss_weight):
    start = outputs["start_logits"].flatten(1)
    end = outputs["end_logits"].flatten(1)
    span_loss = (
        torch.nn.functional.cross_entropy(start, batch["start_positions"])
        + torch.nn.functional.cross_entropy(end, batch["end_positions"])
    ) / 2.0
    passage_logits = outputs["passage_logits"]
    passage_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        passage_logits[batch["passage_mask"]],
        batch["passage_labels"][batch["passage_mask"]],
    )
    return span_loss + passage_loss_weight * passage_loss, span_loss, passage_loss


def main():
    args = parse_args()
    if args.save_steps < 1:
        raise ValueError("--save-steps must be at least 1")
    if args.batch_size < 1 or args.gradient_accumulation < 1:
        raise ValueError("--batch-size and --gradient-accumulation must be positive")
    if args.epochs < 1 or args.max_passages < 1 or args.max_length < 2:
        raise ValueError("--epochs, --max-passages and --max-length must be positive")
    set_seed(args.seed)
    from transformers import AutoTokenizer

    train_path = args.data_dir / args.track / "train.jsonl"
    dev_path = args.data_dir / args.track / "dev.jsonl"
    test_path = args.data_dir / args.track / "test.jsonl"
    manifest_path = args.data_dir / "manifest.json"
    missing = [
        path for path in (train_path, dev_path, test_path, manifest_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Extractive data is incomplete; run `python -m qa.prepare_extractive` first. "
            f"Missing: {missing}"
        )
    examples = read_jsonl(train_path)
    if args.max_train_examples:
        examples = examples[: args.max_train_examples]
    dataset = ExtractiveDataset(examples)
    sampler = EpochRandomSampler(dataset, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.encoder, use_fast=True)
    collator = ExtractiveCollator(tokenizer, args.max_passages, args.max_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collator,
    )

    run_dir = args.output_dir / f"{args.track}_{args.method}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    resume_path = args.resume_from_checkpoint
    if (run_dir / "COMPLETE").is_file() and resume_path is None:
        raise FileExistsError(
            f"Completed experiment already exists at {run_dir}; use a new --output-dir "
            "for changed settings, or pass an explicit checkpoint to resume."
        )
    if resume_path is None and not (run_dir / "COMPLETE").is_file():
        pointer = run_dir / "latest_checkpoint.txt"
        if pointer.is_file():
            candidate = run_dir / pointer.read_text(encoding="utf-8").strip()
            if (candidate / "trainer_state.pt").is_file():
                resume_path = candidate
                print(f"resuming from {resume_path}")

    model = build_extractive_model(args.method, args.encoder)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = args.max_steps or math.ceil(len(loader) / args.gradient_accumulation) * args.epochs
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=max(1, min(500, total_steps))
    )
    state = None
    if resume_path is not None:
        state = load_state(resume_path)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        move_optimizer_state(optimizer, device)
        scheduler.load_state_dict(state["scheduler"])
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])

    run_args = vars(args).copy()
    run_args["data_sha256"] = {
        "train": sha256_file(train_path),
        "dev": sha256_file(dev_path),
        "test": sha256_file(test_path),
        "manifest": sha256_file(manifest_path),
    }
    code_root = Path(__file__).parent
    code_files = (
        "extractive_data.py", "extractive_models.py", "prepare_extractive.py",
        "train_extractive.py", "evaluate_extractive.py", "metrics.py",
    )
    run_args["code_sha256"] = {
        name: sha256_file(code_root / name) for name in code_files
    }
    run_args["batch_order"] = "torch.randperm seeded with seed + epoch"
    run_args["encoder_revision"] = getattr(model.encoder.config, "_commit_hash", None)
    run_args["scheduled_optimizer_steps"] = total_steps
    run_args["train_micro_batches_per_epoch"] = len(loader)
    run_args["train_examples"] = len(examples)
    run_args["optimizer"] = type(optimizer).__name__
    run_args["weight_decay"] = optimizer.defaults["weight_decay"]
    run_args["gradient_clip_norm"] = 1.0
    run_args["scheduler"] = "LinearLR(start_factor=0.1, total_iters=min(500, total_steps))"
    run_args["precision"] = "float32"
    run_args["trainable_parameters"] = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if resume_path is not None:
        previous_args_path = resume_path.parents[1] / "run_args.json"
        if previous_args_path.is_file():
            previous_args = json.loads(previous_args_path.read_text(encoding="utf-8"))
            if "code_sha256" not in previous_args:
                raise ValueError(
                    "Checkpoint metadata lacks code hashes and cannot be resumed "
                    "under a verified fair comparison; use a new OUTPUT_ROOT."
                )
            invariant_keys = (
                "encoder", "encoder_revision", "track", "data_dir", "output_dir", "max_passages",
                "max_length", "batch_size", "gradient_accumulation", "epochs",
                "max_steps", "learning_rate", "passage_loss_weight", "seed",
                "max_train_examples", "save_steps", "data_sha256", "code_sha256",
                "batch_order", "optimizer", "weight_decay", "gradient_clip_norm",
                "scheduler", "precision",
            )
            changed = {
                key: (previous_args.get(key), run_args.get(key))
                for key in invariant_keys
                if key in previous_args and previous_args.get(key) != run_args.get(key)
            }
            if previous_args.get("method", args.method) != args.method:
                changed["method"] = (previous_args.get("method"), args.method)
            if changed:
                raise ValueError(
                    "Resume settings/data differ from the checkpoint; start a new run. "
                    f"Changed values: {changed}"
                )
    (run_dir / "run_args.json").write_text(
        json.dumps(run_args, default=str, indent=2) + "\n", encoding="utf-8"
    )
    start_epoch = int(state["epoch"]) if state else 0
    start_batch = int(state["next_batch"]) if state else 0
    step = int(state["step"]) if state else 0
    if args.max_steps and step >= args.max_steps:
        (run_dir / "COMPLETE").write_text("completed\n", encoding="utf-8")
        return

    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        for batch_index, batch in enumerate(loader):
            if epoch == start_epoch and batch_index < start_batch:
                continue
            tensor_batch = {
                key: value.to(device)
                for key, value in batch.items()
                if torch.is_tensor(value)
            }
            outputs = model(**tensor_batch)
            loss, span_loss, passage_loss = compute_loss(
                outputs, tensor_batch, args.passage_loss_weight
            )
            group_start = (batch_index // args.gradient_accumulation) * args.gradient_accumulation
            group_size = min(args.gradient_accumulation, len(loader) - group_start)
            (loss / group_size).backward()
            should_update = (
                (batch_index + 1) % args.gradient_accumulation == 0
                or batch_index + 1 == len(loader)
            )
            if should_update:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step == 1 or step % 25 == 0:
                    print(
                        f"epoch={epoch + 1} step={step} "
                        f"loss={loss.item():.4f} span={span_loss.item():.4f} "
                        f"passage={passage_loss.item():.4f}"
                    )
                if step % args.save_steps == 0:
                    checkpoint = save_checkpoint(
                        model, tokenizer, optimizer, scheduler, run_dir,
                        epoch, batch_index + 1, step
                    )
                    print(f"saved checkpoint to {checkpoint}")
                if args.max_steps and step >= args.max_steps:
                    break
        if args.max_steps and step >= args.max_steps:
            break

    tokenizer.save_pretrained(run_dir)
    torch.save(model.state_dict(), run_dir / "model_state.pt")
    (run_dir / "COMPLETE").write_text("completed\n", encoding="utf-8")
    print(f"saved extractive checkpoint to {run_dir}; train_steps={step}")


if __name__ == "__main__":
    main()
