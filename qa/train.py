#!/usr/bin/env python3
"""Train an adapted multilingual abstractive QA model."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from .data import METHODS, build_source, build_target, load_prepared_examples
from .modeling import configure_tokenizer


class QADataset(Dataset):
    def __init__(self, examples, method, tokenizer, max_source_length, max_target_length):
        self.examples = examples
        self.method = method
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        source = build_source(example, self.method)
        target = build_target(example)
        encoded = self.tokenizer(
            source,
            max_length=self.max_source_length,
            truncation=True,
            padding=False,
        )
        target_encoded = self.tokenizer(
            text_target=target,
            max_length=self.max_target_length,
            truncation=True,
            padding=False,
        )
        encoded["labels"] = target_encoded["input_ids"]
        return encoded


def collate(tokenizer):
    def _collate(batch):
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]
        result = tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            return_tensors="pt",
        )
        label_batch = tokenizer.pad({"input_ids": labels}, return_tensors="pt")[
            "input_ids"
        ]
        label_batch[label_batch == tokenizer.pad_token_id] = -100
        result["labels"] = label_batch
        return result

    return _collate


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--track", choices=("all_matched", "true_multihop"), default="true_multihop")
    parser.add_argument("--model-name", default="google/mt5-base")
    parser.add_argument("--data-dir", type=Path, default=root / "dataset" / "QA" / "abstractive")
    parser.add_argument("--corpus-dir", type=Path, default=root / "dataset" / "IR" / "structured")
    parser.add_argument("--output-dir", type=Path, default=root / "runs" / "qa")
    parser.add_argument("--max-source-length", type=int, default=2048)
    parser.add_argument("--max-target-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument(
        "--save-steps",
        type=int,
        default=50,
        help="Save a resumable checkpoint every N optimizer steps.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=None,
        help="Resume from a checkpoint directory created by this trainer.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_checkpoint(
    model,
    tokenizer,
    optimizer,
    scheduler,
    run_dir: Path,
    epoch: int,
    next_batch_index: int,
    step: int,
    device: torch.device,
) -> Path:
    checkpoint_dir = run_dir / "checkpoints" / f"step-{step:08d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    state = {
        "epoch": epoch,
        "next_batch_index": next_batch_index,
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["cuda_rng"] = torch.cuda.get_rng_state_all()
    torch.save(state, checkpoint_dir / "trainer_state.pt")
    (run_dir / "latest_checkpoint.txt").write_text(
        f"checkpoints/{checkpoint_dir.name}\n", encoding="utf-8"
    )
    return checkpoint_dir


def _load_checkpoint_state(path: Path):
    state_path = path / "trainer_state.pt"
    if not state_path.is_file():
        raise FileNotFoundError(f"Missing trainer state: {state_path}")
    try:
        return torch.load(state_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(state_path, map_location="cpu")


def _move_optimizer_state_to_device(optimizer, device: torch.device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def main():
    args = parse_args()
    if args.save_steps < 1:
        raise ValueError("--save-steps must be at least 1")
    set_seed(args.seed)
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install requirements-qa.txt before training") from exc

    train_path = args.data_dir / args.track / "train.jsonl"
    dev_path = args.data_dir / args.track / "dev.jsonl"
    if not train_path.is_file() or not dev_path.is_file():
        raise FileNotFoundError(
            "Prepared QA data not found. Run `python -m qa.prepare_abstractive` first."
        )

    run_dir = args.output_dir / f"{args.track}_{args.method}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    resume_path = args.resume_from_checkpoint
    if resume_path is None and not (run_dir / "COMPLETE").is_file():
        latest_pointer = run_dir / "latest_checkpoint.txt"
        if latest_pointer.is_file():
            candidate = run_dir / latest_pointer.read_text(encoding="utf-8").strip()
            if (candidate / "trainer_state.pt").is_file():
                resume_path = candidate
                print(f"resuming automatically from {resume_path}")

    model_source = resume_path if resume_path is not None else args.model_name
    if resume_path is not None and not resume_path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {resume_path}")
    tokenizer = configure_tokenizer(
        AutoTokenizer.from_pretrained(model_source), args.model_name
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_source)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if device.type == "cpu":
        print("WARNING: no CUDA device detected; use --max-steps for a smoke test.")

    train_examples = load_prepared_examples(
        train_path, args.corpus_dir / "train_corpus.jsonl"
    )
    dev_examples = load_prepared_examples(
        dev_path, args.corpus_dir / "dev_corpus.jsonl"
    )
    if args.max_train_examples:
        train_examples = train_examples[: args.max_train_examples]
    train_dataset = QADataset(
        train_examples, args.method, tokenizer, args.max_source_length, args.max_target_length
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        # Stable ordering makes skipping already-processed batches reliable
        # after a restart from a checkpoint.
        shuffle=False,
        collate_fn=collate(tokenizer),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = args.max_steps or math.ceil(len(train_loader) / args.gradient_accumulation) * args.epochs
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=max(1, min(500, total_steps)))

    (run_dir / "run_args.json").write_text(
        json.dumps(vars(args), default=str, indent=2) + "\n", encoding="utf-8"
    )
    start_epoch = 0
    start_batch_index = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)
    step = 0
    if resume_path is not None:
        state = _load_checkpoint_state(resume_path)
        optimizer.load_state_dict(state["optimizer"])
        _move_optimizer_state_to_device(optimizer, device)
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"])
        start_batch_index = int(state["next_batch_index"])
        step = int(state["step"])
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and "cuda_rng" in state:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        print(
            f"loaded checkpoint: epoch={start_epoch} "
            f"next_batch={start_batch_index} step={step}"
        )

    if args.max_steps and step >= args.max_steps:
        print(f"max_steps={args.max_steps} already reached; nothing to train")
        model.save_pretrained(run_dir)
        tokenizer.save_pretrained(run_dir)
        (run_dir / "COMPLETE").write_text("completed\n", encoding="utf-8")
        return

    for epoch in range(start_epoch, args.epochs):
        for batch_index, batch in enumerate(train_loader):
            if epoch == start_epoch and batch_index < start_batch_index:
                continue
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            loss = output.loss / args.gradient_accumulation
            loss.backward()
            if (batch_index + 1) % args.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step == 1 or step % 50 == 0:
                    print(f"epoch={epoch + 1} step={step} loss={loss.item() * args.gradient_accumulation:.4f}")
                if step % args.save_steps == 0:
                    checkpoint_dir = _save_checkpoint(
                        model,
                        tokenizer,
                        optimizer,
                        scheduler,
                        run_dir,
                        epoch,
                        batch_index + 1,
                        step,
                        device,
                    )
                    print(f"saved checkpoint to {checkpoint_dir}")
                if args.max_steps and step >= args.max_steps:
                    break
        if args.max_steps and step >= args.max_steps:
            break

    model.save_pretrained(run_dir)
    tokenizer.save_pretrained(run_dir)
    (run_dir / "COMPLETE").write_text("completed\n", encoding="utf-8")
    print(f"saved checkpoint to {run_dir}; train_steps={step}; dev_examples={len(dev_examples)}")


if __name__ == "__main__":
    main()
