"""Train the shared-encoder dense retrieval family and paper-inspired variants."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import re
from pathlib import Path

import torch
from torch.nn import functional as F

from .data import read_jsonl, read_qrels
from .search import mean_pool


TRAINABLE_METHODS = ("dense", "mdr", "m3", "baleen", "mopo")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=TRAINABLE_METHODS, required=True)
    parser.add_argument("--track", choices=("all_matched", "true_multihop"), default="true_multihop")
    parser.add_argument("--data-dir", type=Path, default=root / "dataset/IR/experiments")
    parser.add_argument("--corpus", type=Path, default=root / "dataset/IR/experiments/corpus.jsonl")
    parser.add_argument("--output-root", type=Path, default=root / "runs/ir")
    parser.add_argument("--encoder", default="xlm-roberta-base")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--mopo-weight", type=float, default=0.1)
    parser.add_argument("--mopo-momentum", type=float, default=0.999)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def concise_fact(text: str, question: str, max_chars: int = 500) -> str:
    sentences = [part.strip() for part in SENTENCE_RE.split(text) if part.strip()]
    if not sentences:
        return text[:max_chars]
    query_terms = set(re.findall(r"\w+", question.lower()))
    scored = []
    for index, sentence in enumerate(sentences):
        terms = set(re.findall(r"\w+", sentence.lower()))
        score = len(query_terms & terms) / max(1, len(query_terms))
        scored.append((score, -index, sentence))
    selected = [row[2] for row in sorted(scored, reverse=True)[:2]]
    return " ".join(selected)[:max_chars]


def prepare_training_examples(rows, qrels, corpus_by_id, method, epoch=0):
    examples = []
    for row in rows:
        query_id = str(row["query_id"])
        positives = [doc for doc, gain in qrels[query_id].items() if gain > 0]
        positives = [doc for doc in row.get("positive_doc_ids", positives) if doc in corpus_by_id]
        if not positives:
            continue
        hop = epoch % len(positives)
        if method == "dense":
            examples.append({
                "query": row["query"],
                "posterior_query": row["query"],
                "positive_ids": [positives[hop]],
            })
            continue
        history = [
            concise_fact(str(corpus_by_id[doc_id].get("text", "")), row["query"])
            for doc_id in positives[:hop]
        ]
        target_doc = positives[hop]
        query = row["query"]
        if history and method in {"mdr", "m3", "baleen", "mopo"}:
            query += "\nEvidence so far: " + " ".join(history)
        posterior = query + "\nPosterior evidence: " + concise_fact(
            str(corpus_by_id[target_doc].get("text", "")), row["query"]
        )
        examples.append({
            "query": query,
            "posterior_query": posterior,
            "positive_ids": [target_doc],
            "source_query_id": query_id,
            "hop": hop,
        })
    return examples


def encode(model, tokenizer, texts, device, max_length):
    batch = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    batch = {key: value.to(device) for key, value in batch.items()}
    hidden = model(**batch).last_hidden_state
    return batch, hidden, mean_pool(hidden, batch["attention_mask"])


def focused_maxsim(q_hidden, q_mask, d_hidden, d_mask):
    """FLIPR-style top-fraction token MaxSim, computed for an in-batch matrix."""
    q_hidden = F.normalize(q_hidden.float(), dim=-1)
    d_hidden = F.normalize(d_hidden.float(), dim=-1)
    q_mask = q_mask.bool()
    d_mask = d_mask.bool()
    batch_q, q_len, _ = q_hidden.shape
    batch_d, d_len, _ = d_hidden.shape
    scores = q_hidden.new_zeros((batch_q, batch_d))
    for i in range(batch_q):
        q_tokens = q_hidden[i, q_mask[i]]
        if q_tokens.shape[0] == 0:
            continue
        similarities = torch.einsum("qd,bld->qbl", q_tokens, d_hidden)
        similarities = similarities.masked_fill(~d_mask.unsqueeze(0), -1e4)
        per_token = similarities.max(dim=-1).values.transpose(0, 1)
        keep = max(1, math.ceil(per_token.shape[-1] * 0.4))
        scores[i] = per_token.topk(min(keep, per_token.shape[-1]), dim=-1).values.sum(dim=-1)
    return scores


def contrastive_loss(scores, labels, temperature):
    logits = scores / temperature
    log_denominator = torch.logsumexp(logits, dim=-1)
    log_positive = torch.logsumexp(logits.masked_fill(~labels, -1e4), dim=-1)
    return (log_denominator - log_positive).mean()


def checkpoint(model, teacher, optimizer, scheduler, epoch, next_batch, step, run_dir):
    path = run_dir / "checkpoints" / f"step-{step:08d}"
    path.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "teacher": teacher.state_dict() if teacher is not None else None,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "next_batch": next_batch,
        "step": step,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "python_rng": random.getstate(),
    }, path / "trainer_state.pt")
    (run_dir / "latest_checkpoint.txt").write_text(f"checkpoints/{path.name}\n", encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.gradient_accumulation < 1 or args.epochs < 1:
        raise ValueError("batch size, gradient accumulation and epochs must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

    split_dir = args.data_dir / args.track
    train_path = split_dir / "train_queries.jsonl"
    rows = read_jsonl(train_path)
    if args.max_train_examples:
        rows = rows[:args.max_train_examples]
    qrels = read_qrels(split_dir / "train_qrels.tsv")
    corpus = read_jsonl(args.corpus)
    corpus_by_id = {record["doc_id"]: record for record in corpus}
    if not rows:
        raise ValueError("No train examples remain after joining queries, qrels and corpus")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(args.encoder, use_fast=True)
    model = AutoModel.from_pretrained(args.encoder).to(device)
    teacher = copy.deepcopy(model).to(device).eval() if args.method == "mopo" else None
    if teacher is not None:
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

    run_dir = args.output_root / "models" / f"{args.track}_{args.method}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "COMPLETE").is_file():
        raise FileExistsError(f"Completed run exists at {run_dir}; select a new output root or remove it manually")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = args.max_steps or math.ceil(len(rows) / args.batch_size / args.gradient_accumulation) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, int(total_steps * 0.06)), total_steps)
    start_epoch, start_batch, global_step = 0, 0, 0
    pointer = run_dir / "latest_checkpoint.txt"
    if pointer.is_file():
        candidate = run_dir / pointer.read_text(encoding="utf-8").strip()
        state_path = candidate / "trainer_state.pt"
        if state_path.is_file():
            try:
                state = torch.load(state_path, map_location="cpu", weights_only=False)
            except TypeError:
                state = torch.load(state_path, map_location="cpu")
            model.load_state_dict(state["model"])
            if teacher is not None and state.get("teacher") is not None:
                teacher.load_state_dict(state["teacher"])
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
            print(f"resuming from {candidate} at step {global_step}")

    train_meta = {
        "method": args.method,
        "track": args.track,
        "encoder": args.encoder,
        "seed": args.seed,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "max_train_examples": args.max_train_examples,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "temperature": args.temperature,
        "mopo_weight": args.mopo_weight,
        "mopo_momentum": args.mopo_momentum,
        "train_queries_sha256": file_hash(train_path),
        "train_qrels_sha256": file_hash(split_dir / "train_qrels.tsv"),
        "corpus_sha256": file_hash(args.corpus),
        "num_train_queries": len(rows),
        "resolved_optimizer_steps": total_steps,
        "paper_adaptation": {
            "dense": "shared-encoder in-batch multi-positive contrastive retriever",
            "mdr": "MDR-style teacher-forced evidence concatenation and greedy iterative inference",
            "m3": "M3-inspired contrastive plus binary relevance mixed objective; no FEVER NLI labels are available",
            "baleen": "Baleen/FLIPR-style focused token MaxSim and extractive query-focused condensation",
            "mopo": "MoPo-style EMA posterior encoder with KL regularization; posterior uses gold evidence only during training",
        }[args.method],
    }
    (run_dir / "run_config.json").write_text(json.dumps(train_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    step_in_epoch = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, args.epochs):
        examples = prepare_training_examples(rows, qrels, corpus_by_id, args.method, epoch)
        generator = torch.Generator().manual_seed(args.seed + epoch)
        permutation = torch.randperm(len(examples), generator=generator).tolist()
        ordered = [examples[index] for index in permutation]
        batches = [ordered[i:i + args.batch_size] for i in range(0, len(ordered), args.batch_size)]
        begin = start_batch if epoch == start_epoch else 0
        for batch_index in range(begin, len(batches)):
            batch_rows = batches[batch_index]
            candidate_ids = list(dict.fromkeys(doc_id for row in batch_rows for doc_id in row["positive_ids"]))
            doc_text = [f"{corpus_by_id[doc_id].get('title', '')} {corpus_by_id[doc_id].get('text', '')}" for doc_id in candidate_ids]
            query_text = [row["query"] for row in batch_rows]
            q_batch, q_hidden, q_vectors = encode(model, tokenizer, query_text, device, args.max_length)
            d_batch, d_hidden, d_vectors = encode(model, tokenizer, doc_text, device, args.max_length)
            labels = torch.tensor(
                [[doc_id in row["positive_ids"] for doc_id in candidate_ids] for row in batch_rows],
                dtype=torch.bool, device=device,
            )
            if args.method == "baleen":
                scores = focused_maxsim(q_hidden, q_batch["attention_mask"], d_hidden, d_batch["attention_mask"])
            else:
                scores = q_vectors @ d_vectors.T
            loss = contrastive_loss(scores, labels, args.temperature)

            if args.method == "m3":
                relevance_logits = scores / args.temperature
                loss = loss + 0.25 * F.binary_cross_entropy_with_logits(relevance_logits, labels.float())
            elif args.method == "mopo":
                posterior_queries = [row["posterior_query"] for row in batch_rows]
                with torch.no_grad():
                    _p_batch, _p_hidden, posterior_vectors = encode(
                        teacher, tokenizer, posterior_queries, device, args.max_length
                    )
                    posterior_scores = posterior_vectors @ d_vectors.T
                    teacher_probs = torch.softmax(posterior_scores / args.temperature, dim=-1)
                prior_log_probs = torch.log_softmax(scores / args.temperature, dim=-1)
                kl = F.kl_div(prior_log_probs, teacher_probs, reduction="batchmean")
                loss = loss + args.mopo_weight * kl

            (loss / args.gradient_accumulation).backward()
            step_in_epoch += 1
            is_update = step_in_epoch % args.gradient_accumulation == 0 or batch_index + 1 == len(batches)
            if is_update:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if teacher is not None:
                    with torch.no_grad():
                        for teacher_param, model_param in zip(teacher.parameters(), model.parameters()):
                            teacher_param.mul_(args.mopo_momentum).add_(model_param, alpha=1.0 - args.mopo_momentum)
                if global_step % args.save_steps == 0:
                    checkpoint(model, teacher, optimizer, scheduler, epoch, batch_index + 1, global_step, run_dir)
                if global_step % 20 == 0:
                    print(f"method={args.method} epoch={epoch + 1} step={global_step}/{total_steps} loss={loss.item():.4f}")
                if global_step >= total_steps:
                    break
        if global_step >= total_steps:
            break
        start_batch = 0
        step_in_epoch = 0

    model.save_pretrained(run_dir)
    tokenizer.save_pretrained(run_dir)
    if teacher is not None:
        torch.save(teacher.state_dict(), run_dir / "posterior_ema.pt")
    (run_dir / "COMPLETE").write_text(f"steps={global_step}\n", encoding="utf-8")
    print(f"completed: {run_dir} (steps={global_step})")


if __name__ == "__main__":
    main()
