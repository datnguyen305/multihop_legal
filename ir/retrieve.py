"""Run one retrieval method against the same corpus and qrels for a split."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .data import read_jsonl, read_qrels, retrieval_metrics, write_jsonl
from .search import SQLiteBM25, dense_search, encode_texts, ensure_dense_index, rrf_fuse, sha256_file
from .train_dense import concise_fact, focused_maxsim


METHODS = ("bm25", "dense", "hybrid", "mdr", "m3", "baleen", "mopo", "gmr", "ircot")
DOC_SEPARATOR = "<DOC>"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--split", choices=("dev", "test", "train"), default="dev")
    parser.add_argument("--track", choices=("all_matched", "true_multihop"), default="true_multihop")
    parser.add_argument("--data-dir", type=Path, default=root / "dataset/IR/experiments")
    parser.add_argument("--corpus", type=Path, default=root / "dataset/IR/experiments/corpus.jsonl")
    parser.add_argument("--output-root", type=Path, default=root / "runs/ir")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--encoder", default="xlm-roberta-base")
    parser.add_argument("--reasoner-model", default="google/flan-t5-small")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--search-depth", type=int, default=100)
    parser.add_argument("--max-hops", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def encoder_queries(model, tokenizer, queries, device, batch_size, max_length):
    return encode_texts(model, tokenizer, queries, device, batch_size, max_length)


def late_interaction_rerank(query, candidate_rows, model, tokenizer, device, max_length):
    if not candidate_rows:
        return []
    d_texts = [f"{row.get('title', '')} {row.get('text', '')}" for row in candidate_rows]
    scored = []
    model.eval()
    with torch.inference_mode():
        q_inputs = tokenizer([query], padding=True, truncation=True, max_length=max_length,
                             return_tensors="pt").to(device)
        q_hidden = model(**q_inputs).last_hidden_state
        for start in range(0, len(d_texts), 16):
            d_inputs = tokenizer(d_texts[start:start + 16], padding=True, truncation=True,
                                 max_length=max_length, return_tensors="pt").to(device)
            d_hidden = model(**d_inputs).last_hidden_state
            values = focused_maxsim(
                q_hidden, q_inputs["attention_mask"], d_hidden, d_inputs["attention_mask"]
            )[0].cpu().tolist()
            scored.extend((candidate_rows[start + offset]["doc_id"], float(value))
                          for offset, value in enumerate(values))
    return sorted(scored, key=lambda item: (-item[1], item[0]))


def query_facts(query: str, text: str) -> str:
    return concise_fact(text, query, max_chars=500)


def make_docid_prefix_constraint(tokenizer, doc_ids, decoder_start_id=None):
    """Build a token trie constraining GMR decoding to IDs in this corpus."""
    terminal = -1
    root = {}
    for doc_id in sorted(doc_ids):
        token_ids = tokenizer.encode(doc_id, add_special_tokens=False)
        if not token_ids:
            continue
        node = root
        for token_id in token_ids:
            node = node.setdefault(int(token_id), {})
        node[terminal] = True
    separator_id = tokenizer.convert_tokens_to_ids(DOC_SEPARATOR)
    eos_id = tokenizer.eos_token_id
    start_id = decoder_start_id if decoder_start_id is not None else tokenizer.pad_token_id

    def allowed(_batch_id, decoder_input_ids):
        generated = decoder_input_ids.tolist()
        if generated and isinstance(generated[0], list):
            generated = generated[0]
        if generated and generated[0] == start_id:
            generated = generated[1:]
        if separator_id in generated:
            last_separator = len(generated) - 1 - generated[::-1].index(separator_id)
            current = generated[last_separator + 1:]
        else:
            current = generated
        node = root
        for token_id in current:
            child = node.get(int(token_id))
            if child is None or token_id == terminal:
                return [eos_id]
            node = child
        choices = [key for key in node if key != terminal]
        if terminal in node:
            if separator_id is not None and separator_id not in choices:
                choices.append(separator_id)
            if eos_id is not None and eos_id not in choices:
                choices.append(eos_id)
        return choices or ([eos_id] if eos_id is not None else [])

    return allowed


def run_gmr(rows, corpus_ids, model_path, device, max_target_length=256):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    separator_id = tokenizer.convert_tokens_to_ids(DOC_SEPARATOR)
    if separator_id == tokenizer.unk_token_id:
        raise ValueError(f"The GMR checkpoint tokenizer does not contain {DOC_SEPARATOR}")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device).eval()
    constraint = make_docid_prefix_constraint(
        tokenizer, corpus_ids, model.config.decoder_start_token_id
    )
    normalized_ids = {re.sub(r"\s+", "", doc_id).casefold(): doc_id for doc_id in corpus_ids}
    output = {}
    for number, row in enumerate(rows, 1):
        encoded = tokenizer(str(row["query"]), return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded, do_sample=False, num_beams=1, max_new_tokens=max_target_length,
                prefix_allowed_tokens_fn=constraint,
            )
        generated_ids = generated[0].tolist()
        if generated_ids and generated_ids[0] == model.config.decoder_start_token_id:
            generated_ids = generated_ids[1:]
        segments = []
        current = []
        for token_id in generated_ids:
            if token_id == separator_id or token_id == tokenizer.eos_token_id:
                if current:
                    segments.append(tokenizer.decode(current, skip_special_tokens=True).strip())
                    current = []
                if token_id == tokenizer.eos_token_id:
                    break
            else:
                current.append(token_id)
        if current:
            segments.append(tokenizer.decode(current, skip_special_tokens=True).strip())
        docs = []
        for candidate in segments:
            resolved = normalized_ids.get(re.sub(r"\s+", "", candidate).casefold())
            if resolved and resolved not in docs:
                docs.append(resolved)
        output[str(row["query_id"])] = [(doc_id, float(len(docs) - rank)) for rank, doc_id in enumerate(docs)]
        if number % 100 == 0:
            print(f"GMR constrained generation: {number}/{len(rows)}")
    del model
    return output


def run_ircot(rows, corpus_by_id, bm25, reasoner_name, device, depth, hops):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(reasoner_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(reasoner_name).to(device).eval()
    output = {}
    for number, row in enumerate(rows, 1):
        question = str(row["query"])
        initial = bm25.search(question, depth)
        collected = [initial]
        context = initial[:4]
        used = set()
        reasoning = ""
        for _hop in range(max(1, hops - 1)):
            passages = []
            for doc_id, _score in context:
                if doc_id not in used and doc_id in corpus_by_id:
                    used.add(doc_id)
                    record = corpus_by_id[doc_id]
                    passages.append(f"{record.get('title', '')}: {query_facts(question, record.get('text', ''))}")
            prompt = (
                "Given the legal question and retrieved evidence, write only the next concise "
                "reasoning sentence needed to find another relevant legal passage. Do not answer yet.\n"
                f"Question: {question}\nEvidence:\n" + "\n".join(passages) +
                (f"\nReasoning so far: {reasoning}" if reasoning else "") + "\nNext sentence:"
            )
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768).to(device)
            with torch.inference_mode():
                generated = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=64)
            reasoning = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
            reasoning = reasoning.split("\n")[0]
            followup = bm25.search(f"{question} {reasoning}", depth)
            collected.append(followup)
            context = followup[:4]
        output[str(row["query_id"])] = rrf_fuse(*collected)[:depth]
        if number % 100 == 0:
            print(f"IRCoT retrieval: {number}/{len(rows)}")
    del model
    return output


def main() -> None:
    args = parse_args()
    experiment_started = time.perf_counter()
    if args.top_k < 1 or args.search_depth < args.top_k or args.max_hops < 1:
        raise ValueError("require top-k >=1, search-depth >= top-k, and max-hops >=1")
    split_dir = args.data_dir / args.track
    rows = read_jsonl(split_dir / f"{args.split}_queries.jsonl")
    qrels = read_qrels(split_dir / f"{args.split}_qrels.tsv")
    records = read_jsonl(args.corpus)
    corpus_by_id = {record["doc_id"]: record for record in records}
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rankings = {}
    model_used = None

    if args.method == "gmr":
        model_path = args.model or args.output_root / "models" / f"{args.track}_gmr_seed{args.seed}"
        if not Path(model_path).is_dir():
            raise FileNotFoundError(f"GMR model directory not found: {model_path}; train it first")
        rankings = run_gmr(rows, set(corpus_by_id), Path(model_path), device)
        model_used = str(model_path)
        bm25 = None
    else:
        bm25 = None

    if args.method in {"bm25", "hybrid", "ircot"}:
        bm25 = SQLiteBM25(args.corpus, args.output_root / "index")
    if args.method == "gmr":
        pass
    elif args.method == "bm25":
        for number, row in enumerate(rows, 1):
            rankings[str(row["query_id"])] = bm25.search(str(row["query"]), args.search_depth)
            if number % 100 == 0:
                print(f"BM25 retrieval: {number}/{len(rows)}", flush=True)
    elif args.method == "ircot":
        rankings = run_ircot(rows, corpus_by_id, bm25, args.reasoner_model, device,
                             args.search_depth, args.max_hops)
    elif args.method != "gmr":
        model_path = args.model or args.output_root / "models" / f"{args.track}_{args.method}_seed{args.seed}"
        if args.method in {"dense", "hybrid"}:
            model_path = args.model or args.output_root / "models" / f"{args.track}_dense_seed{args.seed}"
        if not Path(model_path).is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_path}; train it first")
        model_used = str(model_path)
        records, vectors = ensure_dense_index(
            args.corpus, Path(model_path), args.output_root / "index" / "dense",
            args.batch_size, args.max_length, str(device),
        )
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = AutoModel.from_pretrained(model_path).to(device).eval()
        questions = [str(row["query"]) for row in rows]
        single_vectors = encoder_queries(model, tokenizer, questions, device, args.batch_size, args.max_length)
        single_rankings = dense_search(single_vectors, records, vectors, args.search_depth)
        if args.method == "baleen":
            for i, (row, candidates) in enumerate(zip(rows, single_rankings)):
                candidates_to_rerank = [corpus_by_id[doc_id] for doc_id, _score in candidates[:20]
                                        if doc_id in corpus_by_id]
                late_ranked = late_interaction_rerank(
                    str(row["query"]), candidates_to_rerank, model, tokenizer, device, args.max_length
                )
                late_ids = {doc_id for doc_id, _score in late_ranked}
                single_rankings[i] = late_ranked + [pair for pair in candidates if pair[0] not in late_ids]
                if (i + 1) % 200 == 0:
                    print(f"Baleen FLIPR first-hop rerank: {i + 1}/{len(rows)}")
        if args.method in {"dense", "hybrid"}:
            for number, (row, single) in enumerate(zip(rows, single_rankings), 1):
                question = str(row["query"])
                ranked = single if args.method == "dense" else rrf_fuse(
                    single, bm25.search(question, args.search_depth)
                )[:args.search_depth]
                rankings[str(row["query_id"])] = ranked[:args.search_depth]
                if number % 200 == 0:
                    print(f"{args.method} retrieval: {number}/{len(rows)}")
        else:
            step_rankings = [[single] for single in single_rankings]
            histories: list[list[str]] = [[] for _ in rows]
            seen_docs: list[set[str]] = [set() for _ in rows]
            for i, (row, single) in enumerate(zip(rows, single_rankings)):
                first_doc = single[0][0] if single else None
                if first_doc and first_doc in corpus_by_id:
                    seen_docs[i].add(first_doc)
                    histories[i].append(query_facts(str(row["query"]), str(corpus_by_id[first_doc].get("text", ""))))
            for hop in range(1, args.max_hops):
                current_queries = [
                    str(row["query"]) + ("\nEvidence so far: " + " ".join(history[-3:]) if history else "")
                    for row, history in zip(rows, histories)
                ]
                hop_vectors = encoder_queries(model, tokenizer, current_queries, device,
                                              args.batch_size, args.max_length)
                hop_rankings = dense_search(hop_vectors, records, vectors, args.search_depth)
                for i, candidates in enumerate(hop_rankings):
                    if args.method == "baleen":
                        current = current_queries[i]
                        candidate_records = [corpus_by_id[doc_id] for doc_id, _score in candidates
                                             if doc_id in corpus_by_id][:20]
                        late_ranked = late_interaction_rerank(current, candidate_records, model,
                                                              tokenizer, device, args.max_length)
                        late_ids = {doc_id for doc_id, _score in late_ranked}
                        candidates = late_ranked + [pair for pair in candidates if pair[0] not in late_ids]
                    step_rankings[i].append(candidates)
                    next_doc = next((doc_id for doc_id, _score in candidates
                                     if doc_id not in seen_docs[i]), None)
                    if next_doc:
                        seen_docs[i].add(next_doc)
                        histories[i].append(query_facts(
                            str(rows[i]["query"]), str(corpus_by_id[next_doc].get("text", ""))
                        ))
                print(f"{args.method} retrieval hop {hop + 1}/{args.max_hops}")
            for i, row in enumerate(rows):
                if args.method == "m3":
                    multi = rrf_fuse(*step_rankings[i][1:]) if len(step_rankings[i]) > 1 else []
                    ranked = rrf_fuse(step_rankings[i][0], multi)[:args.search_depth]
                else:
                    ranked = rrf_fuse(*step_rankings[i])[:args.search_depth]
                rankings[str(row["query_id"])] = ranked
                if (i + 1) % 200 == 0:
                    print(f"{args.method} retrieval: {i + 1}/{len(rows)}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if bm25 is not None:
        bm25.close()
    if set(rankings) != set(qrels):
        missing = sorted(set(qrels) - set(rankings))[:5]
        extra = sorted(set(rankings) - set(qrels))[:5]
        raise ValueError(f"Retriever query IDs differ from qrels; missing={missing}, extra={extra}")

    run_dir = args.output_root / "runs" / f"{args.track}_{args.method}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = run_dir / f"{args.split}_rankings.jsonl"
    write_jsonl(ranking_path, [
        {"query_id": qid, "results": [{"doc_id": doc, "score": score} for doc, score in ranked]}
        for qid, ranked in rankings.items()
    ])
    # Only metrics at the requested output cutoffs are scored; the runner gives
    # every method the same maximum ranking depth.
    ranked_ids = {qid: [doc for doc, _score in ranking] for qid, ranking in rankings.items()}
    cutoffs = tuple(sorted({1, 5, 10, 20, 100, args.top_k}))
    metrics = retrieval_metrics(qrels, ranked_ids, cutoffs=cutoffs)
    metrics.update({"method": args.method, "split": args.split, "track": args.track,
                    "seed": args.seed, "top_k": args.top_k, "search_depth": args.search_depth,
                    "max_hops": args.max_hops, "corpus_path": str(args.corpus),
                    "corpus_sha256": sha256_file(args.corpus),
                    "qrels_sha256": sha256_file(split_dir / f"{args.split}_qrels.tsv"),
                    "queries_sha256": sha256_file(split_dir / f"{args.split}_queries.jsonl"),
                    "model": model_used,
                    "reasoner_model": args.reasoner_model if args.method == "ircot" else None,
                    "runtime_seconds": round(time.perf_counter() - experiment_started, 3),
                    "results_per_query_mean": round(
                        sum(len(ranking) for ranking in rankings.values()) / max(1, len(rankings)), 4
                    )})
    metrics_path = run_dir / f"{args.split}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
