"""BM25, dense, and reciprocal-rank-fusion search primitives."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.functional import normalize

from .data import read_jsonl


TOKEN_RE = re.compile(r"\w+", re.UNICODE)
VI_STOPWORDS = {
    "ai", "bao", "bằng", "bị", "bởi", "cả", "các", "cái", "cần", "cho", "chứ",
    "chưa", "có", "của", "cùng", "cũng", "đã", "đang", "đây", "để", "đến",
    "đều", "được", "gì", "hay", "khi", "không", "là", "lại", "lên", "mà",
    "mỗi", "một", "nào", "này", "nếu", "ngay", "như", "những", "ở", "phải",
    "qua", "ra", "rằng", "rất", "sau", "sẽ", "theo", "thì", "trên", "trong",
    "trước", "từ", "từng", "và", "vẫn", "vào", "với", "vừa", "về", "do",
}


def lexical_terms(text: str) -> list[str]:
    terms = TOKEN_RE.findall(text.lower())
    content_terms = [term for term in terms if term not in VI_STOPWORDS or term.isdigit()]
    return content_terms or terms


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SQLiteBM25:
    """Disk-backed FTS5 BM25 index, avoiding a large in-memory token matrix."""

    def __init__(self, corpus_path: Path, index_dir: Path):
        self.corpus_path = corpus_path
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        digest = sha256_file(corpus_path)[:16]
        self.db_path = index_dir / f"bm25-{digest}.sqlite"
        self.connection = sqlite3.connect(self.db_path)
        self._ensure_index(digest)

    def _ensure_index(self, digest: str) -> None:
        try:
            self.connection.execute("CREATE VIRTUAL TABLE IF NOT EXISTS fts5_probe USING fts5(text)")
            self.connection.execute("DROP TABLE fts5_probe")
        except sqlite3.OperationalError as exc:
            raise RuntimeError("SQLite was built without FTS5; BM25 requires SQLite FTS5") from exc
        self.connection.execute("CREATE TABLE IF NOT EXISTS ir_meta (key TEXT PRIMARY KEY, value TEXT)")
        existing = self.connection.execute("SELECT value FROM ir_meta WHERE key='corpus_sha256'").fetchone()
        if existing and existing[0] == digest:
            self.connection.execute("CREATE VIRTUAL TABLE IF NOT EXISTS vocab USING fts5vocab(passages, 'row')")
            return
        self.connection.execute("DROP TABLE IF EXISTS passages")
        self.connection.execute(
            "CREATE VIRTUAL TABLE passages USING fts5(doc_id UNINDEXED, title, text, tokenize='unicode61')"
        )
        records = read_jsonl(self.corpus_path)
        batch = []
        for record in records:
            batch.append((record["doc_id"], str(record.get("title", "")), str(record.get("text", ""))))
            if len(batch) >= 500:
                self.connection.executemany("INSERT INTO passages(doc_id,title,text) VALUES(?,?,?)", batch)
                self.connection.commit()
                batch.clear()
        if batch:
            self.connection.executemany("INSERT INTO passages(doc_id,title,text) VALUES(?,?,?)", batch)
        self.connection.execute(
            "INSERT OR REPLACE INTO ir_meta(key,value) VALUES('corpus_sha256',?)", (digest,)
        )
        self.connection.execute("CREATE VIRTUAL TABLE IF NOT EXISTS vocab USING fts5vocab(passages, 'row')")
        self.connection.commit()

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        terms = list(dict.fromkeys(lexical_terms(query)))
        if not terms:
            return []
        frequency: dict[str, int] = {}
        for term in terms:
            folded = "".join(char for char in unicodedata.normalize("NFD", term)
                             if unicodedata.category(char) != "Mn")
            variants = (term, folded)
            for variant in variants:
                row = self.connection.execute("SELECT doc FROM vocab WHERE term=?", (variant,)).fetchone()
                if row:
                    frequency[term] = int(row[0])
                    break
        # Use the most discriminative content words. Removing very common legal
        # question terms prevents FTS5 from scoring almost the whole corpus for
        # every query while retaining normal OR/BM25 ranking semantics.
        selected = sorted(frequency, key=lambda term: (frequency[term], term))[:4]
        query_terms = selected or terms
        match = " OR ".join('"' + term.replace('"', '""') + '"' for term in query_terms)
        rows = self.connection.execute(
            "SELECT doc_id,bm25(passages,0.0,2.0,1.0) AS score "
            "FROM passages WHERE passages MATCH ? ORDER BY score ASC, doc_id ASC LIMIT ?",
            (match, top_k),
        ).fetchall()
        return [(row[0], -float(row[1])) for row in rows]

    def close(self) -> None:
        self.connection.close()


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    return normalize(pooled.float(), p=2, dim=-1)


@torch.inference_mode()
def encode_texts(model, tokenizer, texts: list[str], device: torch.device, batch_size: int,
                 max_length: int) -> np.ndarray:
    vectors = []
    model.eval()
    for offset in range(0, len(texts), batch_size):
        batch = tokenizer(
            texts[offset:offset + batch_size], padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        ).to(device)
        hidden = model(**batch).last_hidden_state
        vectors.append(mean_pool(hidden, batch["attention_mask"]).cpu().numpy().astype(np.float32))
    return np.concatenate(vectors, axis=0) if vectors else np.empty((0, model.config.hidden_size), dtype=np.float32)


def ensure_dense_index(
    corpus_path: Path,
    model_path: Path,
    index_dir: Path,
    batch_size: int = 16,
    max_length: int = 256,
    device_name: str | None = None,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    from transformers import AutoModel, AutoTokenizer

    records = read_jsonl(corpus_path)
    model_path = Path(model_path)
    index_dir.mkdir(parents=True, exist_ok=True)
    corpus_hash = sha256_file(corpus_path)
    weight_files = sorted(model_path.glob("*.safetensors")) + sorted(model_path.glob("pytorch_model*.bin"))
    weights_digest = hashlib.sha256()
    for weight_file in weight_files:
        weights_digest.update(weight_file.name.encode())
        weights_digest.update(sha256_file(weight_file).encode())
    if not weight_files:
        weights_digest.update(str(model_path.resolve()).encode())
    config_hash = hashlib.sha256(
        (str(model_path.resolve()) + corpus_hash + weights_digest.hexdigest() + str(max_length)).encode()
    ).hexdigest()[:16]
    vector_path = index_dir / f"vectors-{config_hash}.npy"
    ids_path = index_dir / f"ids-{config_hash}.json"
    expected_ids = [row["doc_id"] for row in records]
    if vector_path.is_file() and ids_path.is_file():
        with ids_path.open("r", encoding="utf-8") as stream:
            saved_ids = json.load(stream)
        if saved_ids == expected_ids:
            return records, np.load(vector_path, mmap_mode="r")

    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModel.from_pretrained(model_path).to(device)
    model.eval()
    partial_path = index_dir / f"vectors-{config_hash}.partial.npy"
    progress_path = index_dir / f"vectors-{config_hash}.progress.json"
    completed = 0
    if partial_path.is_file() and progress_path.is_file():
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            partial = np.load(partial_path, mmap_mode="r+")
            if (progress.get("config_hash") == config_hash
                    and partial.shape == (len(records), model.config.hidden_size)):
                completed = min(int(progress.get("completed", 0)), len(records))
            else:
                del partial
                partial_path.unlink(missing_ok=True)
                progress_path.unlink(missing_ok=True)
                partial = None
        except (OSError, ValueError, json.JSONDecodeError):
            partial = None
            partial_path.unlink(missing_ok=True)
            progress_path.unlink(missing_ok=True)
    else:
        partial = None
    if partial is None:
        partial = np.lib.format.open_memmap(
            partial_path, mode="w+", dtype=np.float16,
            shape=(len(records), model.config.hidden_size),
        )
        completed = 0
    for offset in range(completed, len(records), batch_size):
        batch_rows = records[offset:offset + batch_size]
        texts = [f"{row.get('title', '')} {row.get('text', '')}" for row in batch_rows]
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            hidden = model(**encoded).last_hidden_state
            pooled = mean_pool(hidden, encoded["attention_mask"]).cpu().numpy().astype(np.float16)
        partial[offset:offset + len(batch_rows)] = pooled
        completed = offset + len(batch_rows)
        if completed == len(records) or completed % max(batch_size * 50, 1) < batch_size:
            partial.flush()
            progress_path.write_text(json.dumps({"config_hash": config_hash, "completed": completed}), encoding="utf-8")
            print(f"dense corpus embeddings: {completed}/{len(records)}")
    partial.flush()
    del partial
    partial_path.replace(vector_path)
    ids_path.write_text(json.dumps(expected_ids, ensure_ascii=False), encoding="utf-8")
    progress_path.unlink(missing_ok=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return records, np.load(vector_path, mmap_mode="r")


def dense_search(query_vectors: np.ndarray, records: list[dict[str, Any]], vectors: np.ndarray,
                 top_k: int) -> list[list[tuple[str, float]]]:
    doc_ids = [record["doc_id"] for record in records]
    queries = np.asarray(query_vectors, dtype=np.float32)
    n_docs = len(doc_ids)
    k = min(top_k, n_docs)
    output: list[list[tuple[str, float]]] = [[] for _ in range(len(queries))]
    query_batch = 32
    doc_chunk = 16384
    for q_start in range(0, len(queries), query_batch):
        q = queries[q_start:q_start + query_batch].T
        count = q.shape[1]
        best_scores = np.full((k, count), -np.inf, dtype=np.float32)
        best_positions = np.full((k, count), -1, dtype=np.int64)
        for d_start in range(0, n_docs, doc_chunk):
            d_end = min(d_start + doc_chunk, n_docs)
            block = np.asarray(vectors[d_start:d_end], dtype=np.float32)
            block_scores = block @ q
            block_positions = np.broadcast_to(
                np.arange(d_start, d_end, dtype=np.int64)[:, None], block_scores.shape
            )
            combined_scores = np.concatenate((best_scores, block_scores), axis=0)
            combined_positions = np.concatenate((best_positions, block_positions), axis=0)
            selected = np.argpartition(combined_scores, -k, axis=0)[-k:, :]
            best_scores = np.take_along_axis(combined_scores, selected, axis=0)
            best_positions = np.take_along_axis(combined_positions, selected, axis=0)
        for local_q in range(count):
            order = np.argsort(best_scores[:, local_q])[::-1]
            output[q_start + local_q] = [
                (doc_ids[int(best_positions[index, local_q])], float(best_scores[index, local_q]))
                for index in order
            ]
    return output


def rrf_fuse(*ranked_lists: list[tuple[str, float]], k: int = 60) -> list[tuple[str, float]]:
    fused: dict[str, float] = {}
    for ranking in ranked_lists:
        for rank, (doc_id, _score) in enumerate(ranking, 1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))
