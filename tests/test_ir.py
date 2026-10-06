from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ir.data import retrieval_metrics, validate_queries_and_qrels
from ir.evaluate_qa import make_examples
from ir.search import SQLiteBM25, dense_search, rrf_fuse
from ir.retrieve import make_docid_prefix_constraint
from ir.train_dense import focused_maxsim, prepare_training_examples


class RetrievalMetricTests(unittest.TestCase):
    def test_metrics_capture_multi_positive_evidence(self):
        qrels = {"q1": {"a": 1, "b": 1}, "q2": {"c": 1}}
        rankings = {"q1": ["x", "a", "b"], "q2": ["c"]}
        result = retrieval_metrics(qrels, rankings, cutoffs=(1, 2, 3))
        self.assertAlmostEqual(result["recall@1"], 0.5)
        self.assertAlmostEqual(result["recall@2"], 0.75)
        self.assertAlmostEqual(result["complete_evidence@3"], 1.0)
        self.assertAlmostEqual(result["mrr@3"], 0.75)

    def test_query_and_qrels_must_match_corpus(self):
        queries = [{"query_id": "q1"}]
        validate_queries_and_qrels(queries, {"q1": {"d1": 1}}, {"d1"})
        with self.assertRaises(ValueError):
            validate_queries_and_qrels(queries, {"q1": {"missing": 1}}, {"d1"})


class SearchTests(unittest.TestCase):
    def test_rrf_is_deterministic_and_promotes_consensus(self):
        first = [("a", 1.0), ("b", 0.5)]
        second = [("b", 1.0), ("c", 0.5)]
        fused = rrf_fuse(first, second, k=10)
        self.assertEqual(fused[0][0], "b")

    def test_dense_search_returns_exact_cosine_order(self):
        rows = [{"doc_id": "a"}, {"doc_id": "b"}, {"doc_id": "c"}]
        vectors = np.asarray([[1, 0], [0, 1], [-1, 0]], dtype=np.float16)
        result = dense_search(np.asarray([[0.9, 0.1]], dtype=np.float32), rows, vectors, 2)
        self.assertEqual([doc for doc, _ in result[0]], ["a", "b"])

    def test_disk_bm25_indexes_and_ranks_documents(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            corpus = root / "corpus.jsonl"
            records = [
                {"doc_id": "a", "title": "", "text": "luật lao động nghỉ phép năm"},
                {"doc_id": "b", "title": "", "text": "luật đất đai quyền sử dụng đất"},
            ]
            corpus.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
                              encoding="utf-8")
            index = SQLiteBM25(corpus, root / "index")
            try:
                hits = index.search("nghỉ phép lao động", 2)
            finally:
                index.close()
            self.assertEqual(hits[0][0], "a")

    def test_gmr_prefix_constraint_only_emits_valid_document_ids(self):
        class TinyTokenizer:
            pad_token_id = 0
            eos_token_id = 1
            unk_token_id = 99

            def convert_tokens_to_ids(self, token):
                return 9 if token == "<DOC>" else self.unk_token_id

            def encode(self, text, add_special_tokens=False):
                return {"doc-a": [2, 3], "doc-b": [2, 4]}[text]

        constraint = make_docid_prefix_constraint(TinyTokenizer(), ["doc-a", "doc-b"], 0)
        self.assertEqual(constraint(0, np.asarray([0])), [2])
        self.assertEqual(set(constraint(0, np.asarray([0, 2]))), {3, 4})
        self.assertEqual(set(constraint(0, np.asarray([0, 2, 3]))), {1, 9})
        self.assertEqual(constraint(0, np.asarray([0, 2, 3, 9])), [2])

    def test_downstream_qa_uses_retrieved_contexts_and_original_answers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            track_dir = root / "track"
            qa_dir = root / "qa"
            track_dir.mkdir()
            qa_dir.mkdir()
            (track_dir / "test_queries.jsonl").write_text(
                json.dumps({"query_id": "test_1", "source_qa_id": "1", "query": "câu hỏi"},
                           ensure_ascii=False) + "\n", encoding="utf-8")
            corpus_path = root / "corpus.jsonl"
            corpus_path.write_text(
                json.dumps({"doc_id": "d1", "title": "Điều 1", "text": "nội dung"},
                           ensure_ascii=False) + "\n", encoding="utf-8")
            rankings_path = root / "rankings.jsonl"
            rankings_path.write_text(
                json.dumps({"query_id": "test_1", "results": [{"doc_id": "d1", "score": 1.0}]})
                + "\n", encoding="utf-8")
            (qa_dir / "test_multihop.json").write_text(
                json.dumps({"1": {"question": "câu hỏi", "answer": "đáp án"}}, ensure_ascii=False),
                encoding="utf-8")
            examples = make_examples("test", track_dir, corpus_path, rankings_path, qa_dir, top_k=5)
            self.assertEqual(examples[0]["answer"], "đáp án")
            self.assertEqual(examples[0]["candidate_contexts"][0]["doc_id"], "d1")

    def test_dense_variants_have_one_paired_training_row_per_query_and_rotate_hops(self):
        rows = [{"query_id": "q1", "query": "q1?", "positive_doc_ids": ["d1", "d2"]}]
        qrels = {"q1": {"d1": 1, "d2": 1}}
        corpus = {"d1": {"text": "Evidence first."}, "d2": {"text": "Evidence second."}}
        base = prepare_training_examples(rows, qrels, corpus, "dense", epoch=0)
        hop0 = prepare_training_examples(rows, qrels, corpus, "mdr", epoch=0)
        hop1 = prepare_training_examples(rows, qrels, corpus, "mdr", epoch=1)
        self.assertEqual((len(base), len(hop0), len(hop1)), (1, 1, 1))
        self.assertEqual(base[0]["positive_ids"], hop0[0]["positive_ids"])
        self.assertIn("Evidence so far", hop1[0]["query"])
        self.assertEqual(hop1[0]["positive_ids"], ["d2"])

    def test_focused_maxsim_scores_query_document_pairs(self):
        query = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        documents = torch.tensor([[[1.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 1.0]]])
        q_mask = torch.tensor([[1, 1]])
        d_mask = torch.tensor([[1, 1], [1, 1]])
        scores = focused_maxsim(query, q_mask, documents, d_mask)
        self.assertEqual(tuple(scores.shape), (1, 2))
        self.assertAlmostEqual(scores[0, 0].item(), scores[0, 1].item())


if __name__ == "__main__":
    unittest.main()
