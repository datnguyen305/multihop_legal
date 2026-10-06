"""Data preparation helpers for weakly supervised extractive QA."""

from __future__ import annotations

import re
from collections import Counter
from itertools import islice
from typing import Any

import torch


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
SENTENCE_RE = re.compile(r"(?<=[.!?;])\s+|\n+")


def regex_tokens(text: str) -> list[str]:
    return [match.group(0) for match in TOKEN_RE.finditer(text or "")]


def detokenize_tokens(tokens: list[str]) -> str:
    output = ""
    no_space_before = {",", ".", ";", ":", "!", "?", "%", ")", "]", "}", "»"}
    no_space_after = {"(", "[", "{", "«"}
    for token in tokens:
        if not output:
            output = token
        elif token in no_space_before:
            output += token
        elif output[-1:] in no_space_after:
            output += token
        else:
            output += " " + token
    return output


def _find_subsequence(haystack: list[str], needle: list[str]) -> tuple[int, int] | None:
    if not needle or len(needle) > len(haystack):
        return None
    last = len(haystack) - len(needle) + 1
    for start in range(last):
        if haystack[start : start + len(needle)] == needle:
            return start, start + len(needle) - 1
    return None


def _answer_fragments(answer: str) -> list[list[str]]:
    fragments: list[list[str]] = []
    for raw in SENTENCE_RE.split(answer):
        raw = re.sub(r"^\s*[-+•*]\s*", "", raw).strip()
        raw = re.sub(r"^\s*\(?\d+[.)]\s*", "", raw).strip()
        tokens = regex_tokens(raw)
        if len(tokens) >= 2:
            fragments.append(tokens)
    # Prefer the longest answer fragments, with deterministic de-duplication.
    unique: dict[tuple[str, ...], list[str]] = {}
    for fragment in fragments:
        unique.setdefault(tuple(fragment), fragment)
    return sorted(unique.values(), key=lambda value: (-len(value), value))


def best_overlap_span(
    answer_tokens: list[str], context_tokens: list[str]
) -> tuple[int, int, float]:
    """Find a deterministic high-overlap contiguous span.

    The search checks windows around the answer length and sentence-like
    punctuation boundaries. It is intentionally deterministic so every model
    receives identical weak labels.
    """
    if not answer_tokens or not context_tokens:
        return 0, max(0, min(len(context_tokens) - 1, 0)), 0.0
    answer_length = len(answer_tokens)
    max_length = min(len(context_tokens), max(1, min(1536, answer_length * 2)))
    lengths = {
        max(1, answer_length // 2),
        max(1, answer_length - 16),
        answer_length,
        min(max_length, answer_length + 16),
        max_length,
    }
    lengths = sorted(length for length in lengths if 1 <= length <= len(context_tokens))

    answer_counts = Counter(answer_tokens)
    best = (0, 0, -1.0, 0)
    for length in lengths:
        window_counts = Counter(context_tokens[:length])
        overlap = sum(
            min(answer_counts[token], count)
            for token, count in window_counts.items()
        )
        for start in range(0, len(context_tokens) - length + 1):
            if start:
                removed = context_tokens[start - 1]
                before = min(answer_counts.get(removed, 0), window_counts.get(removed, 0))
                window_counts[removed] -= 1
                overlap -= before - min(answer_counts.get(removed, 0), window_counts.get(removed, 0))
                added = context_tokens[start + length - 1]
                before = min(answer_counts.get(added, 0), window_counts.get(added, 0))
                window_counts[added] += 1
                overlap += min(answer_counts.get(added, 0), window_counts.get(added, 0)) - before
            score = 2.0 * overlap / (len(answer_tokens) + length) if overlap else 0.0
            candidate = (start, start + length - 1, score, length)
            # Score, then shorter span, then earlier span.
            if (score, -length, -start) > (best[2], -best[3], -best[0]):
                best = candidate
    return best[0], best[1], max(0.0, best[2])


def find_answer_span(
    answer: str,
    context_tokens: list[str],
    span_boundaries: list[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Match in priority order without allowing labels to cross passage edges."""
    answer_tokens = regex_tokens(answer)
    boundaries = span_boundaries or [(0, len(context_tokens) - 1)]
    boundaries = [
        (max(0, start), min(len(context_tokens) - 1, end))
        for start, end in boundaries
        if start <= end and start < len(context_tokens) and end >= 0
    ]

    def find_in_boundaries(needle: list[str]) -> tuple[int, int] | None:
        for lower, upper in boundaries:
            local = _find_subsequence(context_tokens[lower : upper + 1], needle)
            if local is not None:
                return lower + local[0], lower + local[1]
        return None

    exact = find_in_boundaries(answer_tokens)
    if exact is not None:
        return {"start": exact[0], "end": exact[1], "method": "exact", "overlap_score": 1.0}

    for fragment in _answer_fragments(answer):
        match = find_in_boundaries(fragment)
        if match is not None:
            return {"start": match[0], "end": match[1], "method": "fragment", "overlap_score": 1.0}

    answer_words = [token for token in answer_tokens if re.match(r"\w", token, re.UNICODE)]
    for lower, upper in boundaries:
        local_tokens = context_tokens[lower : upper + 1]
        word_positions = [
            index for index, token in enumerate(local_tokens)
            if re.match(r"\w", token, re.UNICODE)
        ]
        context_words = [local_tokens[index] for index in word_positions]
        word_match = _find_subsequence(context_words, answer_words)
        if word_match is not None:
            return {
                "start": lower + word_positions[word_match[0]],
                "end": lower + word_positions[word_match[1]],
                "method": "subsequence",
                "overlap_score": 1.0,
            }

    best = (0, 0, 0.0, 0)
    for lower, upper in boundaries:
        local_start, local_end, score = best_overlap_span(
            answer_tokens, context_tokens[lower : upper + 1]
        )
        candidate = (
            lower + local_start,
            lower + local_end,
            score,
            local_end - local_start + 1,
        )
        if (score, -candidate[3], -candidate[0]) > (best[2], -best[3], -best[0]):
            best = candidate
    start, end, score = best[:3]
    return {
        "start": start,
        "end": end,
        "method": "best_overlap",
        "overlap_score": score,
    }


def build_passages(
    contexts: list[dict[str, Any]],
    max_passages: int = 6,
    passage_tokens: int = 256,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Round-robin context chunks into a bounded, answer-independent input."""
    flat_tokens: list[str] = []
    max_context_tokens = max_passages * passage_tokens
    context_chunks: list[list[tuple[int, list[str]]]] = []
    for context in contexts:
        text = str(context.get("text", ""))
        context_tokens = [
            match.group(0) for match in islice(TOKEN_RE.finditer(text), max_context_tokens)
        ]
        chunks = []
        for chunk_index in range(0, len(context_tokens), passage_tokens):
            chunk_tokens = context_tokens[chunk_index : chunk_index + passage_tokens]
            if chunk_tokens:
                chunks.append((chunk_index // passage_tokens, chunk_tokens))
        context_chunks.append(chunks)

    selected: list[dict[str, Any]] = []
    max_rounds = max((len(chunks) for chunks in context_chunks), default=0)
    for round_index in range(max_rounds):
        for context_index, context in enumerate(contexts):
            if round_index >= len(context_chunks[context_index]):
                continue
            if len(selected) >= max_passages:
                return selected, flat_tokens
            chunk_index, chunk_tokens = context_chunks[context_index][round_index]
            global_start = len(flat_tokens)
            flat_tokens.extend(chunk_tokens)
            selected.append(
                {
                    "doc_id": context.get("doc_id", ""),
                    "title": context.get("title", ""),
                    "is_positive": bool(context.get("is_positive", False)),
                    "source_context_index": context_index,
                    "chunk_index": chunk_index // passage_tokens,
                    "global_start": global_start,
                    "global_end": len(flat_tokens) - 1,
                }
            )
    return selected, flat_tokens


class ExtractiveCollator:
    """Tokenize each 256-token passage and retain global context positions."""

    def __init__(self, tokenizer, max_passages: int = 6, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_passages = max_passages
        self.max_length = max_length

    @staticmethod
    def _token_spans(text: str) -> list[tuple[int, int]]:
        return [match.span() for match in TOKEN_RE.finditer(text)]

    def _encode_passage(self, question: str, passage_text: str) -> dict[str, Any]:
        encoded = self.tokenizer(
            question,
            passage_text,
            truncation="only_second",
            max_length=self.max_length,
            padding=False,
            return_offsets_mapping=True,
        )
        sequence_ids = encoded.sequence_ids()
        offsets = encoded["offset_mapping"]
        token_spans = self._token_spans(passage_text)
        local_to_subword: list[list[int]] = []
        passage_positions = [
            index for index, sequence_id in enumerate(sequence_ids) if sequence_id == 1
        ]
        for token_start, token_end in token_spans:
            matches = [
                index
                for index in passage_positions
                if offsets[index][0] < token_end and offsets[index][1] > token_start
            ]
            local_to_subword.append(matches)
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "passage_positions": passage_positions,
            "local_to_subword": local_to_subword,
        }

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        encoded_examples: list[list[dict[str, Any]]] = []
        max_length = 1
        for example in examples:
            encoded_passages = []
            for passage in example.get("passages", [])[: self.max_passages]:
                passage_text = detokenize_tokens(
                    example["flat_tokens"][
                        int(passage["global_start"]): int(passage["global_end"]) + 1
                    ]
                )
                encoded = self._encode_passage(
                    str(example["question"]), passage_text
                )
                encoded_passages.append(encoded)
                max_length = max(max_length, len(encoded["input_ids"]))
            encoded_examples.append(encoded_passages)

        batch_size = len(examples)
        input_ids = torch.full(
            (batch_size, self.max_passages, max_length),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros_like(input_ids)
        context_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        global_positions = torch.full_like(input_ids, -1)
        passage_mask = torch.zeros((batch_size, self.max_passages), dtype=torch.bool)
        passage_labels = torch.zeros((batch_size, self.max_passages), dtype=torch.float)
        start_positions = torch.zeros(batch_size, dtype=torch.long)
        end_positions = torch.zeros(batch_size, dtype=torch.long)

        metadata = []
        for batch_index, (example, encoded_passages) in enumerate(
            zip(examples, encoded_examples)
        ):
            metadata.append(
                {
                    "query_id": example["query_id"],
                    "answer": example["answer"],
                    "flat_tokens": example["flat_tokens"],
                }
            )
            start_lookup: dict[int, tuple[int, int]] = {}
            end_lookup: dict[int, tuple[int, int]] = {}
            for passage_index, (passage, encoded) in enumerate(
                zip(example.get("passages", [])[: self.max_passages], encoded_passages)
            ):
                length = len(encoded["input_ids"])
                input_ids[batch_index, passage_index, :length] = torch.tensor(
                    encoded["input_ids"], dtype=torch.long
                )
                attention_mask[batch_index, passage_index, :length] = torch.tensor(
                    encoded["attention_mask"], dtype=torch.long
                )
                passage_mask[batch_index, passage_index] = True
                passage_labels[batch_index, passage_index] = float(
                    passage.get("is_answer_passage", passage.get("is_positive", False))
                )
                for local_token, subword_positions in enumerate(
                    encoded["local_to_subword"]
                ):
                    if not subword_positions:
                        continue
                    global_index = int(passage["global_start"]) + local_token
                    for subword_index in subword_positions:
                        context_mask[batch_index, passage_index, subword_index] = True
                        global_positions[batch_index, passage_index, subword_index] = global_index
                    start_lookup[global_index] = (passage_index, subword_positions[0])
                    end_lookup[global_index] = (passage_index, subword_positions[-1])

            start_global = int(example["start_position"])
            end_global = int(example["end_position"])
            if start_global not in start_lookup or end_global not in end_lookup:
                raise ValueError(
                    f"Could not map gold span for {example['query_id']} "
                    f"({start_global}:{end_global}) to tokenizer positions; "
                    "the span may fall beyond a truncated passage."
                )
            start_passage, start_subword = start_lookup[start_global]
            end_passage, end_subword = end_lookup[end_global]
            start_positions[batch_index] = start_passage * max_length + start_subword
            end_positions[batch_index] = end_passage * max_length + end_subword

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "context_mask": context_mask,
            "global_positions": global_positions,
            "passage_mask": passage_mask,
            "passage_labels": passage_labels,
            "start_positions": start_positions,
            "end_positions": end_positions,
            "metadata": metadata,
        }
