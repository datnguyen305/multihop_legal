"""Lightweight answer-generation metrics with optional BERTScore."""

from __future__ import annotations

import re
import unicodedata


class _EmptyWordNet:
    """METEOR fallback for languages/environments without an NLTK WordNet corpus."""

    @staticmethod
    def synsets(_word: str) -> list:
        return []


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str) -> list[str]:
    return normalize_text(text).split()


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred = tokens(prediction)
    ref = tokens(reference)
    if not pred or not ref:
        return float(pred == ref)
    previous = [0] * (len(ref) + 1)
    for token in pred:
        current = [0]
        for index, ref_token in enumerate(ref, start=1):
            current.append(previous[index - 1] + 1 if token == ref_token else max(previous[index], current[-1]))
        previous = current
    lcs = previous[-1]
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def meteor_score(prediction: str, reference: str) -> float:
    if normalize_text(prediction) == normalize_text(reference):
        return 1.0
    try:
        from nltk.translate.meteor_score import single_meteor_score
    except ImportError:
        return 0.0
    reference_tokens = tokens(reference)
    prediction_tokens = tokens(prediction)
    try:
        return float(single_meteor_score(reference_tokens, prediction_tokens))
    except LookupError:
        # NLTK's WordNet synonym lookup is English-only and often absent in
        # offline environments. Preserve exact/stem matching without requiring
        # an external corpus download (the same fallback is used for every run).
        return float(
            single_meteor_score(
                reference_tokens,
                prediction_tokens,
                wordnet=_EmptyWordNet(),
            )
        )


def compute_basic_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    if not predictions:
        return {"rougeL": 0.0, "meteor": 0.0}
    rouge = sum(rouge_l_f1(p, r) for p, r in zip(predictions, references)) / len(predictions)
    meteor = sum(meteor_score(p, r) for p, r in zip(predictions, references)) / len(predictions)
    return {"rougeL": rouge, "meteor": meteor}


def compute_bertscore(
    predictions: list[str], references: list[str], model_type: str = "xlm-roberta-large"
) -> float | None:
    try:
        from bert_score import score
    except ImportError:
        return None
    _, _, f1 = score(predictions, references, model_type=model_type, verbose=True)
    return float(f1.mean().item())
