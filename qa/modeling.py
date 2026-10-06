"""Model-family helpers for fair multilingual Vietnamese QA experiments."""

from __future__ import annotations


MODEL_PRESETS = {
    "vit5": "VietAI/vit5-base",
    "bartpho": "vinai/bartpho-syllable-base",
    "mt5": "google/mt5-base",
    "mbart": "facebook/mbart-large-50-many-to-many-mmt",
}


def model_family(model_name: str) -> str:
    name = model_name.lower()
    if "mbart" in name:
        return "mbart"
    if "bartpho" in name:
        return "bartpho"
    if "vit5" in name:
        return "vit5"
    if "mt5" in name:
        return "mt5"
    return "generic"


def configure_tokenizer(tokenizer, model_name: str):
    """Configure Vietnamese source/target language when the tokenizer needs it."""
    if model_family(model_name) == "mbart":
        tokenizer.src_lang = "vi_VN"
        tokenizer.tgt_lang = "vi_VN"
    return tokenizer


def generation_kwargs(tokenizer, model_name: str) -> dict:
    """Return generation controls shared by evaluation, always greedy."""
    kwargs = {"num_beams": 1, "do_sample": False}
    if model_family(model_name) == "mbart":
        language_ids = getattr(tokenizer, "lang_code_to_id", {})
        if "vi_VN" not in language_ids:
            raise ValueError("mBART tokenizer does not expose the vi_VN language code")
        kwargs["forced_bos_token_id"] = language_ids["vi_VN"]
    return kwargs
