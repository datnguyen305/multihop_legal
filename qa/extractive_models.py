"""Paper-inspired extractive QA heads on a shared XLM-R encoder."""

from __future__ import annotations

import torch
from torch import nn


class GraphBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        heads = num_heads if hidden_size % num_heads == 0 else 1
        self.attention = nn.MultiheadAttention(hidden_size, heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if not torch.all(mask.any(dim=-1)):
            raise ValueError("Each example needs at least one real passage")
        values = values.masked_fill(~mask.unsqueeze(-1), 0.0)
        attended, _ = self.attention(
            values,
            values,
            values,
            key_padding_mask=~mask,
            need_weights=False,
        )
        values = self.norm(values + attended)
        values = values.masked_fill(~mask.unsqueeze(-1), 0.0)
        values = self.norm(values + self.ffn(values))
        return values.masked_fill(~mask.unsqueeze(-1), 0.0)


class BaseExtractiveModel(nn.Module):
    def __init__(self, encoder_name: str):
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = int(self.encoder.config.hidden_size)
        self.hidden_size = hidden
        self.dropout = nn.Dropout(0.1)
        self.span_head = nn.Linear(hidden, 2)
        self.passage_head = nn.Linear(hidden, 1)

    def encode(self, input_ids, attention_mask, context_mask):
        batch_size, passages, length = input_ids.shape
        encoded = self.encoder(
            input_ids=input_ids.view(batch_size * passages, length),
            attention_mask=attention_mask.view(batch_size * passages, length),
        ).last_hidden_state.view(batch_size, passages, length, self.hidden_size)
        context_weights = context_mask.float().unsqueeze(-1)
        context_sum = (encoded * context_weights).sum(dim=2)
        context_count = context_weights.sum(dim=2).clamp_min(1.0)
        passage_repr = context_sum / context_count
        return encoded, passage_repr

    def apply_span_head(self, encoded):
        logits = self.span_head(self.dropout(encoded))
        return logits[..., 0], logits[..., 1]

    def forward(self, input_ids, attention_mask, context_mask, passage_mask, **kwargs):
        encoded, passage_repr = self.encode(input_ids, attention_mask, context_mask)
        start_logits, end_logits = self.apply_span_head(encoded)
        passage_logits = self.passage_head(self.dropout(passage_repr)).squeeze(-1)
        start_logits = start_logits.masked_fill(~context_mask, -1e4)
        end_logits = end_logits.masked_fill(~context_mask, -1e4)
        passage_logits = passage_logits.masked_fill(~passage_mask, -1e4)
        return {
            "start_logits": start_logits,
            "end_logits": end_logits,
            "passage_logits": passage_logits,
        }


class QANetExtractiveModel(BaseExtractiveModel):
    """QANet-inspired convolution and self-attention span head."""

    def __init__(self, encoder_name: str):
        super().__init__(encoder_name)
        self.conv = nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=5, padding=2)
        heads = 8 if self.hidden_size % 8 == 0 else 1
        self.self_attention = nn.MultiheadAttention(
            self.hidden_size, heads, batch_first=True
        )
        self.qanet_norm = nn.LayerNorm(self.hidden_size)

    def forward(self, input_ids, attention_mask, context_mask, passage_mask, **kwargs):
        encoded, passage_repr = self.encode(input_ids, attention_mask, context_mask)
        batch, passages, length, hidden = encoded.shape
        context = encoded.view(batch * passages, length, hidden)
        convolved = torch.relu(self.conv(context.transpose(1, 2)).transpose(1, 2))
        attended, _ = self.self_attention(
            convolved,
            convolved,
            convolved,
            key_padding_mask=self._safe_context_mask(context_mask, batch, passages, length),
            need_weights=False,
        )
        attended = attended.masked_fill(
            ~context_mask.view(batch * passages, length).any(dim=-1)[:, None, None],
            0.0,
        )
        encoded = self.qanet_norm(context + attended).view(batch, passages, length, hidden)
        start_logits, end_logits = self.apply_span_head(encoded)
        passage_logits = self.passage_head(self.dropout(passage_repr)).squeeze(-1)
        return {
            "start_logits": start_logits.masked_fill(~context_mask, -1e4),
            "end_logits": end_logits.masked_fill(~context_mask, -1e4),
            "passage_logits": passage_logits.masked_fill(~passage_mask, -1e4),
        }

    @staticmethod
    def _safe_context_mask(context_mask, batch, passages, length):
        flat_mask = context_mask.view(batch * passages, length)
        safe_mask = flat_mask.clone()
        empty_rows = ~safe_mask.any(dim=-1)
        safe_mask[empty_rows, 0] = True
        return ~safe_mask


class DFGNExtractiveModel(BaseExtractiveModel):
    """DFGN-inspired dynamic fusion of passage graph and query state."""

    def __init__(self, encoder_name: str):
        super().__init__(encoder_name)
        self.graph = nn.ModuleList([GraphBlock(self.hidden_size) for _ in range(2)])
        self.query_gate = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.fusion = nn.Linear(self.hidden_size * 2, self.hidden_size)

    def forward(self, input_ids, attention_mask, context_mask, passage_mask, **kwargs):
        encoded, passage_repr = self.encode(input_ids, attention_mask, context_mask)
        passage_weights = passage_mask.unsqueeze(-1).to(encoded.dtype)
        query = (
            (encoded[:, :, 0, :] * passage_weights).sum(dim=1, keepdim=True)
            / passage_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        )
        graph = passage_repr
        for layer in self.graph:
            graph = layer(graph, passage_mask)
            gate = torch.sigmoid(self.query_gate(torch.cat([graph, query.expand_as(graph)], dim=-1)))
            graph = gate * graph + (1.0 - gate) * passage_repr
        fused = self.fusion(torch.cat([encoded, graph.unsqueeze(2).expand_as(encoded)], dim=-1))
        start_logits, end_logits = self.apply_span_head(fused)
        passage_logits = self.passage_head(self.dropout(graph)).squeeze(-1)
        return {
            "start_logits": start_logits.masked_fill(~context_mask, -1e4),
            "end_logits": end_logits.masked_fill(~context_mask, -1e4),
            "passage_logits": passage_logits.masked_fill(~passage_mask, -1e4),
        }


class HGNExtractiveModel(BaseExtractiveModel):
    """HGN-inspired hierarchical passage graph and token gating."""

    def __init__(self, encoder_name: str):
        super().__init__(encoder_name)
        self.graph = GraphBlock(self.hidden_size)
        self.hierarchy_gate = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.fusion = nn.Linear(self.hidden_size * 2, self.hidden_size)

    def forward(self, input_ids, attention_mask, context_mask, passage_mask, **kwargs):
        encoded, passage_repr = self.encode(input_ids, attention_mask, context_mask)
        graph = self.graph(passage_repr, passage_mask)
        gate = torch.sigmoid(self.hierarchy_gate(torch.cat([passage_repr, graph], dim=-1)))
        hierarchical = gate * graph + (1.0 - gate) * passage_repr
        fused = self.fusion(torch.cat([encoded, hierarchical.unsqueeze(2).expand_as(encoded)], dim=-1))
        start_logits, end_logits = self.apply_span_head(fused)
        passage_logits = self.passage_head(self.dropout(hierarchical)).squeeze(-1)
        return {
            "start_logits": start_logits.masked_fill(~context_mask, -1e4),
            "end_logits": end_logits.masked_fill(~context_mask, -1e4),
            "passage_logits": passage_logits.masked_fill(~passage_mask, -1e4),
        }


class CoGExtractiveModel(BaseExtractiveModel):
    """CoG-inspired iterative cognitive graph over candidate passages."""

    def __init__(self, encoder_name: str):
        super().__init__(encoder_name)
        self.graph = nn.ModuleList([GraphBlock(self.hidden_size) for _ in range(2)])
        self.query_update = nn.GRUCell(self.hidden_size, self.hidden_size)
        self.fusion = nn.Linear(self.hidden_size * 3, self.hidden_size)

    def forward(self, input_ids, attention_mask, context_mask, passage_mask, **kwargs):
        encoded, passage_repr = self.encode(input_ids, attention_mask, context_mask)
        passage_weights = passage_mask.unsqueeze(-1).to(encoded.dtype)
        query = (
            (encoded[:, :, 0, :] * passage_weights).sum(dim=1)
            / passage_weights.sum(dim=1).clamp_min(1.0)
        )
        graph = passage_repr
        for layer in self.graph:
            graph = layer(graph, passage_mask)
            scores = self.passage_head(graph).squeeze(-1).masked_fill(~passage_mask, -1e4)
            weights = torch.softmax(scores, dim=-1)
            selected = torch.bmm(weights.unsqueeze(1), graph).squeeze(1)
            query = self.query_update(selected, query)
        fused = self.fusion(
            torch.cat(
                [
                    encoded,
                    graph.unsqueeze(2).expand_as(encoded),
                    query[:, None, None, :].expand_as(encoded),
                ],
                dim=-1,
            )
        )
        start_logits, end_logits = self.apply_span_head(fused)
        passage_logits = self.passage_head(self.dropout(graph)).squeeze(-1)
        return {
            "start_logits": start_logits.masked_fill(~context_mask, -1e4),
            "end_logits": end_logits.masked_fill(~context_mask, -1e4),
            "passage_logits": passage_logits.masked_fill(~passage_mask, -1e4),
        }


MODEL_CLASSES = {
    "cog": CoGExtractiveModel,
    "dfgn": DFGNExtractiveModel,
    "hgn": HGNExtractiveModel,
    "qanet": QANetExtractiveModel,
}


def build_extractive_model(method: str, encoder_name: str) -> nn.Module:
    if method not in MODEL_CLASSES:
        raise ValueError(f"Unknown extractive method: {method}")
    return MODEL_CLASSES[method](encoder_name)
