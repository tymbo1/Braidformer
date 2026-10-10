"""A0 conventional decoder-only Transformer control.

This module intentionally contains only the control architecture.  It has no
recurrent path, retrieval mechanism, experts, conditional depth, or compute
controller.  Those mechanisms belong to later, separately configured stages.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from braidformer.config import ExperimentConfig


class ModelConfigError(ValueError):
    """Raised when a model is asked to use an incompatible experiment config."""


@dataclass(frozen=True)
class CausalLMOutput:
    """Outputs from :class:`A0TransformerLM`."""

    logits: Tensor
    loss: Tensor | None = None


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        dropout: float,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        if hidden_size % num_attention_heads:
            raise ModelConfigError("hidden_size must be divisible by num_attention_heads")
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.attention_dropout = dropout
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=use_bias)
        self.output = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        batch_size, sequence_length, hidden_size = hidden_states.shape
        qkv = self.qkv(hidden_states)
        query, key, value = qkv.chunk(3, dim=-1)

        query = query.view(
            batch_size,
            sequence_length,
            self.num_attention_heads,
            self.head_dim,
        ).transpose(1, 2)
        key = key.view(
            batch_size,
            sequence_length,
            self.num_attention_heads,
            self.head_dim,
        ).transpose(1, 2)
        value = value.view(
            batch_size,
            sequence_length,
            self.num_attention_heads,
            self.head_dim,
        ).transpose(1, 2)

        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size,
            sequence_length,
            hidden_size,
        )
        return self.residual_dropout(self.output(attended))


class FeedForward(nn.Module):
    """The position-wise feed-forward block used by the A0 control."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        self.input = nn.Linear(hidden_size, intermediate_size, bias=use_bias)
        self.activation = nn.GELU()
        self.output = nn.Linear(intermediate_size, hidden_size, bias=use_bias)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.residual_dropout(self.output(self.activation(self.input(hidden_states))))


class TransformerBlock(nn.Module):
    """A pre-normalized Transformer block for the A0 control."""

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        self.attention = CausalSelfAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            dropout=config.dropout,
            use_bias=config.use_bias,
        )
        self.feed_forward_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        self.feed_forward = FeedForward(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            dropout=config.dropout,
            use_bias=config.use_bias,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = hidden_states + self.attention(self.attention_norm(hidden_states))
        hidden_states = hidden_states + self.feed_forward(
            self.feed_forward_norm(hidden_states)
        )
        return hidden_states


class A0TransformerLM(nn.Module):
    """A conventional decoder-only language model used as the A0 control.

    The model consumes a validated :class:`ExperimentConfig` and refuses
    configurations for later architecture stages.  This prevents an A1-A5
    experiment from silently running the A0 control under the wrong label.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        config.validate()
        if config.stage != "A0":
            raise ModelConfigError(
                f"A0TransformerLM requires stage 'A0', received {config.stage!r}"
            )
        if config.model_type != "transformer":
            raise ModelConfigError(
                "A0TransformerLM requires model_type 'transformer', "
                f"received {config.model_type!r}"
            )

        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.context_length, config.hidden_size)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.num_layers)
        )
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def parameter_count(self, trainable_only: bool = True) -> int:
        """Return the number of unique scalar parameters in the model."""

        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if not trainable_only or parameter.requires_grad
        )

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
    ) -> CausalLMOutput:
        """Run a causal language-model forward pass.

        When ``labels`` are supplied, the loss is the standard next-token
        cross entropy: token ``t`` predicts token ``t + 1``.  ``-100`` labels
        are ignored using PyTorch's conventional cross-entropy sentinel.
        """

        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.dtype != torch.long:
            raise ValueError("input_ids must use torch.long dtype")
        batch_size, sequence_length = input_ids.shape
        if sequence_length == 0:
            raise ValueError("input_ids sequence length must be positive")
        if sequence_length > self.config.context_length:
            raise ValueError(
                "input sequence length exceeds configured context_length "
                f"({sequence_length} > {self.config.context_length})"
            )
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")
            if labels.dtype != torch.long:
                raise ValueError("labels must use torch.long dtype")
            if sequence_length < 2:
                raise ValueError("at least two tokens are required to compute causal loss")

        positions = torch.arange(sequence_length, device=input_ids.device)
        hidden_states = self.token_embedding(input_ids)
        hidden_states = hidden_states + self.position_embedding(positions).unsqueeze(0)
        hidden_states = self.embedding_dropout(hidden_states)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        logits = self.lm_head(self.final_norm(hidden_states))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, self.config.vocab_size),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return CausalLMOutput(logits=logits, loss=loss)


__all__ = [
    "A0TransformerLM",
    "CausalLMOutput",
    "ModelConfigError",
]
