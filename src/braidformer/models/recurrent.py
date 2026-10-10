"""A1 recurrent/state-space-majority decoder-only language model.

A1 changes only the sequence-mixing path relative to the A0 control.  Each
block uses a gated diagonal state-space recurrence with a reset state for each
input sequence.  The feed-forward path, embeddings, normalization, context
handling, loss, and tied language-model head remain compatible with A0.

This is an experimental control, not a claim that recurrent processing is
superior.  The implementation deliberately contains no attention, experts,
conditional depth, retrieval, or compute controller.  An optional associative
scan path is provided for benchmarking; the sequential path remains the
default reference implementation.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from braidformer.config import ExperimentConfig
from braidformer.models.transformer import CausalLMOutput, FeedForward, ModelConfigError


class GatedDiagonalStateSpaceMixer(nn.Module):
    """A causal diagonal state-space recurrence with input/output gates.

    The recurrence has one scalar transition coefficient per hidden channel.
    For a sequence ``x[0:L]`` it computes, independently for every batch item,

    ``state[t] = decay * state[t-1] + (1 - decay) * input_gate[t] * tanh(candidate[t])``

    followed by an output gate and a dense projection.  The state is reset to
    zero at the beginning of every supplied sequence, so packed training
    windows cannot leak state across examples.
    """

    def __init__(
        self,
        hidden_size: int,
        dropout: float,
        use_bias: bool = True,
        initial_decay: float = 0.9,
        use_parallel_scan: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ModelConfigError("hidden_size must be positive")
        if not 0.0 < initial_decay < 1.0:
            raise ModelConfigError("initial_decay must be strictly between 0 and 1")

        self.hidden_size = hidden_size
        self.use_parallel_scan = use_parallel_scan
        self.input_projection = nn.Linear(hidden_size, 3 * hidden_size, bias=use_bias)
        self.output_projection = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.state_transition_logit = nn.Parameter(torch.empty(hidden_size))
        self.residual_dropout = nn.Dropout(dropout)
        initial_logit = math.log(initial_decay / (1.0 - initial_decay))
        nn.init.constant_(self.state_transition_logit, initial_logit)

    def _sequential_scan(
        self,
        candidate: Tensor,
        input_gate: Tensor,
        decay: Tensor,
    ) -> Tensor:
        """Reference recurrence; intentionally simple and auditable."""

        batch_size, sequence_length, hidden_size = candidate.shape
        state = candidate.new_zeros((batch_size, hidden_size))
        outputs: list[Tensor] = []

        for timestep in range(sequence_length):
            candidate_t = torch.tanh(candidate[:, timestep, :])
            input_gate_t = torch.sigmoid(input_gate[:, timestep, :])
            state = decay * state + (1.0 - decay) * input_gate_t * candidate_t
            outputs.append(state)

        return torch.stack(outputs, dim=1)

    @staticmethod
    def _parallel_scan(decay: Tensor, updates: Tensor) -> Tensor:
        """Compute the same diagonal recurrence with an associative scan.

        Each timestep is represented by an affine update ``(a_t, b_t)``
        mapping ``state`` to ``a_t * state + b_t``.  Composition is

        ``(a_2, b_2) o (a_1, b_1) = (a_2*a_1, a_2*b_1 + b_2)``.

        The doubling scan below applies that associative composition in
        ``O(log(sequence_length))`` rounds of tensor operations.  It is
        mathematically valid here because both ``decay`` and ``updates`` are
        independent of the evolving state.
        """

        batch_size, sequence_length, hidden_size = updates.shape
        scan_a = decay.view(1, 1, hidden_size).expand(
            batch_size,
            sequence_length,
            hidden_size,
        )
        scan_b = updates
        offset = 1
        while offset < sequence_length:
            left_a = scan_a[:, :-offset, :]
            left_b = scan_b[:, :-offset, :]
            right_a = scan_a[:, offset:, :]
            right_b = scan_b[:, offset:, :]
            combined_a = right_a * left_a
            combined_b = right_a * left_b + right_b
            scan_a = torch.cat((scan_a[:, :offset, :], combined_a), dim=1)
            scan_b = torch.cat((scan_b[:, :offset, :], combined_b), dim=1)
            offset *= 2
        return scan_b

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Apply the recurrent mixer with a fresh state for each sequence."""

        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, sequence, hidden]")
        batch_size, sequence_length, hidden_size = hidden_states.shape
        if hidden_size != self.hidden_size:
            raise ValueError(
                f"hidden state width must be {self.hidden_size}, received {hidden_size}"
            )
        if sequence_length == 0:
            raise ValueError("hidden_states sequence length must be positive")

        projected = self.input_projection(hidden_states)
        candidate, input_gate, output_gate = projected.chunk(3, dim=-1)
        decay = torch.sigmoid(self.state_transition_logit).to(dtype=hidden_states.dtype)
        updates = (1.0 - decay).view(1, 1, hidden_size)
        updates = updates * torch.sigmoid(input_gate) * torch.tanh(candidate)
        if self.use_parallel_scan:
            states = self._parallel_scan(decay, updates)
        else:
            states = self._sequential_scan(candidate, input_gate, decay.unsqueeze(0))

        mixed = torch.sigmoid(output_gate) * states
        return self.residual_dropout(self.output_projection(mixed))


class A1RecurrentStateSpaceBlock(nn.Module):
    """A pre-normalized A1 recurrent block with the A0 feed-forward path."""

    def __init__(self, config: ExperimentConfig, *, use_parallel_scan: bool = False) -> None:
        super().__init__()
        self.state_space_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
        )
        self.state_space = GatedDiagonalStateSpaceMixer(
            hidden_size=config.hidden_size,
            dropout=config.dropout,
            use_bias=config.use_bias,
            use_parallel_scan=use_parallel_scan,
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
        hidden_states = hidden_states + self.state_space(self.state_space_norm(hidden_states))
        hidden_states = hidden_states + self.feed_forward(
            self.feed_forward_norm(hidden_states)
        )
        return hidden_states


class A1RecurrentStateSpaceLM(nn.Module):
    """A1 decoder-only LM with recurrent/state-space-majority sequence mixing.

    A1 accepts only an explicit ``stage='A1'`` and
    ``model_type='recurrent_state_space'`` configuration.  This guard keeps a
    later architecture from being run accidentally under the A0 label.
    """

    def __init__(self, config: ExperimentConfig, *, use_parallel_scan: bool = False) -> None:
        super().__init__()
        config.validate()
        if config.stage != "A1":
            raise ModelConfigError(
                f"A1RecurrentStateSpaceLM requires stage 'A1', received {config.stage!r}"
            )
        if config.model_type != "recurrent_state_space":
            raise ModelConfigError(
                "A1RecurrentStateSpaceLM requires model_type 'recurrent_state_space', "
                f"received {config.model_type!r}"
            )

        self.config = config
        self.use_parallel_scan = use_parallel_scan
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.context_length, config.hidden_size)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            A1RecurrentStateSpaceBlock(config, use_parallel_scan=use_parallel_scan)
            for _ in range(config.num_layers)
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
        """Run a causal language-model forward pass with optional loss."""

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
    "A1RecurrentStateSpaceBlock",
    "A1RecurrentStateSpaceLM",
    "GatedDiagonalStateSpaceMixer",
]
