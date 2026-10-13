"""A small controlled associative lane for the post-campaign falsification.

This module is intentionally separate from the frozen A0 and A1 definitions.
It wraps an unchanged A1 base and adds one 64-wide key/value lane.  The lane
has two modes with identical parameters:

* ``protected`` commits one binding and holds it between explicit value
  events;
* ``ordinary`` applies the same learned decay/update at every token.

The marker-directed event schedule is only for the frozen synthetic one-binding
diagnostic.  It does not read target labels or target values.
"""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from braidformer.config import ExperimentConfig
from braidformer.models.recurrent import A1RecurrentStateSpaceLM
from braidformer.models.transformer import CausalLMOutput, ModelConfigError


RECORD_TOKEN = 30_001
QUERY_TOKEN = 30_002
ANSWER_TOKEN = 30_003
LANE_WIDTH = 64
EXPECTED_LANE_PARAMETERS = 99_010


class A1AssociativeLaneLM(nn.Module):
    """Unchanged A1 plus a minimal one-binding associative lane."""

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        mode: str,
        lane_width: int = LANE_WIDTH,
        use_parallel_scan: bool = True,
    ) -> None:
        super().__init__()
        if mode not in {"protected", "ordinary"}:
            raise ModelConfigError(f"unknown associative-lane mode: {mode!r}")
        if lane_width != LANE_WIDTH:
            raise ModelConfigError(
                f"the frozen lane protocol requires lane_width={LANE_WIDTH}"
            )
        if config.stage != "A1P" or config.model_type != "recurrent_state_space_associative_lane":
            raise ModelConfigError(
                "A1AssociativeLaneLM requires stage='A1P' and "
                "model_type='recurrent_state_space_associative_lane'"
            )

        self.config = config
        self.mode = mode
        self.lane_width = lane_width
        base_config = replace(
            config,
            stage="A1",
            model_type="recurrent_state_space",
        )
        self.base = A1RecurrentStateSpaceLM(
            base_config,
            use_parallel_scan=use_parallel_scan,
        )
        self.key_projection = nn.Linear(config.hidden_size, lane_width, bias=True)
        self.value_projection = nn.Linear(config.hidden_size, lane_width, bias=True)
        self.readout = nn.Linear(lane_width, config.hidden_size, bias=True)
        self.memory_decay_logit = nn.Parameter(torch.empty(lane_width))
        self.read_scale = nn.Parameter(torch.tensor(4.0))
        self.read_bias = nn.Parameter(torch.tensor(0.0))
        self._reset_lane_parameters()

    def _reset_lane_parameters(self) -> None:
        for module in (self.key_projection, self.value_projection, self.readout):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        with torch.no_grad():
            initial_decay = torch.full_like(self.memory_decay_logit, 0.9)
            self.memory_decay_logit.copy_(torch.logit(initial_decay))
            self.read_scale.fill_(4.0)
            self.read_bias.zero_()

    def parameter_count(self, trainable_only: bool = True) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if not trainable_only or parameter.requires_grad
        )

    def _validate_input(self, input_ids: Tensor, labels: Tensor | None) -> None:
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be a rank-2 torch.long tensor")
        if input_ids.shape[1] == 0 or input_ids.shape[1] > self.config.context_length:
            raise ValueError("input sequence length is outside the configured context")
        if labels is not None and (labels.shape != input_ids.shape or labels.dtype != torch.long):
            raise ValueError("labels must be torch.long with the input shape")

    def _base_hidden(self, input_ids: Tensor) -> Tensor:
        sequence_length = input_ids.shape[1]
        positions = torch.arange(sequence_length, device=input_ids.device)
        hidden_states = self.base.token_embedding(input_ids)
        hidden_states = hidden_states + self.base.position_embedding(positions).unsqueeze(0)
        hidden_states = self.base.embedding_dropout(hidden_states)
        for block in self.base.blocks:
            hidden_states = block(hidden_states)
        return hidden_states

    def _lane_forward(
        self,
        hidden_states: Tensor,
        input_ids: Tensor,
        *,
        return_trace: bool,
    ) -> tuple[Tensor, dict[str, Tensor] | None]:
        batch_size, sequence_length, _hidden_size = hidden_states.shape
        key_candidates = self.key_projection(hidden_states)
        value_candidates = self.value_projection(hidden_states)
        decay = torch.sigmoid(self.memory_decay_logit).to(dtype=hidden_states.dtype)

        key_event = torch.zeros(
            (batch_size, sequence_length), dtype=torch.bool, device=input_ids.device
        )
        if sequence_length > 1:
            key_event[:, 1:] = input_ids[:, :-1].eq(RECORD_TOKEN)
        value_event = torch.zeros_like(key_event)
        if sequence_length > 1:
            value_event[:, 1:] = key_event[:, :-1]
        answer_event = input_ids.eq(ANSWER_TOKEN)
        query_event = torch.zeros_like(answer_event)
        if sequence_length > 1:
            query_event[:, 1:] = input_ids[:, :-1].eq(QUERY_TOKEN)

        pending_key = hidden_states.new_zeros((batch_size, self.lane_width))
        memory_key = hidden_states.new_zeros((batch_size, self.lane_width))
        memory_value = hidden_states.new_zeros((batch_size, self.lane_width))
        deltas: list[Tensor] = []
        memory_key_trace: list[Tensor] = []
        memory_value_trace: list[Tensor] = []
        similarity_trace: list[Tensor] = []
        gate_trace: list[Tensor] = []

        for timestep in range(sequence_length):
            key_mask = key_event[:, timestep].unsqueeze(-1)
            value_mask = value_event[:, timestep].unsqueeze(-1)
            answer_mask = answer_event[:, timestep].unsqueeze(-1)
            pending_key = torch.where(key_mask, key_candidates[:, timestep, :], pending_key)

            if self.mode == "protected":
                committed_key = decay * memory_key + (1.0 - decay) * pending_key
                committed_value = decay * memory_value + (1.0 - decay) * value_candidates[:, timestep, :]
                memory_key = torch.where(value_mask, committed_key, memory_key)
                memory_value = torch.where(value_mask, committed_value, memory_value)
            else:
                ordinary_key = key_candidates[:, timestep, :]
                ordinary_value = value_candidates[:, timestep, :]
                ordinary_key = torch.where(value_mask, pending_key, ordinary_key)
                memory_key = decay * memory_key + (1.0 - decay) * ordinary_key
                memory_value = decay * memory_value + (1.0 - decay) * ordinary_value

            if timestep == 0:
                query_key = key_candidates[:, timestep, :].detach() * 0.0
            else:
                query_key = key_candidates[:, timestep - 1, :]
            similarity = (
                F.normalize(memory_key, dim=-1, eps=1e-6)
                * F.normalize(query_key, dim=-1, eps=1e-6)
            ).sum(dim=-1)
            read_gate = torch.sigmoid(
                self.read_scale.to(dtype=similarity.dtype) * similarity
                + self.read_bias.to(dtype=similarity.dtype)
            )
            lane_read = self.readout(read_gate.unsqueeze(-1) * memory_value)
            delta = torch.where(answer_mask, lane_read, torch.zeros_like(lane_read))
            deltas.append(delta)
            if return_trace:
                memory_key_trace.append(memory_key.detach())
                memory_value_trace.append(memory_value.detach())
                similarity_trace.append(similarity.detach())
                gate_trace.append(read_gate.detach())

        lane_delta = torch.stack(deltas, dim=1)
        if not return_trace:
            return lane_delta, None
        return lane_delta, {
            "memory_key": torch.stack(memory_key_trace, dim=1),
            "memory_value": torch.stack(memory_value_trace, dim=1),
            "similarity": torch.stack(similarity_trace, dim=1),
            "read_gate": torch.stack(gate_trace, dim=1),
            "lane_delta": lane_delta.detach(),
            "key_event": key_event,
            "value_event": value_event,
            "query_event": query_event,
            "answer_event": answer_event,
        }

    def _forward_impl(
        self,
        input_ids: Tensor,
        labels: Tensor | None,
        *,
        return_trace: bool,
    ) -> tuple[CausalLMOutput, dict[str, Tensor] | None]:
        self._validate_input(input_ids, labels)
        hidden_states = self._base_hidden(input_ids)
        lane_delta, trace = self._lane_forward(
            hidden_states,
            input_ids,
            return_trace=return_trace,
        )
        hidden_states = hidden_states + lane_delta
        logits = self.base.lm_head(self.base.final_norm(hidden_states))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, self.config.vocab_size),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return CausalLMOutput(logits=logits, loss=loss), trace

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
    ) -> CausalLMOutput:
        output, _trace = self._forward_impl(input_ids, labels, return_trace=False)
        return output

    def forward_with_lane_trace(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
    ) -> tuple[CausalLMOutput, dict[str, Tensor]]:
        output, trace = self._forward_impl(input_ids, labels, return_trace=True)
        if trace is None:
            raise RuntimeError("lane trace was not returned")
        return output, trace


__all__ = [
    "A1AssociativeLaneLM",
    "ANSWER_TOKEN",
    "EXPECTED_LANE_PARAMETERS",
    "LANE_WIDTH",
    "QUERY_TOKEN",
    "RECORD_TOKEN",
]
