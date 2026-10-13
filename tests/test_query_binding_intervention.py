from __future__ import annotations

import pytest
import torch

from scripts.run_relational_curriculum_query_binding_intervention import (
    query_conditioned_binding_loss,
)


def test_query_conditioned_binding_loss_scores_key_value_pair() -> None:
    embedding = torch.zeros(16_000, 4)
    embedding[1_000] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    embedding[1_001] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    embedding[8_000] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    embedding[8_001] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    states = torch.zeros(2, 5, 4)
    states[0, 4] = embedding[1_000]
    states[1, 4] = embedding[1_001]
    input_ids = torch.tensor(
        [
            [30_000, 30_001, 1_000, 30_002, 1_000],
            [30_000, 30_001, 1_001, 30_002, 1_001],
        ],
        dtype=torch.long,
    )
    labels = torch.tensor(
        [
            [-100, -100, -100, -100, 8_000],
            [-100, -100, -100, -100, 8_001],
        ],
        dtype=torch.long,
    )
    correct = query_conditioned_binding_loss(states, input_ids, labels, embedding)
    swapped = query_conditioned_binding_loss(
        states,
        input_ids,
        torch.tensor(
            [
                [-100, -100, -100, -100, 8_001],
                [-100, -100, -100, -100, 8_000],
            ],
            dtype=torch.long,
        ),
        embedding,
    )
    assert torch.isfinite(correct)
    assert float(correct) < float(swapped)


def test_query_conditioned_binding_loss_requires_one_query_marker() -> None:
    states = torch.zeros(1, 4, 4)
    input_ids = torch.tensor([[30_000, 30_001, 1_000, 1_000]], dtype=torch.long)
    labels = torch.tensor([[-100, -100, -100, 8_000]], dtype=torch.long)
    embedding = torch.ones(16_000, 4)
    with pytest.raises(ValueError, match="exactly one query marker"):
        query_conditioned_binding_loss(states, input_ids, labels, embedding)


def test_query_conditioned_binding_loss_supports_multi_token_targets() -> None:
    embedding = torch.zeros(16_000, 4)
    embedding[1_000] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    embedding[8_000] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    embedding[8_001] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    states = torch.zeros(1, 6, 4)
    states[0, 4] = embedding[1_000]
    input_ids = torch.tensor(
        [[30_000, 30_001, 1_000, 30_002, 1_000, 8_000]], dtype=torch.long
    )
    labels = torch.tensor(
        [[-100, -100, -100, -100, 8_000, 8_001]], dtype=torch.long
    )
    loss = query_conditioned_binding_loss(states, input_ids, labels, embedding)
    assert torch.isfinite(loss)
