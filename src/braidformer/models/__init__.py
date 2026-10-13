"""Model implementations for the staged BraidFormer experiments."""

from braidformer.models.recurrent import (
    A1RecurrentStateSpaceBlock,
    A1RecurrentStateSpaceLM,
    GatedDiagonalStateSpaceMixer,
    initial_state_transition_decays,
)
from braidformer.models.protected_lane import (
    A1AssociativeLaneLM,
    ANSWER_TOKEN,
    EXPECTED_LANE_PARAMETERS,
    LANE_WIDTH,
    QUERY_TOKEN,
    RECORD_TOKEN,
)
from braidformer.models.transformer import A0TransformerLM, CausalLMOutput, ModelConfigError

__all__ = [
    "A0TransformerLM",
    "A1RecurrentStateSpaceBlock",
    "A1RecurrentStateSpaceLM",
    "A1AssociativeLaneLM",
    "ANSWER_TOKEN",
    "CausalLMOutput",
    "GatedDiagonalStateSpaceMixer",
    "EXPECTED_LANE_PARAMETERS",
    "LANE_WIDTH",
    "QUERY_TOKEN",
    "RECORD_TOKEN",
    "initial_state_transition_decays",
    "ModelConfigError",
]
