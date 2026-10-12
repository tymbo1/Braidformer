"""Model implementations for the staged BraidFormer experiments."""

from braidformer.models.recurrent import (
    A1RecurrentStateSpaceBlock,
    A1RecurrentStateSpaceLM,
    GatedDiagonalStateSpaceMixer,
    initial_state_transition_decays,
)
from braidformer.models.transformer import A0TransformerLM, CausalLMOutput, ModelConfigError

__all__ = [
    "A0TransformerLM",
    "A1RecurrentStateSpaceBlock",
    "A1RecurrentStateSpaceLM",
    "CausalLMOutput",
    "GatedDiagonalStateSpaceMixer",
    "initial_state_transition_decays",
    "ModelConfigError",
]
