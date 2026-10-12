from scripts.analyze_a1_state_decay_forensics import (
    DISTRACTOR_MARKER,
    PROBE_DISTANCES,
    VALUE_BASE,
    _probe_example,
)


def test_forensic_probe_examples_are_deterministic_and_distance_aligned() -> None:
    for split in ("train", "heldout"):
        for index in (0, 17, 511):
            for distance in PROBE_DISTANCES:
                first = _probe_example(split, index, distance)
                second = _probe_example(split, index, distance)
                assert first == second
                tokens, target = first
                assert len(tokens) == 8 + distance
                assert tokens[3] == target
                assert tokens[4 : 4 + distance] == (DISTRACTOR_MARKER,) * distance
                assert tokens[-1] == target
                assert VALUE_BASE <= target < VALUE_BASE + 8_000


def test_forensic_probe_splits_use_disjoint_key_domains() -> None:
    train_tokens, _ = _probe_example("train", 7, 16)
    heldout_tokens, _ = _probe_example("heldout", 7, 16)

    assert train_tokens[2] != heldout_tokens[2]
    assert train_tokens[2] < 8_000
    assert heldout_tokens[2] >= 20_000
