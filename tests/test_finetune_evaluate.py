"""Offline metric checks for malformed outputs, exact types, and abstention."""

from typing import Any

from finetune.evaluate import canonical, score, summarize


def example(calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Build an embedded held-out record without service clients or model files.

    Args:
        calls:
            Native function name/argument targets; empty denotes abstention.

    """
    return {
        "id": "fixture",
        "family": "fixture",
        "command_language": "en",
        "surface_language": "en",
        "entity_languages": {},
        "asr_errors": [],
        "music_state": "off",
        "messages": [
            {"role": "user", "content": "Turn on the light"},
            {"role": "assistant", "tool_calls": [{"function": call} for call in calls]},
        ],
    }


def test_typed_call_and_invalid_scalar() -> None:
    """Require a true boolean rather than a numerically equal integer argument."""
    row = example([{"name": "set_light", "arguments": {"on": True}}])
    valid = score(
        row, "<|tool_call_start|>[set_light(on=True)]<|tool_call_end|><|im_end|>", 12
    )
    invalid = score(
        row, "<|tool_call_start|>[set_light(on=1)]<|tool_call_end|><|im_end|>", 12
    )
    assert valid["correct"] and not invalid["correct"]
    assert invalid["error"]
    assert canonical(True) != canonical(1)


def test_abstention_requires_valid_nonempty_output() -> None:
    """Reject malformed, empty, and truncated responses instead of counting abstention."""
    row = example([])
    assert score(row, "Please clarify.<|im_end|>", 5)["correct"]
    for raw, count in [
        ("<|im_end|>", 1),
        ("<|tool_call_start|>[", 4),
        ("Please clarify.", 128),
    ]:
        result = score(row, raw, count)
        assert not result["correct"] and result["error"]


def test_metrics_include_false_activations_and_slice_counts() -> None:
    """Report valid unwanted actions separately from malformed generations."""
    row = example([])
    results = [
        score(row, "Please clarify.<|im_end|>", 5),
        score(row, "<|tool_call_start|>[resume_music()]<|tool_call_end|><|im_end|>", 8),
        score(row, "<|tool_call_start|>", 1),
    ]
    metrics = summarize(results)
    assert metrics["slices"]["no_tool"] == {"n": 3, "correct": 1, "accuracy": 1 / 3}
    assert metrics["invalid"] == 1
    assert metrics["false_tool_activations"] == 1


def test_relative_volume_default_matches_explicit_target() -> None:
    """Score omitted relative amounts using the same five-point default as dispatch."""
    row = example(
        [{"name": "volume_music", "arguments": {"action": "louder", "level": 5}}]
    )
    result = score(
        row,
        "<|tool_call_start|>[volume_music(action='louder')]<|tool_call_end|><|im_end|>",
        12,
    )
    assert result["correct"]
