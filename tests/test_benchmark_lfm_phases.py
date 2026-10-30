"""Offline contract tests for matched LFM phase measurements."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tools.benchmark_lfm_phases import load_tokens, measure


class FakeBackend:
    """Record stateful calls and supply deterministic finite logits."""

    calls: list[Any]
    """Ordered reset and inference events."""

    def __init__(self) -> None:
        """Create an empty call log."""
        self.calls = []

    def reset(self) -> None:
        """Record each independent sequence reset."""
        self.calls.append("reset")

    def infer(self, tokens: list[int]) -> np.ndarray:
        """Record exact tokens and return final-position fixture logits.

        Args:
            tokens:
                Full prompt or one teacher-forced decode token.
        """
        self.calls.append(tokens.copy())
        result = np.zeros(65536, dtype=np.float32)
        result[tokens[-1]] = 1
        return result

    def device_ns(self) -> None:
        """Represent a wall-only run without device profiling."""
        return


def test_measure_teacher_forced_context_and_warmups() -> None:
    """Every repetition resets and replays identical tokens and contexts."""
    backend = FakeBackend()
    records: list[dict[str, Any]] = []
    logits = measure(backend, [3, 4], [5, 6], 1, 2, records.append)
    assert backend.calls == ["reset", [3, 4], [5], [6]] * 3
    assert logits.shape == (3, 65536)
    assert [int(np.argmax(row)) for row in logits] == [4, 5, 6]
    assert [r["context_tokens"] for r in records] == [2, 3, 4] * 3
    assert [r["warmup"] for r in records] == [True] * 3 + [False] * 6
    assert [r["input_token"] for r in records[:3]] == [None, 5, 6]
    assert all(r["profile_sum_ns"] is None for r in records)


def test_fixture_validation(tmp_path: Path) -> None:
    """Reject malformed fixtures and preserve explicit valid token sequences.

    Args:
        tmp_path:
            Isolated temporary directory provided by pytest.
    """
    path = tmp_path / "tokens.json"
    path.write_text(json.dumps({"prompt": [1, 2], "continuation": [3]}))
    assert load_tokens(path) == ([1, 2], [3])
    for bad in (
        {"prompt": [True], "continuation": [3]},
        {"prompt": [65536], "continuation": [3]},
        {"prompt": [], "continuation": [3]},
        {"prompt": [1], "continuation": [3], "extra": 1},
    ):
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            load_tokens(path)
