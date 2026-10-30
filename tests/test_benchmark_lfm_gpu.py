"""Model- and hardware-independent checks for the LFM weather benchmark."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hoast.llm import Generation, ToolCall
from tools.benchmark_lfm_gpu import (
    REQUESTS,
    RunnerTrace,
    choose_device,
    common_prefix,
    parity,
    result_record,
    run_requests,
    sha256,
)


def test_prefix_and_device_selection() -> None:
    """Require a unique named GPU and an exact nonempty common token prefix."""
    assert common_prefix([[1, 2, 3], [1, 2, 4], [1, 2]]) == (1, 2)
    for sequences in ([], [[]], [[1], [2]]):
        with pytest.raises(ValueError):
            common_prefix(sequences)
    devices = [SimpleNamespace(name="GPU alpha"), SimpleNamespace(name="GPU beta")]
    assert choose_device(devices, "ALPHA") is devices[0]
    for name in ("", "GPU", "missing"):
        with pytest.raises(ValueError):
            choose_device(devices, name)


def test_result_parity_and_hash(tmp_path: Path) -> None:
    """Preserve full calls and raw text and reject mismatched workloads.

    Args:
        tmp_path:
            Directory for an isolated digest fixture.
    """
    path = tmp_path / "fixture"
    path.write_bytes(b"abc")
    assert (
        sha256(path)
        == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    generation = Generation(
        "", (ToolCall("weather", {"city": "Paris"}),), "raw", 5, 3, 1.0, 0.5
    )
    result = result_record("question", generation, 1.5)
    assert result["calls"] == [{"name": "weather", "arguments": {"city": "Paris"}}]
    assert parity([result], [result])[0]["calls_equal"]
    changed = dict(result, raw="other", calls=[])
    assert not parity([result], [changed])[0]["raw_equal"]
    assert not parity([result], [changed])[0]["calls_equal"]
    with pytest.raises(ValueError):
        parity([result], [])
    with pytest.raises(ValueError):
        parity([result], [dict(result, request="different")])


def test_request_measurement_and_runner_trace() -> None:
    """Call each synthetic request once and time one prefix build without a GPU."""
    seen: list[str] = []

    def generate(text: str) -> Generation:
        """Return a fixed response and record its request.

        Args:
            text:
                Synthetic request passed to the fake backend.
        """
        seen.append(text)
        return Generation("ok", (), "ok", 1, 1, 0.0, 0.0)

    rows = run_requests(generate, REQUESTS)
    assert seen == list(REQUESTS) and len(rows) == 3
    assert all(row["wall_seconds"] >= 0 for row in rows)

    class FakeRunner:
        """In-memory runner supporting the tracing interface."""

        workspace: Any = SimpleNamespace()
        """Synthetic workspace placeholder."""

        def reset(self) -> None:
            """Accept a state reset."""

        def prefill(self, ids: list[int]) -> list[int]:
            """Return the supplied prompt IDs.

            Args:
                ids:
                    Synthetic prompt IDs.
            """
            return ids

        def cache_prefix(self, ids: list[int]) -> list[int]:
            """Return the prefix IDs without device access.

            Args:
                ids:
                    Synthetic prefix IDs.
            """
            return ids

        def prefill_cached(self, ids: list[int]) -> list[int]:
            """Return the supplied cached prompt IDs.

            Args:
                ids:
                    Synthetic prompt IDs.
            """
            return ids

        def decode(self, token: int) -> int:
            """Return the supplied generated token.

            Args:
                token:
                    Synthetic token ID.
            """
            return token

    trace = RunnerTrace(FakeRunner())
    trace.reset()
    assert trace.prefill([1]) == [1]
    assert trace.cache_prefix([1, 2]) == [1, 2]
    assert trace.prefill_cached([1, 2, 3]) == [1, 2, 3]
    assert trace.decode(4) == 4
    assert len(trace.cache_build_seconds) == 1 and trace.cache_build_seconds[0] >= 0
