"""CPU-only tests for the live strict-prefix adapter contracts."""

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from test_lfm_gpu_generation import make_adapter, scores

from hoast.input_text import REPAIR_FEEDBACK_KEY
from hoast.lfm import LFMConfig
from hoast.lfm_gpu import generation, live
from hoast.lfm_gpu import runner as runner_module
from hoast.lfm_gpu.generation import GPUGeneration
from hoast.lfm_gpu.live import LiveLFM2
from hoast.session import Session


def test_callback_abort_restores_strict_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Callback failures abort token generation and preserve the strict snapshot."""

    class Streamer:
        """Forward one fragment per token for deterministic cancellation testing."""

        def __init__(self, callback: Any) -> None:
            self.callback = callback

        def put(self, tokens: np.ndarray) -> None:
            self.callback(str(tokens[0]))

        def end(self) -> None:
            pass

    monkeypatch.setattr(generation, "_text_streamer", lambda t, c, s: Streamer(c))
    plain, runner, tokenizer = make_adapter([scores(**{"3": 5}), scores(**{"7": 5})])
    adapter = GPUGeneration(runner, tokenizer, plain.tools, plain.config, (1,), True)
    adapter.initialize_prefix()
    received: list[str] = []

    def cancel(fragment: str) -> None:
        received.append(fragment)
        raise RuntimeError("cancel")

    with pytest.raises(RuntimeError, match="cancel"):
        adapter.generate_messages([{"role": "user", "content": "hello"}], cancel)
    assert received == ["3"]
    assert runner.prefix == (1,)
    assert adapter.generate("hello").text == "Hello"


def test_live_repair_marker_and_exact_prefix_miss() -> None:
    """Reject public repair markers while accepting internally constructed feedback."""
    plain, runner, tokenizer = make_adapter([scores(**{"7": 5})])
    adapter = GPUGeneration(runner, tokenizer, plain.tools, plain.config, (1,), True)
    adapter.initialize_prefix()
    history = [{"role": "user", "content": "Syntax: [x=1]", REPAIR_FEEDBACK_KEY: True}]
    with pytest.raises(ValueError, match="Internal repair"):
        adapter.generate_messages(history)
    with pytest.raises(TypeError):
        adapter.generate_messages(history, allow_repair=True)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="Internal repair"):
        adapter._generate_repair(history, "diagnostic")
    adapter._generate_repair([{"role": "user", "content": "hello"}], "Syntax: [x=1]")
    assert tokenizer.history[2]["content"] == "Syntax: [x=1]"
    tokenizer.token_ids = (2, 1)
    with pytest.raises(
        ValueError, match="^Prompt does not match strict stable prefix$"
    ):
        adapter._generate_repair(
            [{"role": "user", "content": "hello"}], "Syntax: [x=1]"
        )
    assert runner.prefix == (1,)


def test_live_reentrancy_and_public_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lifecycle calls from callbacks fail without changing loaded model state."""

    class Streamer:
        """Send one fragment to the supplied callback."""

        def __init__(self, callback: Any) -> None:
            self.callback = callback

        def put(self, tokens: np.ndarray) -> None:
            self.callback("fragment")

        def end(self) -> None:
            pass

    monkeypatch.setattr(generation, "_text_streamer", lambda t, c, s: Streamer(c))
    plain, runner, tokenizer = make_adapter([scores(**{"3": 5}), scores(**{"7": 5})])
    adapter = GPUGeneration(runner, tokenizer, plain.tools, plain.config, (1,), True)
    adapter.initialize_prefix()
    model = LiveLFM2(
        LFMConfig(
            Path("unused"),
            max_new_tokens=3,
            max_context_tokens=32,
            reclaim_after_call=False,
        ),
        plain.tools,
    )
    model._model, model._tokenizer = adapter, tokenizer
    with pytest.raises(ValueError, match="Internal repair"):
        model.generate_messages(
            [{"role": "user", "content": "x", REPAIR_FEEDBACK_KEY: True}]
        )

    def callback(fragment: str) -> None:
        for operation in (model.close, model.load):
            with pytest.raises(RuntimeError, match="Reentrant"):
                operation()

    model.generate_messages([{"role": "user", "content": "hello"}], callback)
    assert model._model is adapter and model._tokenizer is tokenizer
    model.close()
    assert model._model is None


def test_rollback_failure_preserves_callback_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rollback failure faults the adapter and is noted on the original error."""

    class Streamer:
        """Abort generation after arranging a rollback failure."""

        def put(self, tokens: np.ndarray) -> None:
            runner.fail_reset = True
            raise ValueError("callback failure")

    monkeypatch.setattr(generation, "_text_streamer", lambda t, c, s: Streamer())
    plain, runner, tokenizer = make_adapter([scores(**{"3": 5})])
    adapter = GPUGeneration(runner, tokenizer, plain.tools, plain.config, (1,), True)
    adapter.initialize_prefix()
    with pytest.raises(ValueError, match="callback failure") as caught:
        adapter.generate_messages(
            [{"role": "user", "content": "hello"}], lambda _: None
        )
    assert any(
        "rollback failed" in note and "reset failure" in note
        for note in caught.value.__notes__
    )
    with pytest.raises(RuntimeError, match="snapshot is invalid"):
        adapter.generate("hello")


def test_session_repair_uses_private_live_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Session constructs transient repair feedback and keeps it out of history."""

    class Streamer:
        """Accept tokens without decoding intermediate fragments."""

        def put(self, tokens: np.ndarray) -> None:
            pass

        def end(self) -> None:
            pass

    monkeypatch.setattr(generation, "_text_streamer", lambda t, c, s: Streamer())
    plain, runner, tokenizer = make_adapter([scores(**{"7": 5})])
    adapter = GPUGeneration(runner, tokenizer, plain.tools, plain.config, (1,), True)
    adapter.initialize_prefix()
    model = LiveLFM2(
        LFMConfig(
            Path("unused"),
            max_new_tokens=2,
            max_context_tokens=32,
            reclaim_after_call=False,
        ),
        plain.tools,
    )
    model._model, model._tokenizer = adapter, tokenizer
    original = tokenizer.decode
    calls = 0

    def decode(tokens: list[int], **options: Any) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            return "<|tool_call_start|>[broken(]<|tool_call_end|>"
        return original(tokens, **options)

    monkeypatch.setattr(tokenizer, "decode", decode)
    session = Session(model)
    assert "".join(session.stream("hello", max_repair_attempts=1)) == "Hello"
    assert all(REPAIR_FEEDBACK_KEY not in message for message in session.history)
    assert "Validation error:" in tokenizer.history[-1]["content"]


@pytest.mark.parametrize("capacity,chunk", [(32, 32), (256, 128)])
def test_load_workspace_capacity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capacity: int, chunk: int
) -> None:
    """Live loading caps prefill chunks at the configured workspace capacity."""
    import transformers

    (tmp_path / "openvino_model.xml").touch()
    (tmp_path / "tokenizer_config.json").write_text(
        '{"tokenizer_class":"PreTrainedTokenizerFast"}'
    )
    plain, fake_runner, tokenizer = make_adapter([scores(**{"7": 5})])
    fake_cl = SimpleNamespace(
        get_platforms=lambda: [SimpleNamespace(get_devices=lambda **kw: [object()])],
        device_type=SimpleNamespace(GPU=1),
        Context=lambda devices: object(),
        CommandQueue=lambda context: object(),
    )
    monkeypatch.setitem(sys.modules, "pyopencl", fake_cl)
    monkeypatch.setattr(
        transformers.PreTrainedTokenizerFast,
        "from_pretrained",
        lambda *args, **kwargs: tokenizer,
    )
    monkeypatch.setattr(live, "stable_prefix", lambda *args: (1,))

    def make_runner(queue: Any, path: Path, spec: Any) -> Any:
        assert spec.capacity == capacity and spec.chunk_size == chunk
        return fake_runner

    monkeypatch.setattr(runner_module, "Runner", make_runner)
    model = LiveLFM2(
        LFMConfig(tmp_path, max_new_tokens=2, max_context_tokens=capacity), plain.tools
    )
    assert model.load() is model
    model.close()
