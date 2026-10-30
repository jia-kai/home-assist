"""CPU-only tests for the live strict-prefix adapter contracts."""

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from test_lfm_gpu_generation import FakeRunner, make_adapter, scores

from hoast.input_text import REPAIR_FEEDBACK_KEY
from hoast.lfm import LFMConfig
from hoast.lfm_gpu import generation, live
from hoast.lfm_gpu import runner as runner_module
from hoast.lfm_gpu.generation import GPUGeneration
from hoast.lfm_gpu.live import LiveLFM2
from hoast.prompts import build_system_prompt
from hoast.session import Session


def test_stable_prefix_covers_all_tool_definitions() -> None:
    """The derived exact token prefix includes the final rendered tool schema."""
    class CharacterTokenizer:
        """Use character IDs to make template boundaries directly inspectable."""

        def apply_chat_template(self, messages: Any, **options: Any) -> str:
            """Render the system, complete tool list, and variable user content.

            Args:
                messages: System and user messages.

                **options: Template options including tool declarations.
            """
            return (
                "<|im_start|>system\n" + messages[0]["content"]
                + "\nList of tools: " + repr(options["tools"])
                + "<|im_end|>\n<|im_start|>user\n" + messages[1]["content"]
            )

        def __call__(self, prompt: str, **options: Any) -> Any:
            """Tokenize each character as an integer.

            Args:
                prompt: Rendered conversation.

                **options: Tokenizer options.
            """
            return SimpleNamespace(input_ids=np.array([[ord(c) for c in prompt]]))

        def decode(self, ids: list[int], **options: Any) -> str:
            """Decode character IDs.

            Args:
                ids: Character IDs.

                **options: Decode options.
            """
            return "".join(chr(i) for i in ids)

    plain, _, _ = make_adapter([scores(**{"7": 5})])
    tokenizer = CharacterTokenizer()
    prefix = live.stable_prefix(tokenizer, "fixed", plain.tools)
    decoded = tokenizer.decode(list(prefix))
    assert repr(live.tool_declarations(plain.tools)) in decoded
    assert "<|im_end|>\n<|im_start|>user\n" in decoded
    for user in ("hello", "goodbye", "中"):
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": "fixed"}, {"role": "user", "content": user}],
            tools=live.tool_declarations(plain.tools),
        )
        assert tuple(ord(c) for c in prompt[:len(prefix)]) == prefix


class DualRunner(FakeRunner):
    """CPU runner fixture recording independent snapshots and selected slots."""

    def __init__(self) -> None:
        """Create two empty slots with deterministic EOS logits."""
        super().__init__([scores(**{"7": 5})])
        self.workspace.spec.prefix_slots = 2
        self.snapshots: dict[int, tuple[int, ...]] = {}
        self.selected: list[int] = []

    def cache_prefix(self, token_ids: list[int], *, slot: int = 0) -> np.ndarray:
        """Save the requested slot.

        Args:
            token_ids: Prefix token IDs.

            slot: Snapshot index.
        """
        self.snapshots[slot] = tuple(token_ids)
        return self.outputs[0]

    def prefill_cached(self, token_ids: list[int], *, slot: int = 0) -> np.ndarray:
        """Restore one snapshot and validate its exact prefix.

        Args:
            token_ids: Prompt token IDs.

            slot: Snapshot index.
        """
        prefix = self.snapshots[slot]
        assert tuple(token_ids[:len(prefix)]) == prefix
        self.selected.append(slot)
        return self.outputs[0]


def test_dual_strict_snapshots_selection_and_mismatch() -> None:
    """Alternate both snapshots and reject a miss without touching runner state."""
    plain, _, tokenizer = make_adapter([scores(**{"7": 5})])
    runner = DualRunner()
    adapter = GPUGeneration(runner, tokenizer, plain.tools, plain.config, (1,), True, (2,))
    adapter.initialize_prefix()
    assert runner.snapshots == {0: (1,), 1: (2,)}
    for slot in (0, 1, 0, 1):
        tokenizer.token_ids = (slot + 1, 3)
        adapter.generate("hello")
    assert runner.selected == [0, 0, 1, 1, 0, 0, 1, 1]
    before = len(runner.sequences)
    tokenizer.token_ids = (3, 1)
    with pytest.raises(ValueError, match="strict stable prefix"):
        adapter.generate("hello")
    assert len(runner.sequences) == before


def test_second_snapshot_callback_failure_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A callback exception restores the selected second snapshot.

    Args:
        monkeypatch: Replaces streaming with a deterministic throwing callback.
    """
    class Streamer:
        """Forward emitted token text to a caller callback."""

        def put(self, tokens: np.ndarray) -> None:
            """Abort after prefill.

            Args:
                tokens: Emitted token IDs.
            """
            raise RuntimeError("callback failed")

    monkeypatch.setattr(generation, "_text_streamer", lambda *args: Streamer())
    plain, _, tokenizer = make_adapter([scores(**{"7": 5})])
    runner = DualRunner()
    adapter = GPUGeneration(runner, tokenizer, plain.tools, plain.config, (1,), True, (2,))
    adapter.initialize_prefix()
    tokenizer.token_ids = (2, 3)
    with pytest.raises(RuntimeError, match="callback failed"):
        adapter.generate_messages([{"role": "user", "content": "hello"}], lambda _: None)
    assert runner.selected == [1, 1]
    tokenizer.token_ids = (1, 3)
    assert adapter.generate("hello").text == "Hello"


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
    model._base_prompt = model.config.system_prompt
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
    model._base_prompt = model.config.system_prompt
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
    plain, _, tokenizer = make_adapter([scores(**{"7": 5})])
    fake_runner = DualRunner()
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
    monkeypatch.setattr(
        live, "stable_prefix", lambda t, prompt, tools: (1,) if '"on"' in prompt else (2,)
    )

    def make_runner(queue: Any, path: Path, spec: Any) -> Any:
        assert spec.capacity == capacity and spec.chunk_size == chunk
        assert spec.prefix_slots == 2
        return fake_runner

    monkeypatch.setattr(runner_module, "Runner", make_runner)
    model = LiveLFM2(
        LFMConfig(tmp_path, max_new_tokens=2, max_context_tokens=capacity), plain.tools
    )
    assert model.load() is model
    model.close()


def test_loaded_base_prompt_selects_off_without_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Direct generation maps the captured base to off across close and reload.

    Args:
        monkeypatch: Supplies deterministic tokenizer and OpenCL substitutes.

        tmp_path: Temporary model artifact directory.
    """
    import transformers

    (tmp_path / "openvino_model.xml").touch()
    (tmp_path / "tokenizer_config.json").write_text(
        '{"tokenizer_class":"PreTrainedTokenizerFast"}'
    )
    plain, _, tokenizer = make_adapter([scores(**{"7": 5})])
    fake_runner = DualRunner()
    fake_cl = SimpleNamespace(
        get_platforms=lambda: [SimpleNamespace(get_devices=lambda **kw: [object()])],
        device_type=SimpleNamespace(GPU=1),
        Context=lambda devices: object(),
        CommandQueue=lambda context: object(),
    )
    monkeypatch.setitem(sys.modules, "pyopencl", fake_cl)
    monkeypatch.setattr(
        transformers.PreTrainedTokenizerFast, "from_pretrained", lambda *a, **kw: tokenizer
    )
    monkeypatch.setattr(
        live, "stable_prefix", lambda t, prompt, tools: (1,) if '"on"' in prompt else (2,)
    )
    monkeypatch.setattr(runner_module, "Runner", lambda *a: fake_runner)
    original_template = tokenizer.apply_chat_template

    def render(history: Any, **options: Any) -> str:
        """Select token IDs from the rendered system message.

        Args:
            history: Structured conversation, including the system prompt.

            **options: Native template options.
        """
        system = history[0]["content"]
        tokenizer.token_ids = (1 if '"on"' in system else 2 if '"off"' in system else 3, 3)
        return original_template(history, **options)

    monkeypatch.setattr(tokenizer, "apply_chat_template", render)
    model = LiveLFM2(
        LFMConfig(tmp_path, max_new_tokens=2, max_context_tokens=32, reclaim_after_call=False),
        plain.tools,
    )
    base = model.config.system_prompt
    model.load()
    assert model.config.system_prompt == base
    assert model.generate("hello").text == "Hello"
    assert fake_runner.selected[-2:] == [1, 1]
    on = build_system_prompt({"status": "observed", "state": "on"}, base)
    model.config = replace(model.config, system_prompt=on)
    model.generate("hello")
    assert fake_runner.selected[-2:] == [0, 0]
    model.config = replace(model.config, system_prompt="unexpected")
    with pytest.raises(ValueError, match="strict stable prefix"):
        model.generate("hello")
    model.close()
    model.config = replace(model.config, system_prompt="new base")
    model.load()
    assert model.config.system_prompt == "new base"
    model.generate("hello")
    assert fake_runner.selected[-2:] == [1, 1]
    model.close()
