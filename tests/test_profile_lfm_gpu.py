"""Hardware-independent checks of custom runner event profiling."""

import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hoast.lfm_gpu.runner import Runner
from tools.profile_lfm_gpu import (
    KernelTrace,
    execute,
    layer_loop_lines,
    main,
    prompt_ids,
    workspace_spec,
)


class FakeRunner:
    """Minimal runner with the same dispatch and layer-loop structure."""

    workspace: Any
    """Fake queue holder."""
    kernels: dict[str, Any]
    """Fake event-producing kernels."""
    weights: dict[str, Any]
    """Synthetic projection weight metadata."""

    def __init__(self) -> None:
        """Create a kernel that records no queue synchronization."""
        self.workspace = SimpleNamespace(queue=object())
        self.kernels = {
            "kernel": lambda *args: SimpleNamespace(
                profile=SimpleNamespace(start=10, end=35)
            )
        }

    def _run(
        self,
        name: str,
        global_size: tuple[int, ...],
        local_size: tuple[int, ...] | None,
        *args: Any,
    ) -> None:
        """Mimic the original runner dispatch.

        Args:
            name:
                Kernel name.

            global_size:
                Work dimensions.

            local_size:
                Optional group dimensions.

            *args:
                Kernel arguments.
        """
        self.kernels[name](self.workspace.queue, global_size, local_size, *args)

    def _project(self, name: Any, *args: Any) -> None:
        """Dispatch a labeled projection.

        Args:
            name:
                Projection label.

            *args:
                Additional synthetic projection arguments.
        """
        self._run("kernel", (1,), None)

    def _chunk(self) -> None:
        """Dispatch kernels before, during and after the layer loop."""
        self._run("kernel", (1,), None)
        for layer in range(2):
            self._project(f"weight.{layer}")
            self._run("kernel", (2,), None)
        self._project("final")


def test_trace_labels_and_no_synchronization() -> None:
    """Retain every event and distinguish layer, projection and final kernels."""
    runner = FakeRunner()
    original = runner._run
    trace = KernelTrace(runner)
    with trace:
        runner._chunk()
        rows, summary = trace.drain()
    assert runner._run == original
    assert [row["layer"] for row in rows] == [None, 0, 0, 1, 1, None]
    operations = [row["operation"] for row in rows]
    assert operations[1] == "weight.0" and operations[3] == "weight.1"
    assert operations[5] == "final"
    assert operations[0].startswith("kernel@line:")
    assert operations[2] == operations[4] and operations[2].startswith("kernel@line:")
    assert sum(row["device_ns"] for row in summary) == 150
    assert all(row["duration_ns"] == 25 for row in rows)
    assert trace.drain() == ([], [])


def test_optimized_kernel_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retain projection names and per-callsite labels for optimized kernels.

    Args:
        monkeypatch:
            Installs the actual projection method on the synthetic runner.
    """

    class OptimizedRunner(FakeRunner):
        """Exercise the real optimized projection selection without OpenCL."""

        def _chunk(self) -> None:
            """Emit a projection and tiled scores in each synthetic layer."""
            for layer in range(2):
                self._project(None, None, f"weight.{layer}", 128, 4, 8)
                self._project(None, None, f"weight.{layer}", 1, 4, 8)
                self._run("lfm_gqa_scores_tiled_16_8_64", (64,), (64,))

    monkeypatch.setattr(OptimizedRunner, "_project", Runner._project)
    runner = OptimizedRunner()
    runner.workspace.spec = SimpleNamespace(
        prepared_prefill=True, prepared_kn=True, persistent_kn=False, vec4_matvec=True
    )
    runner.workspace.buffers = {"prepared_weights": object()}
    runner.weights = {f"weight.{i}": (None, None, None, (8, 4)) for i in range(2)}
    runner.kernels = {
        name: lambda *args: SimpleNamespace(profile=SimpleNamespace(start=10, end=35))
        for name in (
            "lfm_prepare_u8_f16_kn",
            "lfm_matmul_prepared_kn_f16",
            "lfm_matvec_u8_f16_dequant_vec4",
            "lfm_gqa_scores_tiled_16_8_64",
        )
    }
    with KernelTrace(runner) as trace:
        runner._chunk()
        rows, summary = trace.drain()
    assert len(rows) == len(summary) == 8
    for layer in range(2):
        group = rows[layer * 4 : (layer + 1) * 4]
        assert [row["layer"] for row in group] == [layer] * 4
        assert [row["operation"] for row in group[:3]] == [f"weight.{layer}"] * 3
        assert group[3]["operation"].startswith("lfm_gqa_scores_tiled_16_8_64@line:")
        assert all(row["duration_ns"] == 25 for row in group)


def test_workspace_optimized_flags_and_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass feature flags to the workspace and reject incompatible CLI options.

    Args:
        monkeypatch:
            Patches command-line arguments without creating output or using a GPU.
    """
    options = {
        "decode_steps": 8,
        "chunk_size": 128,
        "prepared_prefill": True,
        "prepared_kn": True,
        "tiled_attention": True,
        "vec4_matvec": True,
    }
    spec = workspace_spec(Namespace(**options))
    assert (spec.capacity, spec.chunk_size) == (938, 128)
    assert (
        spec.prepared_prefill
        and spec.prepared_kn
        and spec.tiled_attention
        and spec.vec4_matvec
    )
    options.update(
        prepared_prefill=False,
        prepared_kn=False,
        tiled_attention=False,
        vec4_matvec=False,
    )
    spec = workspace_spec(Namespace(**options))
    assert not (
        spec.prepared_prefill
        or spec.prepared_kn
        or spec.tiled_attention
        or spec.vec4_matvec
    )
    options["prepared_kn"] = True
    with pytest.raises(ValueError, match="prepared_kn requires prepared_prefill"):
        workspace_spec(Namespace(**options))
    monkeypatch.setattr(
        "sys.argv",
        [
            "profile",
            "--checkpoint",
            "x",
            "--device",
            "x",
            "--output",
            "x",
            "--prepared-kn",
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


def test_context_matches_runner_options(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report the same optimization flags actually supplied to the runner.

    Args:
        monkeypatch:
            Replaces native runtimes and artifacts with in-memory fixtures.
    """

    class Device:
        """Synthetic OpenCL device metadata."""

        name: str = "fake"
        """Device name."""
        driver_version: str = "fake-driver"
        """Device driver version."""
        platform: Any = SimpleNamespace(name="fake-platform")
        """Device platform metadata."""

    seen: list[Any] = []

    class DummyRunner(FakeRunner):
        """Record the supplied workspace without creating device resources."""

        def __init__(self, queue: Any, checkpoint: Any, spec: Any) -> None:
            """Capture the spec passed by the profiler.

            Args:
                queue:
                    Synthetic profiling queue.

                checkpoint:
                    Synthetic checkpoint path.

                spec:
                    Actual requested workspace configuration.
            """
            super().__init__()
            seen.append(spec)

    cl = SimpleNamespace(
        get_platforms=lambda: [SimpleNamespace(get_devices=lambda **kw: [Device()])],
        device_type=SimpleNamespace(GPU=1),
        Context=lambda devices: object(),
        CommandQueue=lambda *args, **kwargs: object(),
        command_queue_properties=SimpleNamespace(PROFILING_ENABLE=1),
    )
    tokenizer = SimpleNamespace(from_pretrained=lambda *args, **kwargs: object())
    monkeypatch.setitem(sys.modules, "pyopencl", cl)
    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(AutoTokenizer=tokenizer)
    )
    monkeypatch.setattr("hoast.lfm_gpu.runner.Runner", DummyRunner)
    monkeypatch.setattr("tools.profile_lfm_gpu.sha256", lambda path: "fixture")
    monkeypatch.setattr(
        "tools.profile_lfm_gpu.prompt_ids", lambda tokenizer: list(range(930))
    )
    args = Namespace(
        checkpoint=Path("."),
        device="fake",
        decode_steps=1,
        chunk_size=128,
        prepared_prefill=True,
        prepared_kn=True,
        tiled_attention=True,
        vec4_matvec=True,
        warmups=0,
        repeats=0,
        wall_only=False,
    )
    records: list[dict[str, Any]] = []
    execute(args, records.append)
    context = records[0]
    assert records[-1]["stage"] == "complete"
    for key in ("prepared_prefill", "prepared_kn", "tiled_attention", "vec4_matvec"):
        assert context[key] is True and getattr(seen[0], key) is True


def test_restores_on_failure() -> None:
    """Always restore dispatch after an exception and reject ambiguous source."""
    runner = FakeRunner()
    original = runner._run
    with pytest.raises(RuntimeError), KernelTrace(runner):
        raise RuntimeError("failure")
    assert runner._run == original
    with pytest.raises(ValueError):
        layer_loop_lines(FakeRunner._project)


def test_prompt_count_validation() -> None:
    """Reject workloads whose rendered token count differs from 930."""

    class Tokenizer:
        """Fake tokenizer retaining the requested template options."""

        count: int
        """Number of returned token IDs."""

        def __init__(self, count: int) -> None:
            """Store the synthetic count.

            Args:
                count:
                    Number of token IDs to return.
            """
            self.count = count

        def apply_chat_template(self, *args: Any, **kwargs: Any) -> str:
            """Return synthetic rendered text.

            Args:
                *args:
                    Positional template inputs.

                **kwargs:
                    Template options.
            """
            assert kwargs["add_generation_prompt"] is True
            assert kwargs["tools"]
            return "weather"

        def __call__(self, text: str, **kwargs: Any) -> Any:
            """Return synthetic token IDs.

            Args:
                text:
                    Rendered text.

                **kwargs:
                    Tokenization options.
            """
            assert text == "weather" and kwargs["add_special_tokens"] is False
            return SimpleNamespace(input_ids=[list(range(self.count))])

    assert len(prompt_ids(Tokenizer(930))) == 930
    with pytest.raises(ValueError, match="got 929"):
        prompt_ids(Tokenizer(929))
