"""Offline checks and opt-in IR parity for the experimental LFM runner."""

import hashlib
import json
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

import hoast.lfm_gpu.runner as runner_module
from hoast.lfm_gpu.runner import (
    PERSISTENT_KN_BYTES,
    PREPARED_WEIGHT_BYTES,
    Runner,
    Workspace,
    WorkspaceSpec,
    _validate_memory_limits,
    _validate_norm,
    _verify_artifacts,
    _verify_digest,
    validate_config,
)
from hoast.lfm_gpu.weights import Constant


def test_opencl_sources_are_adjacent_to_runner() -> None:
    """Ensure the runner can load each relocated OpenCL source without a GPU."""
    directory = Path(runner_module.__file__).parent
    for name in (
        "lfm_gpu_kernels.cl",
        "lfm_gpu_ops.cl",
        "lfm_gpu_runner.cl",
    ):
        assert "__kernel" in (directory / name).read_text()


@pytest.fixture
def scheduled_runner(monkeypatch: pytest.MonkeyPatch) -> tuple[Runner, MagicMock]:
    """Create a runner whose actual scheduler dispatches to recorded CPU stubs.

    Args:
        monkeypatch:
            Replaces device kernel dispatch, projection and normalization.
    """
    runner = Runner.__new__(Runner)
    runner.workspace = Workspace.__new__(Workspace)
    runner.workspace.spec = WorkspaceSpec(12, 2, prefix_cache=True)
    runner.workspace.position = 0
    runner.workspace.queue = MagicMock()
    runner.workspace.buffers = {
        name: object() for name in runner.workspace.spec.buffer_sizes
    }
    runner._token_host = np.empty(2, dtype=np.int32)
    runner._logits_host = np.full(65536, 42, dtype=np.float32)
    runner._prefix_logits = np.empty(65536, dtype=np.float32)
    runner._prefix_tokens = None
    runner._lock = threading.Lock()
    runner.valid = True
    runner.weights = {
        "self.model.model.embed_tokens.weight": (None, None, None, (65536, 1024))
    }
    runner.conv = {layer: object() for layer in range(16)}
    runner.cl = SimpleNamespace(enqueue_copy=MagicMock())
    calls = MagicMock()
    monkeypatch.setattr(Runner, "_run", calls.run)
    monkeypatch.setattr(Runner, "_project", calls.project)
    monkeypatch.setattr(Runner, "_norm", calls.norm)
    return runner, calls


@pytest.mark.parametrize("length", [1, 2, 3, 4, 5])
def test_chunk_scheduler_projects_only_final_logits(
    scheduled_runner: tuple[Runner, MagicMock], length: int
) -> None:
    """Dispatch every hybrid layer per chunk and read logits once per prefill.

    Args:
        scheduled_runner:
            Runner with real chunk scheduling and recorded device operations.

        length:
            Prompt length, including exact and partial chunk boundaries.
    """
    runner, calls = scheduled_runner
    logits = runner.prefill(list(range(length)))
    chunks = (length + 1) // 2
    assert runner.workspace.position == length
    np.testing.assert_array_equal(logits, runner._logits_host)
    assert logits is not runner._logits_host
    names = [call.args[0] for call in calls.run.call_args_list]
    assert names.count("lfm_causal_conv3") == chunks * 10
    assert names.count("lfm_kv_append") == chunks * 6
    assert names.count("lfm_last") == 1
    projects = [call.args[2] for call in calls.project.call_args_list]
    assert projects.count("self.model.model.embed_tokens.weight") == 1
    norms = [call.args[2] for call in calls.norm.call_args_list]
    assert norms.count("__module.model.model.embedding_norm/aten::mul/Multiply_1") == 1
    copies = runner.cl.enqueue_copy.call_args_list
    assert len(copies) == chunks + 1
    assert copies[-1].args[1] is runner._logits_host
    assert copies[-1].kwargs["is_blocking"] is True
    runner.decode(7)
    assert runner.workspace.position == length + 1
    assert [call.args[0] for call in calls.run.call_args_list].count("lfm_last") == 2


@pytest.mark.parametrize("tiled", [False, True])
def test_attention_score_dispatch_and_order(
    scheduled_runner: tuple[Runner, MagicMock], tiled: bool
) -> None:
    """Dispatch scores between cache append and application for each chunk.

    Args:
        scheduled_runner:
            Runner with real chunk scheduling and recorded device operations.

        tiled:
            Whether to select the fixed-shape tiled GQA kernel.
    """
    runner, calls = scheduled_runner
    runner.workspace.spec = WorkspaceSpec(12, 2, tiled_attention=tiled)
    runner.prefill([1, 2, 3, 4, 5])
    records = calls.run.call_args_list
    score_name = "lfm_gqa_scores_tiled_16_8_64" if tiled else "lfm_gqa_scores"
    score_indices = [i for i, call in enumerate(records) if call.args[0] == score_name]
    assert len(score_indices) == 18
    for number, index in enumerate(score_indices):
        args = records[index].args
        assert records[index - 1].args[0] == "lfm_kv_append"
        assert records[index + 1].args[0] == "lfm_gqa_apply"
        start = (number // 6) * 2
        count = 1 if start == 4 else 2
        if tiled:
            assert args[1:3] == ((64 * ((start + count + 7) // 8), 1, 8), (64, 1, 1))
            assert args[6:] == (count, start, 12)
        else:
            assert args[1:3] == ((start + count, count * 16), None)
            assert args[6:] == (count, 16, 8, 64, start, 12)
        assert args[3] is runner.workspace.buffers["rotated_q"]
        assert args[5] is runner.workspace.buffers["scores"]


def test_chunk_scheduler_prefix_capture_and_restore(
    scheduled_runner: tuple[Runner, MagicMock], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Project once for a multi-chunk prefix, and once for any nonempty suffix.

    Args:
        scheduled_runner:
            Runner with real chunk scheduling and recorded device operations.

        monkeypatch:
            Replaces the device-to-device prefix copy.
    """
    runner, calls = scheduled_runner
    copy_state = MagicMock()
    monkeypatch.setattr(Runner, "_copy_prefix_state", copy_state)
    prefix = [1, 2, 3]
    saved = runner.cache_prefix(prefix)
    copy_state.assert_called_once_with(3, restore=False)
    assert runner._prefix_tokens == (1, 2, 3)
    runner.workspace.position = 0
    exact = runner.prefill_cached(prefix)
    np.testing.assert_array_equal(exact, saved)
    assert exact is not saved
    assert [call.args[0] for call in calls.run.call_args_list].count("lfm_last") == 1
    runner.workspace.position = 0
    result = runner.prefill_cached(prefix + [4, 5, 6])
    np.testing.assert_array_equal(result, saved)
    assert runner.workspace.position == 6
    assert [call.args[0] for call in calls.run.call_args_list].count("lfm_last") == 2
    assert copy_state.call_count == 3


@pytest.mark.parametrize("operation", ["prefill", "cache_prefix", "prefill_cached"])
def test_intermediate_chunk_failure_invalidates_runner_and_prefix(
    scheduled_runner: tuple[Runner, MagicMock],
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    """Poison state and discard a snapshot if an intermediate chunk fails.

    Args:
        scheduled_runner:
            Runner with real chunk scheduling and recorded device operations.

        monkeypatch:
            Replaces the device-to-device prefix copy.

        operation:
            Entry point whose intermediate execution is interrupted.
    """
    runner, calls = scheduled_runner
    runner._prefix_tokens = (1,)
    monkeypatch.setattr(Runner, "_copy_prefix_state", MagicMock())

    def fail_on_second_embedding(name: str, *args: Any) -> None:
        """Raise when the second chunk starts, after one state update.

        Args:
            name:
                Dispatched kernel name.

            *args:
                Kernel launch dimensions and arguments.
        """
        if (
            name == "lfm_embedding"
            and sum(
                call.args[0] == "lfm_embedding" for call in calls.run.call_args_list
            )
            == 2
        ):
            raise RuntimeError("intermediate device failure")

    calls.run.side_effect = fail_on_second_embedding
    tokens = [1, 2, 3, 4, 5, 6]
    with pytest.raises(RuntimeError, match="intermediate device failure"):
        getattr(runner, operation)(tokens)
    assert not runner.valid and runner._prefix_tokens is None
    assert runner.workspace.position == (3 if operation == "prefill_cached" else 2)
    assert not any(call.args[0] == "lfm_last" for call in calls.run.call_args_list)
    assert all(
        call.args[1] is not runner._logits_host
        for call in runner.cl.enqueue_copy.call_args_list
    )


def config() -> dict[str, object]:
    """Return the model configuration expected by the workspace planner."""
    return {
        "model_type": "lfm2",
        "hidden_size": 1024,
        "num_hidden_layers": 16,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "vocab_size": 65536,
        "conv_L_cache": 3,
        "conv_bias": False,
        "intermediate_size": 6656,
        "block_auto_adjust_ff_dim": True,
        "block_ffn_dim_multiplier": 1.0,
        "block_multiple_of": 256,
        "norm_eps": 1e-5,
        "tie_word_embeddings": True,
        "max_position_embeddings": 128000,
        "layer_types": [
            "full_attention" if i in (2, 5, 8, 10, 12, 14) else "conv"
            for i in range(16)
        ],
        "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "default"},
    }


def test_workspace_bounds_and_layer_allocation() -> None:
    """Reject out-of-bounds capacities and allocate each hybrid layer's state."""
    for capacity, chunk in ((0, 1), (4, 0), (4, 5), (128001, 1), (True, 1)):
        with pytest.raises(ValueError):
            WorkspaceSpec(capacity, chunk)
    sizes = WorkspaceSpec(13, 4).buffer_sizes
    assert sum(name.startswith("key_") for name in sizes) == 6
    assert sum(name.startswith("value_") for name in sizes) == 6
    assert sum(name.startswith("conv_") for name in sizes) == 10
    assert sizes["key_2"] == 13 * 512 * 4
    assert sizes["conv_0"] == 1024 * 3 * 4
    assert sizes["gate"] == 4 * 4608 * 4
    cached_sizes = WorkspaceSpec(13, 4, prefix_cache=True).buffer_sizes
    assert cached_sizes["prefix_key_2"] == sizes["key_2"]
    assert cached_sizes["prefix_value_14"] == sizes["value_14"]
    assert cached_sizes["prefix_conv_0"] == sizes["conv_0"]
    assert "prepared_weights" not in sizes
    assert (
        WorkspaceSpec(128, 128, True).buffer_sizes["prepared_weights"]
        == 9 * 1024 * 1024
    )
    with pytest.raises(TypeError, match="boolean"):
        WorkspaceSpec(128, 128, 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="boolean"):
        WorkspaceSpec(128, 128, prefix_cache=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="prepared_kn"):
        WorkspaceSpec(128, 128, True, prepared_kn=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="requires prepared_prefill"):
        WorkspaceSpec(128, 128, prepared_kn=True)
    with pytest.raises(TypeError, match="vec4_matvec"):
        WorkspaceSpec(128, 128, vec4_matvec=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="tiled_attention"):
        WorkspaceSpec(128, 128, tiled_attention=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="persistent_kn must"):
        WorkspaceSpec(128, 128, persistent_kn=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="budget"):
        WorkspaceSpec(128, 128, persistent_kn_budget_bytes=True)
    with pytest.raises(ValueError, match="requires prepared_kn"):
        WorkspaceSpec(128, 128, persistent_kn=True)
    with pytest.raises(ValueError, match="budget"):
        WorkspaceSpec(128, 128, True, prepared_kn=True, persistent_kn=True)
    with pytest.raises(ValueError, match="budget"):
        WorkspaceSpec(128, 128, persistent_kn_budget_bytes=PERSISTENT_KN_BYTES)
    persistent = WorkspaceSpec(
        128,
        128,
        True,
        prepared_kn=True,
        persistent_kn=True,
        persistent_kn_budget_bytes=PERSISTENT_KN_BYTES,
    )
    assert PERSISTENT_KN_BYTES == 574_619_648
    assert "prepared_weights" not in persistent.buffer_sizes
    assert (
        sum(persistent.buffer_sizes.values())
        - sum(WorkspaceSpec(128, 128, True, prepared_kn=True).buffer_sizes.values())
        == PERSISTENT_KN_BYTES - PREPARED_WEIGHT_BYTES
    )
    persistent_buffers = {
        name: size
        for name, size in persistent.buffer_sizes.items()
        if name.startswith("prepared_kn:")
    }
    assert len(persistent_buffers) == 92
    assert sum(persistent_buffers.values()) == PERSISTENT_KN_BYTES
    assert (
        WorkspaceSpec(128, 128, vec4_matvec=True).buffer_sizes
        == WorkspaceSpec(128, 128).buffer_sizes
    )
    assert (
        WorkspaceSpec(128, 128, True, prepared_kn=True).buffer_sizes["prepared_weights"]
        == PREPARED_WEIGHT_BYTES
    )


def test_prefix_buffer_copy_mapping() -> None:
    """Copy all hybrid states in the right direction and only the valid KV length."""
    runner = Runner.__new__(Runner)
    runner.workspace = Workspace.__new__(Workspace)
    spec = WorkspaceSpec(8, 4, prefix_cache=True, prefix_slots=2)
    runner.workspace.buffers = {name: name for name in spec.buffer_sizes}
    queue = SimpleNamespace(finish=lambda: None)
    runner.workspace.queue = queue
    copies: list[tuple[str, str, int]] = []

    def enqueue_copy(queue: Any, dest: str, src: str, *, byte_count: int) -> None:
        """Record a device-to-device state copy.

        Args:
            queue:
                Mock in-order command queue.

            dest:
                Destination buffer name.

            src:
                Source buffer name.

            byte_count:
                Copied byte count.
        """
        copies.append((dest, src, byte_count))

    runner.cl = SimpleNamespace(enqueue_copy=enqueue_copy)
    runner._copy_prefix_state(3, restore=False)
    assert len(copies) == 22
    assert ("prefix_key_2", "key_2", 3 * 512 * 4) in copies
    assert ("prefix_conv_0", "conv_0", 1024 * 3 * 4) in copies
    copies.clear()
    runner._copy_prefix_state(3, restore=True)
    assert len(copies) == 22
    assert ("key_2", "prefix_key_2", 3 * 512 * 4) in copies
    assert ("conv_0", "prefix_conv_0", 1024 * 3 * 4) in copies
    copies.clear()
    runner._copy_prefix_state(3, restore=False, slot=1)
    assert len(copies) == 22
    assert ("prefix1_key_2", "key_2", 3 * 512 * 4) in copies
    assert ("prefix1_conv_0", "conv_0", 1024 * 3 * 4) in copies
    copies.clear()
    runner._copy_prefix_state(3, restore=True, slot=1)
    assert ("key_2", "prefix1_key_2", 3 * 512 * 4) in copies
    assert ("conv_0", "prefix1_conv_0", 1024 * 3 * 4) in copies


def test_two_prefix_slots_independent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Capture and restore two independent states, logits, and positions."""
    runner = Runner.__new__(Runner)
    runner.workspace = Workspace.__new__(Workspace)
    runner.workspace.spec = WorkspaceSpec(16, 4, prefix_cache=True, prefix_slots=2)
    runner.workspace.position = 0
    runner.valid = True
    runner._lock = threading.Lock()
    runner._prefix_tokens = runner._extra_prefix_tokens = None
    runner._prefix_logits = np.empty(65536, dtype=np.float32)
    runner._extra_prefix_logits = np.empty(65536, dtype=np.float32)
    state: dict[str, Any] = {"live": [], "snapshots": {}}

    def execute(self: Runner, tokens: list[int]) -> np.ndarray:
        """Append token IDs to simulated live state and produce state-dependent logits.

        Args:
            self: Runner with simulated state.

            tokens: IDs appended to that state.
        """
        state["live"].extend(tokens)
        self.workspace.position += len(tokens)
        return np.full(65536, sum(state["live"]), dtype=np.float32)

    def copy(self: Runner, length: int, *, restore: bool, slot: int = 0) -> None:
        """Copy the simulated hybrid state to or from a selected snapshot.

        Args:
            self: Runner whose state is copied.

            length: Captured token count.

            restore: Whether to restore rather than capture.

            slot: Snapshot index.
        """
        if restore:
            state["live"] = list(state["snapshots"][slot])
        else:
            assert len(state["live"]) == length
            state["snapshots"][slot] = list(state["live"])

    monkeypatch.setattr(Runner, "_execute", execute)
    monkeypatch.setattr(Runner, "_copy_prefix_state", copy)
    runner.cache_prefix([1, 2])
    runner.workspace.position = 0
    state["live"] = []
    runner.cache_prefix([8, 9, 10], slot=1)
    for slot, prefix, expected in ((0, [1, 2], 3), (1, [8, 9, 10], 27)):
        runner.workspace.position = 0
        state["live"] = []
        assert runner.prefill_cached(prefix, slot=slot)[0] == expected
        assert runner.workspace.position == len(prefix)
        runner.workspace.position = 0
        state["live"] = []
        assert runner.prefill_cached(prefix + [4], slot=slot)[0] == expected + 4
        assert runner.workspace.position == len(prefix) + 1
    runner.workspace.position = 0
    before = list(state["live"])
    with pytest.raises(ValueError, match="does not match"):
        runner.prefill_cached([1, 2], slot=1)
    assert state["live"] == before
    assert runner._prefix_tokens == (1, 2)
    assert runner._extra_prefix_tokens == (8, 9, 10)


def test_prefix_identity_independence_and_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise snapshots with generated token IDs and a stateful mock backend.

    Args:
        monkeypatch:
            Replaces GPU operations with deterministic stateful functions.
    """
    runner = Runner.__new__(Runner)
    runner.workspace = Workspace.__new__(Workspace)
    runner.workspace.position = 0
    runner.workspace.spec = WorkspaceSpec(12, 4, prefix_cache=True)
    runner.valid = True
    runner._lock = threading.Lock()
    runner._prefix_tokens = None
    runner._prefix_logits = np.empty(65536, dtype=np.float32)
    state: dict[str, Any] = {"live": [], "saved": [], "fail": False}

    def chunk(self: Runner, tokens: list[int], *, final: bool) -> np.ndarray | None:
        """Append mock tokens and encode their accumulated state into logits.

        Args:
            self:
                Runner executing the mock chunk.

            tokens:
                IDs appended to its mock sequence.

            final:
                Whether to return logits for this chunk.
        """
        state["live"].extend(tokens)
        self.workspace.position += len(tokens)
        value = sum((i + 1) * t for i, t in enumerate(state["live"]))
        return np.full(65536, value, dtype=np.float32) if final else None

    def copy_state(self: Runner, length: int, *, restore: bool) -> None:
        """Copy or restore the mock sequence without sharing mutable lists.

        Args:
            self:
                Runner owning the snapshot.

            length:
                Number of token positions being copied.

            restore:
                Whether to restore the snapshot into live state.
        """
        if state["fail"]:
            raise RuntimeError("copy failure")
        if restore:
            state["live"] = state["saved"].copy()
        else:
            assert len(state["live"]) == length
            state["saved"] = state["live"].copy()

    def reset(self: Workspace) -> None:
        """Clear the mock live state while preserving its snapshot.

        Args:
            self:
                Workspace to reset.
        """
        self.position = 0
        state["live"] = []

    monkeypatch.setattr(Runner, "_chunk", chunk)
    monkeypatch.setattr(Runner, "_copy_prefix_state", copy_state)
    monkeypatch.setattr(Workspace, "reset", reset)
    rng = np.random.default_rng(17)
    prefix = rng.integers(1, 100, size=5).tolist()
    suffix = rng.integers(1, 100, size=3).tolist()
    expected = runner.prefill(prefix + suffix)
    runner.reset()
    saved_logits = runner.cache_prefix(prefix)
    saved_logits[:] = -1
    prefix[0] += 1
    runner.reset()
    with pytest.raises(ValueError, match="match"):
        runner.prefill_cached(prefix + suffix)
    prefix[0] -= 1
    assert runner.workspace.position == 0
    cached = runner.prefill_cached(prefix + suffix)
    np.testing.assert_array_equal(cached, expected)
    runner.reset()
    exact = runner.prefill_cached(prefix)
    assert exact[0] != -1
    exact[:] = -2
    runner.reset()
    assert runner.prefill_cached(prefix)[0] != -2
    runner.reset()
    state["fail"] = True
    with pytest.raises(RuntimeError, match="copy failure"):
        runner.prefill_cached(prefix)
    assert not runner.valid and runner._prefix_tokens is None
    state["fail"] = False
    runner.reset()
    with pytest.raises(RuntimeError, match="No valid"):
        runner.prefill_cached(prefix)
    runner.cache_prefix(prefix)
    runner.invalidate_prefix()
    runner.reset()
    with pytest.raises(RuntimeError, match="No valid"):
        runner.prefill_cached(prefix)
    state["fail"] = True
    with pytest.raises(RuntimeError, match="copy failure"):
        runner.cache_prefix(prefix)
    assert not runner.valid and runner._prefix_tokens is None
    runner.workspace.spec = WorkspaceSpec(12, 4)
    runner.reset()
    with pytest.raises(RuntimeError, match="not enabled"):
        runner.cache_prefix(prefix)
    with pytest.raises(RuntimeError, match="not enabled"):
        runner.prefill_cached(prefix)


def test_prepared_projection_dispatch_and_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check selection, ordering and hard scratch bounds without GPU execution.

    Args:
        monkeypatch:
            Replaces kernel dispatch with a recorder.
    """
    runner = Runner.__new__(Runner)
    runner.workspace = Workspace.__new__(Workspace)
    runner.workspace.spec = WorkspaceSpec(128, 128, True)
    scratch = object()
    runner.workspace.buffers = {"prepared_weights": scratch}
    q, z, scale = object(), object(), object()
    runner.weights = {"test": (q, z, scale, (4608, 1024))}
    calls: list[
        tuple[str, tuple[int, ...], tuple[int, ...] | None, tuple[Any, ...]]
    ] = []

    def record(
        self: Runner,
        name: str,
        global_size: tuple[int, ...],
        local_size: tuple[int, ...] | None,
        *args: Any,
    ) -> None:
        """Record a mock kernel invocation.

        Args:
            self:
                Runner issuing the kernel.

            name:
                Kernel name.

            global_size:
                Global dispatch dimensions.

            local_size:
                Local dispatch dimensions.

            *args:
                Positional kernel arguments.
        """
        calls.append((name, global_size, local_size, args))

    monkeypatch.setattr(Runner, "_run", record)
    runner._project("x", "y", "test", 128, 1024, 4608)
    assert [c[0] for c in calls] == ["lfm_prepare_u8_f16", "lfm_matmul_prepared_f16"]
    assert calls[0][3][3] is scratch and calls[1][3][1] is scratch
    assert PREPARED_WEIGHT_BYTES == 2 * 1024 * 4608
    calls.clear()
    runner._project("x", "y", "test", 1, 1024, 4608)
    assert [c[0] for c in calls] == ["lfm_matvec_u8_f16_dequant"]
    calls.clear()
    runner._project("x", "y", "test", 127, 1024, 4608)
    assert [c[0] for c in calls] == ["lfm_matmul_u8_f16_dequant"]
    runner.workspace.spec = WorkspaceSpec(128, 128, False)
    calls.clear()
    runner._project("x", "y", "test", 128, 1024, 4608)
    assert [c[0] for c in calls] == ["lfm_matmul_u8_f16_dequant"]
    runner.workspace.spec = WorkspaceSpec(128, 128, True)
    runner.weights["test"] = (q, z, scale, (4609, 1024))
    with pytest.raises(ValueError, match="scratch"):
        runner._project("x", "y", "test", 128, 1024, 4609)
    runner.workspace.spec = WorkspaceSpec(160, 160, True, prepared_kn=True)
    runner.weights["test"] = (q, z, scale, (47, 33))
    calls.clear()
    runner._project("x", "y", "test", 129, 33, 47)
    assert [c[0] for c in calls] == [
        "lfm_prepare_u8_f16_kn",
        "lfm_matmul_prepared_kn_f16",
    ]
    assert calls[0][1:3] == ((48, 48), (16, 16))
    assert calls[1][1:3] == ((16, 40), (8, 8))
    assert calls[0][3] == (q, z, scale, scratch, 33, 47)
    assert calls[1][3] == ("x", scratch, "y", 129, 33, 47)
    calls.clear()
    runner._project("x", "y", "test", 127, 33, 47)
    assert [c[0] for c in calls] == ["lfm_matmul_u8_f16_dequant"]
    calls.clear()
    runner._project("x", "y", "test", 1, 33, 47)
    assert [c[0] for c in calls] == ["lfm_matvec_u8_f16_dequant"]
    runner.weights["test"] = (q, z, scale, (4609, 1024))
    calls.clear()
    with pytest.raises(ValueError, match="scratch"):
        runner._project("x", "y", "test", 128, 1024, 4609)
    assert not calls

    runner.workspace.spec = WorkspaceSpec(
        160, 160, True, prepared_kn=True, vec4_matvec=True
    )
    runner.weights["test"] = (q, z, scale, (47, 33))
    runner._project("x", "y", "test", 1, 33, 47)
    assert [c[0] for c in calls] == ["lfm_matvec_u8_f16_dequant_vec4"]
    assert calls[0][1:3] == ((128, 6), (128, 1))
    assert calls[0][3] == ("x", q, z, scale, "y", 33, 47)
    calls.clear()
    runner._project("x", "y", "test", 2, 33, 47)
    assert [c[0] for c in calls] == ["lfm_matmul_u8_f16_dequant"]
    calls.clear()
    runner._project("x", "y", "test", 128, 33, 47)
    assert [c[0] for c in calls] == [
        "lfm_prepare_u8_f16_kn",
        "lfm_matmul_prepared_kn_f16",
    ]


@pytest.mark.parametrize("group_size,dimensions", [(128, (16, 16)), (256, (8, 16))])
def test_kn_preparation_rejects_unsupported_device(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    group_size: int,
    dimensions: tuple[int, int],
) -> None:
    """Reject a device that cannot launch the required preparation workgroup.

    Args:
        monkeypatch:
            Replaces artifact verification and the PyOpenCL API.

        tmp_path:
            Synthetic checkpoint directory.

        group_size:
            Maximum work-items in a device workgroup.

        dimensions:
            Maximum work-items on each of the first two workgroup axes.
    """
    api = SimpleNamespace(device_fp_config=SimpleNamespace(DENORM=1))
    monkeypatch.setitem(sys.modules, "pyopencl", api)
    monkeypatch.setattr(runner_module, "validate_config", MagicMock())
    monkeypatch.setattr(runner_module, "_verify_artifacts", MagicMock())
    device = SimpleNamespace(
        extensions="cl_khr_fp16",
        half_fp_config=1,
        max_work_group_size=group_size,
        max_work_item_sizes=dimensions,
    )
    with pytest.raises(RuntimeError, match="16x16 workgroups"):
        Runner(
            SimpleNamespace(device=device),
            tmp_path,
            WorkspaceSpec(128, 128, True, prepared_kn=True),
        )


@pytest.mark.parametrize(
    "option,group_size,dimensions",
    [
        ("vec4_matvec", 127, (128, 1, 1)),
        ("vec4_matvec", 128, (127, 1, 1)),
        ("vec4_matvec", 128, (128,)),
        ("tiled_attention", 63, (64, 1, 1)),
        ("tiled_attention", 64, (63, 1, 1)),
        ("tiled_attention", 64, (64, 1)),
    ],
)
def test_optional_kernel_rejects_device_workgroup_limits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    option: str,
    group_size: int,
    dimensions: tuple[int, ...],
) -> None:
    """Reject enabled kernel shapes before program compilation or allocations.

    Args:
        monkeypatch:
            Replaces artifact checks and the OpenCL API.

        tmp_path:
            Synthetic checkpoint directory.

        option:
            Enabled optional workspace feature.

        group_size:
            Maximum device-wide workgroup size.

        dimensions:
            Device per-axis work-item limits.
    """
    program = MagicMock()
    api = SimpleNamespace(device_fp_config=SimpleNamespace(DENORM=1), Program=program)
    monkeypatch.setitem(sys.modules, "pyopencl", api)
    monkeypatch.setattr(runner_module, "validate_config", MagicMock())
    monkeypatch.setattr(runner_module, "_verify_artifacts", MagicMock())
    device = SimpleNamespace(
        extensions="cl_khr_fp16",
        half_fp_config=1,
        max_work_group_size=group_size,
        max_work_item_sizes=dimensions,
    )
    spec = WorkspaceSpec(128, 128, **{option: True})
    with pytest.raises(RuntimeError, match="Device workgroup limits"):
        Runner(SimpleNamespace(device=device), tmp_path, spec)
    program.assert_not_called()


@pytest.mark.parametrize(
    "option,kernel_name,required",
    [
        ("vec4_matvec", "lfm_matvec_u8_f16_dequant_vec4", 128),
        ("tiled_attention", "lfm_gqa_scores_tiled_16_8_64", 64),
    ],
)
@pytest.mark.parametrize("kernel_limit_ok", [False, True])
def test_optional_kernel_checks_compiled_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    option: str,
    kernel_name: str,
    required: int,
    kernel_limit_ok: bool,
) -> None:
    """Check compiled-kernel capacity at the exact boundary before model loading.

    Args:
        monkeypatch:
            Replaces artifact checks and OpenCL compilation with mocks.

        tmp_path:
            Synthetic checkpoint directory.

        option:
            Enabled optional workspace feature.

        kernel_name:
            Name of the optional kernel under test.

        required:
            Required total local workgroup size.

        kernel_limit_ok:
            Whether the mocked compiled-kernel limit meets the requirement.
    """
    checked = MagicMock(return_value=required if kernel_limit_ok else required - 1)

    def kernel_factory(program: Any, name: str) -> Any:
        """Create a mock kernel exposing the selected compiled-kernel limit.

        Args:
            program:
                Mock compiled program.

            name:
                Requested kernel name.
        """
        return SimpleNamespace(
            get_work_group_info=checked if name == kernel_name else MagicMock()
        )

    program = MagicMock()
    api = SimpleNamespace(
        device_fp_config=SimpleNamespace(DENORM=1),
        kernel_work_group_info=SimpleNamespace(WORK_GROUP_SIZE=123),
        Program=program,
        Kernel=kernel_factory,
    )
    monkeypatch.setitem(sys.modules, "pyopencl", api)
    monkeypatch.setattr(runner_module, "validate_config", MagicMock())
    monkeypatch.setattr(runner_module, "_verify_artifacts", MagicMock())
    load = MagicMock(side_effect=RuntimeError("inventory reached"))
    monkeypatch.setattr(runner_module, "read_weights", load)
    device = SimpleNamespace(
        extensions="cl_khr_fp16",
        half_fp_config=1,
        max_work_group_size=256,
        max_work_item_sizes=(256, 256, 64),
    )
    queue = SimpleNamespace(device=device, context=object())
    message = "inventory reached" if kernel_limit_ok else "Insufficient workgroup size"
    with pytest.raises(RuntimeError, match=message):
        Runner(queue, tmp_path, WorkspaceSpec(128, 128, **{option: True}))
    checked.assert_called_once_with(123, device)
    assert load.call_count == int(kernel_limit_ok)


def test_disabled_optional_kernels_do_not_require_their_limits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Leave default construction independent of optional kernel limit checks.

    Args:
        monkeypatch:
            Replaces artifact checks and OpenCL compilation with mocks.

        tmp_path:
            Synthetic checkpoint directory.
    """
    kernel = MagicMock(return_value=SimpleNamespace())
    api = SimpleNamespace(
        device_fp_config=SimpleNamespace(DENORM=1), Program=MagicMock(), Kernel=kernel
    )
    monkeypatch.setitem(sys.modules, "pyopencl", api)
    monkeypatch.setattr(runner_module, "validate_config", MagicMock())
    monkeypatch.setattr(runner_module, "_verify_artifacts", MagicMock())
    monkeypatch.setattr(
        runner_module,
        "read_weights",
        MagicMock(side_effect=RuntimeError("inventory reached")),
    )
    device = SimpleNamespace(extensions="cl_khr_fp16", half_fp_config=1)
    with pytest.raises(RuntimeError, match="inventory reached"):
        Runner(
            SimpleNamespace(device=device, context=object()),
            tmp_path,
            WorkspaceSpec(8, 2),
        )
    names = [call.args[1] for call in kernel.call_args_list]
    assert "lfm_matvec_u8_f16_dequant_vec4" not in names
    assert "lfm_gqa_scores_tiled_16_8_64" not in names


def test_persistent_memory_limits_include_each_allocation() -> None:
    """Check persistent bytes against both individual and total device limits."""
    workspace = {"scratch": 30}
    model = [50, 20]
    persistent = {"first": 40, "second": 35}
    device = SimpleNamespace(max_mem_alloc_size=50, global_mem_size=175)
    _validate_memory_limits(device, workspace, model, persistent)
    device.global_mem_size = 174
    with pytest.raises(ValueError, match="global memory"):
        _validate_memory_limits(device, workspace, model, persistent)
    device.global_mem_size = 175
    device.max_mem_alloc_size = 39
    with pytest.raises(ValueError, match="allocation"):
        _validate_memory_limits(device, workspace, model, persistent)


def test_persistent_workspace_preallocates_and_fails_early(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allocate the complete persistent inventory during workspace construction.

    Args:
        monkeypatch:
            Replaces the OpenCL allocator and state clearing with CPU stubs.
    """
    spec = WorkspaceSpec(
        128,
        128,
        True,
        prepared_kn=True,
        persistent_kn=True,
        persistent_kn_budget_bytes=PERSISTENT_KN_BYTES,
    )
    allocations = MagicMock(side_effect=lambda *args: object())
    cl = SimpleNamespace(
        command_queue_properties=SimpleNamespace(OUT_OF_ORDER_EXEC_MODE_ENABLE=1),
        mem_flags=SimpleNamespace(READ_WRITE=1),
        Buffer=allocations,
        enqueue_fill_buffer=MagicMock(),
    )
    monkeypatch.setitem(sys.modules, "pyopencl", cl)
    queue = SimpleNamespace(
        properties=0,
        context=object(),
        device=SimpleNamespace(max_mem_alloc_size=1 << 30, global_mem_size=1 << 32),
        finish=MagicMock(),
    )
    workspace = Workspace(queue, spec)
    assert len(allocations.call_args_list) == len(spec.buffer_sizes)
    assert set(workspace.buffers) == set(spec.buffer_sizes)
    assert (
        sum(
            size
            for name, size in spec.buffer_sizes.items()
            if name.startswith("prepared_kn:")
        )
        == PERSISTENT_KN_BYTES
    )
    allocations.reset_mock()
    first_persistent = next(
        index
        for index, name in enumerate(spec.buffer_sizes)
        if name.startswith("prepared_kn:")
    )
    allocations.side_effect = [object()] * first_persistent + [
        MemoryError("device allocation failure")
    ]
    with pytest.raises(MemoryError, match="device allocation failure"):
        Workspace(queue, spec)
    assert allocations.call_count == first_persistent + 1
    queue.device.global_mem_size = sum(spec.buffer_sizes.values()) - 1
    allocations.reset_mock()
    with pytest.raises(ValueError, match="global memory"):
        Workspace(queue, spec)
    allocations.assert_not_called()


def test_persistent_projection_reuse_and_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reuse preallocated buffers and invalidate readiness on device failures.

    Args:
        monkeypatch:
            Replaces GPU dispatch and workspace reset with stubs.
    """
    runner = Runner.__new__(Runner)
    runner.workspace = Workspace.__new__(Workspace)
    runner.workspace.spec = WorkspaceSpec(
        128,
        128,
        prepared_prefill=True,
        prepared_kn=True,
        persistent_kn=True,
        persistent_kn_budget_bytes=PERSISTENT_KN_BYTES,
    )
    runner.workspace.position = 0
    runner.workspace.queue = SimpleNamespace(context=object())
    runner.workspace.buffers = {}
    runner.weights = {
        "first": (object(), object(), object(), (47, 33)),
        "second": (object(), object(), object(), (13, 11)),
    }
    runner._persistent_sizes = {"first": 47 * 33 * 2, "second": 13 * 11 * 2}
    first, second = object(), object()
    runner._prepared_weights = {"first": first, "second": second}
    runner._prepared_ready = set()
    runner.valid = True
    runner._lock = threading.Lock()
    runner._prefix_tokens = (1,)
    allocate = MagicMock(side_effect=lambda *args: object())
    runner.cl = SimpleNamespace(
        Buffer=allocate, mem_flags=SimpleNamespace(READ_WRITE=1)
    )
    dispatch = MagicMock()
    monkeypatch.setattr(Runner, "_run", dispatch)
    reset = MagicMock()
    monkeypatch.setattr(Workspace, "reset", reset)
    runner._project("x", "y", "first", 128, 33, 47)
    runner._project("x", "y", "first", 128, 33, 47)
    runner._project("x", "y", "second", 128, 11, 13)
    allocate.assert_not_called()
    assert runner._prepared_ready == {"first", "second"}
    assert [c.args[0] for c in dispatch.call_args_list] == [
        "lfm_prepare_u8_f16_kn",
        "lfm_matmul_prepared_kn_f16",
        "lfm_matmul_prepared_kn_f16",
        "lfm_prepare_u8_f16_kn",
        "lfm_matmul_prepared_kn_f16",
    ]
    assert dispatch.call_args_list[1].args[4] is first
    assert dispatch.call_args_list[2].args[4] is first
    runner.reset()
    assert runner._prepared_weights["first"] is first
    assert runner._prepared_ready == {"first", "second"}
    runner._persistent_sizes.pop("second")
    with pytest.raises(ValueError, match="Unexpected persistent"):
        runner._project("x", "y", "second", 128, 11, 13)
    runner._prepared_ready.clear()
    dispatch.side_effect = RuntimeError("prepare failure")
    with pytest.raises(RuntimeError, match="prepare failure"):
        runner._project("x", "y", "first", 128, 33, 47)
    assert not runner._prepared_ready
    assert runner._prepared_weights == {"first": first, "second": second}
    runner._prepared_ready.add("first")
    reset.side_effect = RuntimeError("reset failure")
    with pytest.raises(RuntimeError, match="reset failure"):
        runner.reset()
    assert not runner.valid and not runner._prepared_ready
    assert runner._prepared_weights == {"first": first, "second": second}
    assert runner._prefix_tokens is None
    reset.side_effect = None
    dispatch.side_effect = None
    runner.reset()
    runner._project("x", "y", "first", 128, 33, 47)
    assert runner._prepared_weights["first"] is first
    assert runner._prepared_ready == {"first"}
    allocate.assert_not_called()


@pytest.mark.parametrize("operation", ["prefill", "cache_prefix", "prefill_cached"])
def test_persistent_readiness_cleared_on_execution_failure(
    scheduled_runner: tuple[Runner, MagicMock],
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    """Invalidate prepared contents after a failed execution or prefix restore.

    Args:
        scheduled_runner:
            Runner with recorded CPU-only device operations.

        monkeypatch:
            Replaces snapshot copying with a deterministic stub.

        operation:
            Sequence entry point that experiences a device failure.
    """
    runner, calls = scheduled_runner
    runner.workspace.spec = WorkspaceSpec(
        12,
        2,
        prepared_prefill=True,
        prefix_cache=True,
        prepared_kn=True,
        persistent_kn=True,
        persistent_kn_budget_bytes=PERSISTENT_KN_BYTES,
    )
    retained = object()
    runner._prepared_weights = {"first": retained}
    runner._prepared_ready = {"first"}
    runner._prefix_tokens = (1,)
    monkeypatch.setattr(Runner, "_copy_prefix_state", MagicMock())
    calls.run.side_effect = RuntimeError("execution failure")
    with pytest.raises(RuntimeError, match="execution failure"):
        getattr(runner, operation)([1, 2, 3])
    assert not runner.valid and runner._prefix_tokens is None
    assert not runner._prepared_ready
    assert runner._prepared_weights == {"first": retained}


def test_artifact_identity_and_norm_shape(tmp_path: Path) -> None:
    """Reject a changed artifact and mismatched normalization metadata.

    Args:
        tmp_path:
            Temporary directory for a synthetic artifact.
    """
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"test")
    _verify_digest(path, hashlib.sha256(b"test").hexdigest())
    with pytest.raises(ValueError, match="digest"):
        _verify_digest(path, hashlib.sha256(b"other").hexdigest())
    _validate_norm("test", Constant("w", (1, 1, 1024), "f32", 0, 4096), 1024)
    for constant in (
        Constant("w", (1, 1, 1, 64), "f32", 0, 256),
        Constant("w", (1024,), "f32", 0, 4096),
        Constant("w", (1, 1, 1024), "f16", 0, 4096),
        Constant("w", (1, 1, 1024), "f32", 0, 256),
    ):
        with pytest.raises(ValueError, match="normalization"):
            _validate_norm("test", constant, 1024)


def test_artifact_pairs_cannot_be_mixed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject a model BIN from a different otherwise supported graph.

    Args:
        tmp_path:
            Temporary directory for synthetic graphs and weights.

        monkeypatch:
            Replaces the allowlist with synthetic fingerprint pairs.
    """
    xml = tmp_path / "model.xml"
    binary = tmp_path / "model.bin"
    xml.write_bytes(b"graph-a")
    binary.write_bytes(b"weights-b")
    pair_a = (
        hashlib.sha256(b"graph-a").hexdigest(),
        hashlib.sha256(b"weights-a").hexdigest(),
    )
    pair_b = (
        hashlib.sha256(b"graph-b").hexdigest(),
        hashlib.sha256(b"weights-b").hexdigest(),
    )
    monkeypatch.setattr("hoast.lfm_gpu.runner.ARTIFACT_DIGESTS", dict([pair_a, pair_b]))
    with pytest.raises(ValueError, match="digest"):
        _verify_artifacts(xml, binary)
    binary.write_bytes(b"weights-a")
    _verify_artifacts(xml, binary)
    xml.write_bytes(b"unknown")
    with pytest.raises(ValueError, match="graph digest"):
        _verify_artifacts(xml, binary)


def test_validate_config_rejects_wrong_and_missing_values(tmp_path: Path) -> None:
    """Require the complete expected layer schedule and dimensions.

    Args:
        tmp_path:
            Temporary directory containing the synthetic model configuration.
    """
    path = tmp_path / "config.json"
    data = config()
    path.write_text(json.dumps(data), encoding="utf-8")
    validate_config(path)
    del data["tie_word_embeddings"]
    data["tie_embedding"] = True
    path.write_text(json.dumps(data), encoding="utf-8")
    validate_config(path)
    data["tie_word_embeddings"] = False
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="tied embeddings"):
        validate_config(path)
    del data["tie_word_embeddings"]
    data["num_key_value_heads"] = 4
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="num_key_value_heads"):
        validate_config(path)
    del data["num_key_value_heads"]
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="num_key_value_heads"):
        validate_config(path)


def test_workspace_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise allocation and reset without a GPU.

    Args:
        monkeypatch:
            Replaces PyOpenCL with an in-memory allocation and fill recorder.
    """
    fills: list[tuple[Any, int]] = []
    buffers: list[Any] = []

    def allocate(context: Any, flags: int, size: int) -> Any:
        """Record one allocation and return an identifiable buffer.

        Args:
            context:
                Mock OpenCL context.

            flags:
                Mock allocation flags.

            size:
                Buffer size in bytes.
        """
        result = SimpleNamespace(size=size)
        buffers.append(result)
        return result

    def fill(queue: Any, buffer: Any, pattern: Any, offset: int, size: int) -> None:
        """Record a buffer fill without executing device code.

        Args:
            queue:
                Mock OpenCL queue.

            buffer:
                Mock destination buffer.

            pattern:
                Fill pattern.

            offset:
                Offset in bytes.

            size:
                Number of bytes to fill.
        """
        assert pattern == 0 and offset == 0
        fills.append((buffer, size))

    api = SimpleNamespace(
        command_queue_properties=SimpleNamespace(OUT_OF_ORDER_EXEC_MODE_ENABLE=1),
        mem_flags=SimpleNamespace(READ_WRITE=1),
        Buffer=allocate,
        enqueue_fill_buffer=fill,
    )
    monkeypatch.setitem(sys.modules, "pyopencl", api)
    device = SimpleNamespace(max_mem_alloc_size=1 << 30, global_mem_size=1 << 32)
    queue = SimpleNamespace(
        properties=0, context=object(), device=device, finish=lambda: None
    )
    workspace = Workspace(queue, WorkspaceSpec(8, 3))
    assert len(buffers) == len(workspace.spec.buffer_sizes)
    assert len(fills) == 10 and all(size == 1024 * 3 * 4 for _, size in fills)
    workspace.position = 4
    workspace.reset()
    assert workspace.position == 0 and len(fills) == 20
    queue.properties = 1
    with pytest.raises(ValueError, match="in-order"):
        Workspace(queue, WorkspaceSpec(8, 3))
    queue.properties = 0
    device.max_mem_alloc_size = 1
    with pytest.raises(ValueError, match="allocation limit"):
        Workspace(queue, WorkspaceSpec(8, 3))


def test_runner_rejects_invalid_sequence_before_device_work() -> None:
    """Reject invalid input and invalid state without enqueueing GPU work."""
    runner = Runner.__new__(Runner)
    runner.workspace = Workspace.__new__(Workspace)
    runner.workspace.position = 0
    runner.workspace.spec = WorkspaceSpec(4, 2)
    runner.valid = True
    runner._lock = threading.Lock()
    invalid_tokens: list[list[int]] = [[], [-1], [65536], [True], [1, 2, 3, 4, 5]]
    for tokens in invalid_tokens:
        with pytest.raises(ValueError):
            runner.prefill(tokens)
    with pytest.raises(RuntimeError, match="Prefill"):
        runner.decode(1)
    runner.valid = False
    with pytest.raises(RuntimeError, match="Reset"):
        runner.prefill([1])


def test_runner_execution_and_reset_failure_are_poisoned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require reset after failures and never mark a failed reset as valid.

    Args:
        monkeypatch:
            Replaces device operations with deterministic failures.
    """
    runner = Runner.__new__(Runner)
    runner.workspace = Workspace.__new__(Workspace)
    runner.workspace.position = 0
    runner.workspace.spec = WorkspaceSpec(4, 2)
    runner.valid = True
    runner._lock = threading.Lock()
    runner._prefix_tokens = (1,)

    def fail_chunk(self: Runner, tokens: list[int], *, final: bool) -> np.ndarray:
        """Raise a simulated device failure.

        Args:
            self:
                Runner whose chunk execution fails.

            tokens:
                Token IDs of the attempted chunk.

            final:
                Whether the attempted chunk would return logits.
        """
        raise RuntimeError("device failure")

    def fail_reset(self: Workspace) -> None:
        """Raise a simulated state-clear failure.

        Args:
            self:
                Workspace whose state clear fails.
        """
        raise RuntimeError("reset failure")

    monkeypatch.setattr(Runner, "_chunk", fail_chunk)
    monkeypatch.setattr(Workspace, "reset", fail_reset)
    with pytest.raises(RuntimeError, match="device failure"):
        runner.prefill([1])
    assert not runner.valid and runner._prefix_tokens is None
    runner._prefix_tokens = (1,)
    with pytest.raises(RuntimeError, match="reset failure"):
        runner.reset()
    assert not runner.valid and runner._prefix_tokens is None


@pytest.mark.parametrize("operation", ["prefill", "decode"])
@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_device_failure_discards_snapshot_and_reset_does_not_restore_it(
    monkeypatch: pytest.MonkeyPatch, operation: str, failure: type[BaseException]
) -> None:
    """Discard snapshot for execution errors or interruptions in either path.

    Args:
        monkeypatch:
            Replaces chunk execution and state clearing with mock functions.

        operation:
            Public inference entry point to exercise.

        failure:
            Error or interruption type raised by the mock GPU operation.
    """
    runner = Runner.__new__(Runner)
    runner.workspace = Workspace.__new__(Workspace)
    runner.workspace.spec = WorkspaceSpec(4, 2, prefix_cache=True)
    runner.workspace.position = 0 if operation == "prefill" else 1
    runner.valid = True
    runner._lock = threading.Lock()
    runner._prefix_tokens = (1,)

    def fail_chunk(self: Runner, tokens: list[int], *, final: bool) -> np.ndarray:
        """Interrupt a mock chunk.

        Args:
            self:
                Runner executing the mock chunk.

            tokens:
                Tokens for the mock chunk.

            final:
                Whether the attempted chunk would return logits.
        """
        raise failure("device failure")

    def reset(self: Workspace) -> None:
        """Reset the mock workspace.

        Args:
            self:
                Workspace whose position is cleared.
        """
        self.position = 0

    monkeypatch.setattr(Runner, "_chunk", fail_chunk)
    monkeypatch.setattr(Workspace, "reset", reset)
    with pytest.raises(failure, match="device failure"):
        if operation == "prefill":
            runner.prefill([1])
        else:
            runner.decode(1)
    assert not runner.valid and runner._prefix_tokens is None
    runner.reset()
    assert runner.valid and runner._prefix_tokens is None
    with pytest.raises(RuntimeError, match="No valid"):
        runner.prefill_cached([1])


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_reset_failure_discards_existing_snapshot(
    monkeypatch: pytest.MonkeyPatch, failure: type[BaseException]
) -> None:
    """Invalidate an existing snapshot if clearing live state does not complete.

    Args:
        monkeypatch:
            Replaces the workspace reset with a deterministic failure.

        failure:
            Error or interruption type raised by the mock reset.
    """
    runner = Runner.__new__(Runner)
    runner.workspace = Workspace.__new__(Workspace)
    runner.valid = True
    runner._lock = threading.Lock()
    runner._prefix_tokens = (1, 2)

    def fail_reset(self: Workspace) -> None:
        """Interrupt a mock reset.

        Args:
            self:
                Workspace whose reset is interrupted.
        """
        raise failure("reset failure")

    monkeypatch.setattr(Workspace, "reset", fail_reset)
    with pytest.raises(failure, match="reset failure"):
        runner.reset()
    assert not runner.valid and runner._prefix_tokens is None


def test_prefix_token_validation_and_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate exact capacity, token boundaries and mismatch without losing cache.

    Args:
        monkeypatch:
            Replaces chunk and snapshot-copy GPU work with deterministic stubs.
    """
    runner = Runner.__new__(Runner)
    runner.workspace = Workspace.__new__(Workspace)
    runner.workspace.position = 0
    runner.workspace.spec = WorkspaceSpec(4, 2, prefix_cache=True)
    runner.valid = True
    runner._lock = threading.Lock()
    runner._prefix_tokens = (3,)
    runner._prefix_logits = np.empty(65536, dtype=np.float32)

    def chunk(self: Runner, tokens: list[int], *, final: bool) -> np.ndarray | None:
        """Record an appended mock chunk and return independent logits.

        Args:
            self:
                Runner appending the mock chunk.

            tokens:
                Valid token IDs to append.

            final:
                Whether to return logits for this chunk.
        """
        self.workspace.position += len(tokens)
        return np.zeros(65536, dtype=np.float32) if final else None

    def copy_state(self: Runner, length: int, *, restore: bool) -> None:
        """Accept a mock snapshot transfer.

        Args:
            self:
                Runner executing the copy.

            length:
                Number of positions to copy.

            restore:
                Whether the copy restores live state.
        """

    def reset(self: Workspace) -> None:
        """Clear mock live sequence state.

        Args:
            self:
                Workspace whose position is cleared.
        """
        self.position = 0

    monkeypatch.setattr(Runner, "_chunk", chunk)
    monkeypatch.setattr(Runner, "_copy_prefix_state", copy_state)
    monkeypatch.setattr(Workspace, "reset", reset)
    invalid_prefixes: list[list[int]] = [[], [-1], [65536], [True], [1, 2, 3, 4, 5]]
    for invalid in invalid_prefixes:
        with pytest.raises(ValueError):
            runner.cache_prefix(invalid)
        assert runner.valid and runner._prefix_tokens == (3,)
    with pytest.raises(TypeError):
        runner.cache_prefix((0,))  # type: ignore[arg-type]
    runner.cache_prefix([0, 65535, 2, 3])
    runner.reset()
    for invalid in ([0], [0, 65535, 2, 4], [0, 65535, 2, 3, 4], [0, 65535, 2, True]):
        with pytest.raises(ValueError):
            runner.prefill_cached(invalid)
        assert runner.valid and runner._prefix_tokens == (0, 65535, 2, 3)
    runner.prefill_cached([0, 65535, 2, 3])
    assert runner.workspace.position == 4
    with pytest.raises(ValueError, match="capacity"):
        runner.decode(0)
    assert runner.valid and runner._prefix_tokens == (0, 65535, 2, 3)


def test_reset_waits_for_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serialize reset and prefix invalidation against in-flight inference.

    Args:
        monkeypatch:
            Replaces GPU execution and state clearing with synchronization fixtures.
    """
    runner = Runner.__new__(Runner)
    runner.workspace = Workspace.__new__(Workspace)
    runner.workspace.position = 0
    runner.workspace.spec = WorkspaceSpec(4, 2)
    runner.valid = True
    runner._lock = threading.Lock()
    runner._prefix_tokens = (42,)
    entered = threading.Event()
    release = threading.Event()
    reset_finished = threading.Event()
    invalidate_finished = threading.Event()
    errors: list[BaseException] = []

    def chunk(self: Runner, tokens: list[int], *, final: bool) -> np.ndarray | None:
        """Wait until the test permits the mock chunk to finish.

        Args:
            self:
                Runner executing the mock chunk.

            tokens:
                Token IDs for the mock chunk.

            final:
                Whether to return logits for this chunk.
        """
        entered.set()
        if not release.wait(5):
            raise RuntimeError("Timed out waiting for test release")
        self.workspace.position += len(tokens)
        return np.zeros(1, dtype=np.float32) if final else None

    def reset(self: Workspace) -> None:
        """Clear the mock workspace position.

        Args:
            self:
                Workspace receiving the mock reset.
        """
        self.position = 0

    def infer() -> None:
        """Run the mock inference and record thread failures."""
        try:
            runner.prefill([1])
        except RuntimeError as error:
            errors.append(error)

    def reset_runner() -> None:
        """Reset the runner and record thread failures."""
        try:
            runner.reset()
            reset_finished.set()
        except RuntimeError as error:
            errors.append(error)

    def invalidate_runner() -> None:
        """Invalidate the prefix and record thread failures."""
        try:
            runner.invalidate_prefix()
            invalidate_finished.set()
        except RuntimeError as error:
            errors.append(error)

    monkeypatch.setattr(Runner, "_chunk", chunk)
    monkeypatch.setattr(Workspace, "reset", reset)
    inference = threading.Thread(target=infer)
    resetting = threading.Thread(target=reset_runner)
    invalidating = threading.Thread(target=invalidate_runner)
    inference.start()
    try:
        assert entered.wait(5)
        resetting.start()
        invalidating.start()
        assert not reset_finished.wait(0.05)
        assert not invalidate_finished.wait(0.05)
    finally:
        release.set()
        inference.join(5)
        if resetting.ident is not None:
            resetting.join(5)
        if invalidating.ident is not None:
            invalidating.join(5)
    assert (
        not inference.is_alive()
        and not resetting.is_alive()
        and not invalidating.is_alive()
    )
    assert not errors and reset_finished.is_set() and invalidate_finished.is_set()
    assert (
        runner.workspace.position == 0
        and runner.valid
        and runner._prefix_tokens is None
    )


@pytest.mark.skipif(
    os.environ.get("LFM_GPU_RUNNER_TEST") != "1",
    reason="explicit GPU and IR fixture required",
)
def test_real_ir_prefill_decode_and_reset() -> None:
    """Compare GPU logits and greedy tokens with the explicitly supplied CPU IR.

    LFM_GPU_CHECKPOINT identifies either supported checkpoint directory. The
    reference is computed using the exact same IR and weights on the CPU.
    """
    import openvino as ov  # Optional dependency for explicitly enabled IR parity.
    import pyopencl as cl  # Optional native dependency for the explicitly enabled test.

    checkpoint = Path(os.environ["LFM_GPU_CHECKPOINT"])
    devices = [
        d
        for p in cl.get_platforms()
        for d in p.get_devices(device_type=cl.device_type.GPU)
    ]
    match = os.environ.get("LFM_GPU_DEVICE", "").lower()
    devices = [d for d in devices if match in d.name.lower()]
    if len(devices) != 1:
        raise RuntimeError(
            f"Expected one explicitly selected GPU, found {len(devices)}"
        )
    runner = Runner(
        cl.CommandQueue(cl.Context(devices)), checkpoint, WorkspaceSpec(4, 2)
    )
    request = (
        ov.Core()
        .compile_model(
            str(checkpoint / "openvino_model.xml"),
            "CPU",
            {"INFERENCE_PRECISION_HINT": "f32"},
        )
        .create_infer_request()
    )
    request.reset_state()
    actual0, actual1 = runner.prefill([1]), runner.decode(2)
    for index, actual in enumerate((actual0, actual1)):
        request.infer(
            {
                "input_ids": np.array([[index + 1]], dtype=np.int64),
                "attention_mask": np.ones((1, index + 1), dtype=np.int64),
                "beam_idx": np.array([0], dtype=np.int32),
            }
        )
        expected = np.asarray(request.get_tensor("logits").data).reshape(-1, 65536)[-1]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0.04)
        assert np.argmax(actual) == np.argmax(expected)
    runner.reset()
    with pytest.raises(ValueError, match="capacity"):
        runner.prefill([1, 2, 3, 4, 5])
    np.testing.assert_array_equal(runner.prefill([1]), actual0)

    # Compare greedy continuation against the unmodified, stateful CPU IR.
    request.reset_state()
    runner.reset()
    request.infer(
        {
            "input_ids": np.array([[1, 2]], dtype=np.int64),
            "attention_mask": np.ones((1, 2), dtype=np.int64),
            "beam_idx": np.array([0], dtype=np.int32),
        }
    )
    reference_prefill = np.asarray(request.get_tensor("logits").data).reshape(
        -1, 65536
    )[-1]
    np.testing.assert_allclose(
        runner.prefill([1, 2]), reference_prefill, rtol=0, atol=0.04
    )
    request.reset_state()
    runner.reset()
    token = 1
    for step in range(4):
        request.infer(
            {
                "input_ids": np.array([[token]], dtype=np.int64),
                "attention_mask": np.ones((1, step + 1), dtype=np.int64),
                "beam_idx": np.array([0], dtype=np.int32),
            }
        )
        reference_logits = np.asarray(request.get_tensor("logits").data).reshape(
            -1, 65536
        )[-1]
        gpu_logits = runner.prefill([token]) if step == 0 else runner.decode(token)
        next_reference, next_gpu = (
            int(np.argmax(reference_logits)),
            int(np.argmax(gpu_logits)),
        )
        assert next_reference == next_gpu
        token = next_reference


@pytest.mark.skipif(
    os.environ.get("LFM_GPU_PREPARED_TEST") != "1",
    reason="explicit isolated GPU and supported IR required",
)
@pytest.mark.parametrize(
    "prepared_kn,persistent_kn", [(False, False), (True, False), (True, True)]
)
def test_prepared_prefill_against_direct(
    prepared_kn: bool, persistent_kn: bool
) -> None:
    """Compare long prepared prefill and decode with the direct projection path.

    LFM_GPU_CHECKPOINT selects a supported checkpoint. The same preallocated
    runner and device weights are reused for both modes, with state reset.

    Args:
        prepared_kn:
            Whether to use K-major prepared weights and blocked GEMM.

        persistent_kn:
            Whether to retain prepared K-major weights across resets.
    """
    import pyopencl as cl  # Only required for the opt-in isolated GPU test.

    devices = [
        d
        for platform in cl.get_platforms()
        for d in platform.get_devices(device_type=cl.device_type.GPU)
    ]
    match = os.environ.get("LFM_GPU_DEVICE", "").lower()
    devices = [d for d in devices if match in d.name.lower()]
    if len(devices) != 1:
        raise RuntimeError(
            f"Expected one explicitly selected GPU, found {len(devices)}"
        )
    spec = WorkspaceSpec(
        129,
        128,
        True,
        prepared_kn=prepared_kn,
        persistent_kn=persistent_kn,
        persistent_kn_budget_bytes=PERSISTENT_KN_BYTES if persistent_kn else 0,
    )
    runner = Runner(
        cl.CommandQueue(cl.Context(devices)),
        Path(os.environ["LFM_GPU_CHECKPOINT"]),
        spec,
    )
    tokens = [1, 2, 3, 4] * 32
    runner.workspace.spec = WorkspaceSpec(129, 128, False)
    direct = runner.prefill(tokens)
    direct_decode = runner.decode(5)
    runner.reset()
    runner.workspace.spec = spec
    prepared = runner.prefill(tokens)
    prepared_decode = runner.decode(5)
    np.testing.assert_allclose(prepared, direct, rtol=0, atol=0.05)
    np.testing.assert_allclose(prepared_decode, direct_decode, rtol=0, atol=0.05)
    assert np.argmax(prepared) == np.argmax(direct)
    assert np.argmax(prepared_decode) == np.argmax(direct_decode)
    if persistent_kn:
        assert len(runner._prepared_weights) == 92
        assert len(runner._prepared_ready) == 92
        assert sum(runner._persistent_sizes.values()) == PERSISTENT_KN_BYTES
        previous_buffers = runner._prepared_weights.copy()
        runner.reset()
        repeated = runner.prefill(tokens)
        np.testing.assert_array_equal(repeated, prepared)
        assert all(
            runner._prepared_weights[name] is buffer
            for name, buffer in previous_buffers.items()
        )


@pytest.mark.skipif(
    os.environ.get("LFM_GPU_PREFIX_TEST") != "1",
    reason="explicit isolated GPU and supported IR required",
)
def test_real_prefix_snapshot_against_uncached() -> None:
    """Compare independent cached and uncached prefills and greedy continuations.

    LFM_GPU_CHECKPOINT selects a supported checkpoint. Tests include the exact
    prefix, multiple suffixes, resets, and mutation of returned host logits.
    """
    import pyopencl as cl  # Optional dependency for the explicitly enabled GPU test.

    devices = [
        d
        for p in cl.get_platforms()
        for d in p.get_devices(device_type=cl.device_type.GPU)
    ]
    match = os.environ.get("LFM_GPU_DEVICE", "").lower()
    devices = [d for d in devices if match in d.name.lower()]
    if len(devices) != 1:
        raise RuntimeError(
            f"Expected one explicitly selected GPU, found {len(devices)}"
        )
    runner = Runner(
        cl.CommandQueue(cl.Context(devices)),
        Path(os.environ["LFM_GPU_CHECKPOINT"]),
        WorkspaceSpec(32, 8, prefix_cache=True),
    )
    prefix = [1, 2, 3]
    prompts = [prefix, prefix + [4, 5], prefix + [6, 7, 8, 9]]
    references: list[list[np.ndarray]] = []
    for prompt in prompts:
        runner.reset()
        logits = runner.prefill(prompt)
        sequence = [logits]
        for _ in range(3):
            logits = runner.decode(int(np.argmax(logits)))
            sequence.append(logits)
        references.append(sequence)
    runner.reset()
    returned = runner.cache_prefix(prefix)
    returned[:] = 0
    for prompt, expected in zip(prompts, references, strict=True):
        runner.reset()
        logits = runner.prefill_cached(prompt)
        for step in range(4):
            np.testing.assert_allclose(logits, expected[step], rtol=0, atol=0.05)
            assert int(np.argmax(logits)) == int(np.argmax(expected[step]))
            if step < 3:
                logits = runner.decode(int(np.argmax(logits)))
    runner.invalidate_prefix()
    runner.reset()
    with pytest.raises(RuntimeError, match="No valid"):
        runner.prefill_cached(prefix)


@pytest.mark.skipif(
    os.environ.get("LFM_GPU_VEC4_TEST") != "1"
    and os.environ.get("LFM_GPU_TILED_TEST") != "1",
    reason="explicit isolated GPU and supported IR required",
)
@pytest.mark.parametrize("tiled_attention", [False, True])
def test_vec4_matvec_against_scalar_with_prefix_cache(tiled_attention: bool) -> None:
    """Compare scalar and vec4 prefill, cached prefixes and greedy decode.

    The 129-token prefix includes a prepared K-major chunk and a single-token
    tail. Exact and extended prefix restores exercise stored prefix logits,
    single-row and multi-row suffixes, and greedy autoregressive continuation.
    LFM_GPU_CHECKPOINT and LFM_GPU_DEVICE select an explicitly isolated device.

    Args:
        tiled_attention:
            Whether to combine vectorized matvec with tiled attention scores.
    """
    import pyopencl as cl  # Optional dependency for explicitly enabled GPU tests.

    devices = [
        device
        for platform in cl.get_platforms()
        for device in platform.get_devices(device_type=cl.device_type.GPU)
    ]
    match = os.environ.get("LFM_GPU_DEVICE", "").lower()
    devices = [device for device in devices if match in device.name.lower()]
    if not match or len(devices) != 1:
        raise RuntimeError(
            f"Expected one explicitly selected GPU, found {len(devices)}"
        )
    vec_spec = WorkspaceSpec(
        160,
        128,
        prepared_prefill=True,
        prefix_cache=True,
        prepared_kn=True,
        vec4_matvec=True,
        tiled_attention=tiled_attention,
    )
    scalar_spec = WorkspaceSpec(
        160, 128, prepared_prefill=True, prefix_cache=True, prepared_kn=True
    )
    runner = Runner(
        cl.CommandQueue(cl.Context(devices)),
        Path(os.environ["LFM_GPU_CHECKPOINT"]),
        vec_spec,
    )
    prefix = [1, 2, 3, 4] * 32 + [5]
    prompts = [prefix, prefix + [6], prefix + [6, 7, 8]]
    results: dict[str, list[list[np.ndarray]]] = {}
    for label, spec in (("scalar", scalar_spec), ("vec4", vec_spec)):
        runner.workspace.spec = spec
        runner.reset()
        prefix_logits = runner.cache_prefix(prefix)
        sequences: list[list[np.ndarray]] = []
        for prompt in prompts:
            runner.reset()
            logits = runner.prefill_cached(prompt)
            if prompt == prefix:
                np.testing.assert_array_equal(logits, prefix_logits)
            sequence = [logits]
            for _ in range(3):
                logits = runner.decode(int(np.argmax(logits)))
                sequence.append(logits)
            sequences.append(sequence)
        runner.reset()
        uncached = runner.prefill(prompts[-1])
        np.testing.assert_allclose(uncached, sequences[-1][0], rtol=0, atol=0.05)
        assert np.argmax(uncached) == np.argmax(sequences[-1][0])
        results[label] = sequences
    for scalar_sequence, vec_sequence in zip(
        results["scalar"], results["vec4"], strict=True
    ):
        for scalar, vector in zip(scalar_sequence, vec_sequence, strict=True):
            np.testing.assert_allclose(vector, scalar, rtol=0, atol=0.05)
            assert np.argmax(vector) == np.argmax(scalar)
