"""Offline regression checks for speech fusion guards and aggregate CPU affinity."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import openvino as ov
import pytest
from openvino import opset13 as ops

from hoast.runtime import configure_cpu_budget
from hoast.tts_kernels import KERNEL_SOURCE, fuse_cpu_activations, snake_inputs


def test_fusion_requires_exact_data_and_exponent() -> None:
    """Only fuse a genuine Snake expression, including converted scalar exponents."""
    x = ops.parameter([1, 3, 3], np.float32)
    y = ops.parameter([1, 3, 3], np.float32)
    alpha = ops.constant(np.float32(0.5))
    scale = ops.constant(np.float32(2))
    square = ops.power(
        ops.sin(ops.multiply(alpha, x)), ops.convert(ops.constant(2), "f32")
    )
    assert snake_inputs(ops.add(ops.multiply(square, scale), x)) is not None
    assert snake_inputs(ops.add(ops.multiply(square, scale), y)) is None
    cube = ops.power(ops.sin(ops.multiply(alpha, x)), ops.constant(np.float32(3)))
    assert snake_inputs(ops.add(x, ops.multiply(scale, cube))) is None
    time_scale = ops.constant(np.ones(3, dtype=np.float32))
    assert snake_inputs(ops.add(x, ops.multiply(time_scale, square))) is None


def test_corrupt_kernel_never_loads(tmp_path: Path) -> None:
    """Reject a mismatched library checksum before native extension loading.

    Args:
        tmp_path:
            Synthetic library and matching source/version manifest.

    """
    library = tmp_path / "kernel.so"
    core = MagicMock()
    model = MagicMock()
    assert fuse_cpu_activations(core, model, library) == 0
    library.write_bytes(b"corrupt")
    library.with_suffix(".json").write_text(
        json.dumps(
            {
                "openvino": ov.__version__,
                "source_sha256": hashlib.sha256(KERNEL_SOURCE.read_bytes()).hexdigest(),
                "library_sha256": hashlib.sha256(b"expected").hexdigest(),
            }
        )
    )
    with pytest.raises(ValueError, match="checksum"):
        fuse_cpu_activations(core, model, library)
    core.add_extension.assert_not_called()


def test_budget_bounds_existing_threads_and_reconfiguration() -> None:
    """Keep existing native threads within the initial mask across budget changes."""
    with (
        patch("hoast.runtime._initial_affinity", None),
        patch("hoast.runtime._limits", []),
        patch("hoast.runtime.os.environ", {}),
        patch("hoast.runtime.os.sched_getaffinity", return_value={3, 4, 5}),
        patch("hoast.runtime.os.sched_setaffinity") as affinity,
        patch("hoast.runtime.psutil.Process") as process,
        patch("hoast.runtime.threadpool_limits"),
        patch("hoast.runtime.ThreadpoolController"),
    ):
        process.return_value.threads.return_value = [
            SimpleNamespace(id=11),
            SimpleNamespace(id=12),
        ]
        assert configure_cpu_budget(1) == (5,)
        assert all(call.args[1] == (5,) for call in affinity.call_args_list)
        affinity.reset_mock()
        assert configure_cpu_budget(2) == (4, 5)
        assert all(call.args[1] == (4, 5) for call in affinity.call_args_list)
        assert {call.args[0] for call in affinity.call_args_list} == {0, 11, 12}
        with pytest.raises(ValueError):
            configure_cpu_budget(3)
