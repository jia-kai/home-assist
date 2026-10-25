"""Offline regression checks for aggregate speech CPU affinity."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hoast.runtime import configure_cpu_budget


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
