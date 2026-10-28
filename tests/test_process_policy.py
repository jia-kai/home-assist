"""Model-free checks of the early process THP policy and entrypoint ordering."""

import errno
import logging
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hoast import __main__ as entrypoint
from hoast import process_policy


@pytest.mark.parametrize("results", [[0, 1], [0, 0], [-1], [0, -1]])
def test_thp_policy_syscalls(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    results: list[int],
) -> None:
    """Require successful disable and exact readback, preserving syscall errno.

    Args:
        monkeypatch:
            Replaces libc calls without modifying the test process's THP policy.

        caplog:
            Captures the confirmed process policy, not just an attempted action.

        results:
            Disable result followed by readback result when disable succeeds.

    """
    prctl = Mock(side_effect=results)
    library = Mock(return_value=SimpleNamespace(prctl=prctl))
    monkeypatch.setattr(process_policy.sys, "platform", "linux")
    monkeypatch.setattr(process_policy.ctypes, "CDLL", library)
    monkeypatch.setattr(process_policy.ctypes, "get_errno", lambda: errno.EPERM)
    with caplog.at_level(logging.INFO):
        if -1 in results:
            with pytest.raises(OSError) as failure:
                process_policy.disable_transparent_huge_pages()
            assert failure.value.errno == errno.EPERM
        elif results[-1] != 1:
            with pytest.raises(RuntimeError, match="readback: 0"):
                process_policy.disable_transparent_huge_pages()
        else:
            process_policy.disable_transparent_huge_pages()
    library.assert_called_once_with(None, use_errno=True)
    assert prctl.call_args_list[0].args == (41, 1, 0, 0, 0)
    if len(results) > 1:
        assert prctl.call_args_list[1].args == (42, 0, 0, 0, 0)
    assert ("memory.thp status=disabled" in caplog.text) == (results == [0, 1])


def test_thp_policy_requires_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail explicitly on unsupported platforms instead of pretending to opt out.

    Args:
        monkeypatch:
            Supplies a platform without Linux prctl.

    """
    monkeypatch.setattr(process_policy.sys, "platform", "other")
    with pytest.raises(RuntimeError, match="requires Linux"):
        process_policy.disable_transparent_huge_pages()


def test_entrypoint_stops_on_policy_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never enter the application if checked process memory policy fails.

    Args:
        monkeypatch:
            Replaces policy and the application without loading native models.

    """
    application = Mock()
    monkeypatch.setattr(entrypoint, "configure_logging", Mock())
    monkeypatch.setattr(
        entrypoint,
        "disable_transparent_huge_pages",
        Mock(side_effect=OSError(errno.EPERM, "denied")),
    )
    monkeypatch.setitem(sys.modules, "hoast.llm_cli", SimpleNamespace(main=application))
    with pytest.raises(SystemExit) as failure:
        entrypoint.main()
    assert failure.value.code == 1
    application.assert_not_called()


def test_entrypoint_applies_policy_before_heavy_imports() -> None:
    """Check import ordering in a fresh interpreter without numerical libraries."""
    script = """
import importlib.abc
import importlib.util
import sys
from hoast import __main__ as entrypoint
assert not any(name in sys.modules for name in ("numpy", "openvino", "hoast.llm_cli"))
events = []
entrypoint.configure_logging = lambda: None
entrypoint.disable_transparent_huge_pages = lambda: events.append("policy")
class Application(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        if fullname == "hoast.llm_cli":
            assert events == ["policy"]
            events.append("import")
            return importlib.util.spec_from_loader(fullname, self)
    def create_module(self, spec):
        return None
    def exec_module(self, module):
        module.main = lambda: events.append("run")
sys.meta_path.insert(0, Application())
entrypoint.main()
assert events == ["policy", "import", "run"]
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process policy")
def test_real_policy_in_disposable_process() -> None:
    """Verify the actual syscall without changing the test runner or host policy."""
    script = """
from pathlib import Path
from hoast.process_policy import disable_transparent_huge_pages
disable_transparent_huge_pages()
status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines())
assert status["THP_enabled"].strip() == "0"
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
