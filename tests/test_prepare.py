"""Offline bootstrap orchestration and preparation publication regressions."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools.export_chinese_tts import (
    REPOSITORY,
    REVISION,
    WEIGHT_SHA256,
    cached,
    publish,
)
from tools.guarded_run import run_guarded

_BASH = shutil.which("bash")

_UV = """import json, os, pathlib, shutil, sys
with open(os.environ["TRACE"], "a") as log:
    log.write(json.dumps({"tool": "uv", "args": sys.argv[1:], "cwd": os.getcwd()}) + "\\n")
if sys.argv[1] == "sync":
    destination = pathlib.Path(os.environ["UV_PROJECT_ENVIRONMENT"]) / "bin/python"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(os.environ["FAKE_PYTHON"], destination)
    destination.chmod(0o755)
"""
_PYTHON = """import json, os, pathlib, sys
args = sys.argv[1:]
with open(os.environ["TRACE"], "a") as log:
    log.write(json.dumps({"tool": "python", "args": args, "cwd": os.getcwd()}) + "\\n")
if os.environ.get("FAIL_STAGE") in args:
    path = pathlib.Path(args[args.index("--log-file") + 1])
    path.write_text("simulated model failure\\n")
    sys.exit(7)
"""


def executable(path: Path, code: str) -> None:
    """Create a tiny Python-backed executable fixture.

    Args:
        path:
            Destination inside a temporary test directory.

        code:
            Embedded fixture source using only the standard library.

    """
    path.write_text(f"#!{sys.executable}\n{code}")
    path.chmod(0o755)


def bootstrap_fixture(tmp_path: Path, uv_present: bool) -> tuple[Path, dict[str, str]]:
    """Provide isolated executables so bootstrap tests cannot install or download models.

    Args:
        tmp_path:
            Isolated test directory.

        uv_present:
            Whether uv is initially discoverable on the test PATH.

    """
    root = tmp_path / "repo with spaces"
    root.mkdir()
    (root / "prepare.sh").write_bytes(
        (Path(__file__).resolve().parents[1] / "prepare.sh").read_bytes()
    )
    (root / ".python-version").write_text("3.14\n")
    (root / "config.toml").write_text("keep this configuration\n")
    (root / ".env").write_text("FIXTURE_TOKEN=keep-this-value\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("dirname", "date", "mkdir", "sh", "cp", "chmod"):
        source = shutil.which(name)
        assert source is not None
        (bin_dir / name).symlink_to(source)
    for name in ("c++", "systemd-run", "systemctl", "taskset"):
        executable(bin_dir / name, "")
    uv = tmp_path / "uv-template"
    python = tmp_path / "python-template"
    executable(uv, _UV)
    executable(python, _PYTHON)
    if uv_present:
        (bin_dir / "uv").symlink_to(uv)
    executable(
        bin_dir / "curl",
        """import pathlib, sys
destination = pathlib.Path(sys.argv[sys.argv.index("--output") + 1])
destination.write_text('mkdir -p "$UV_INSTALL_DIR"\\ncp "$FAKE_UV" "$UV_INSTALL_DIR/uv"\\nchmod +x "$UV_INSTALL_DIR/uv"\\n')
""",
    )
    environment = os.environ.copy()
    environment.pop("UV_INSTALL_DIR", None)
    environment.pop("FAIL_STAGE", None)
    environment.update(
        PATH=str(bin_dir),
        HOME=str(tmp_path / "home"),
        TRACE=str(tmp_path / "trace.jsonl"),
        FAKE_UV=str(uv),
        FAKE_PYTHON=str(python),
    )
    return root, environment


@pytest.mark.parametrize(
    "uv_present,extra", [(True, []), (False, []), (True, ["--with-functiongemma"])]
)
def test_bootstrap_models_and_paths(
    tmp_path: Path, uv_present: bool, extra: list[str]
) -> None:
    """Prepare ordered, guarded stages from another directory without changing user config.

    Args:
        tmp_path:
            Isolated test directory.

        uv_present:
            Whether to exercise the installation path.

        extra:
            Optional alternative-model flag.

    """
    root, environment = bootstrap_fixture(tmp_path, uv_present)
    assert _BASH is not None
    result = subprocess.run(
        [_BASH, str(root / "prepare.sh"), *extra],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    records = [
        json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()
    ]
    assert all(row["cwd"] == str(root) for row in records)
    assert records[0]["args"] == ["python", "install", "3.14"]
    assert records[1]["args"] == ["sync", "--locked", "--python", "3.14"]
    stages = records[2:]
    names = [
        Path(row["args"][row["args"].index("--log-file") + 1]).stem for row in stages
    ]
    expected = ["gpu-check", "lfm-download", "lfm-gpu", "lfm-cpu", "stt", "tts"]
    if extra:
        expected += ["functiongemma-download", "functiongemma-export"]
    assert names == [*expected, "speech-check"]
    for row in stages:
        args = row["args"]
        assert args[:2] == ["-m", "tools.guarded_run"]
        assert args[args.index("--threads") + 1] in ("1", "2")
        assert args[args.index("--memory-gib") + 1] == "4"
    assert "--chinese" in stages[5]["args"] and "--chinese" in stages[-1]["args"]
    assert (root / "config.toml").read_text() == "keep this configuration\n"
    assert (root / ".env").read_text() == "FIXTURE_TOKEN=keep-this-value\n"


def test_bootstrap_stops_after_failure(tmp_path: Path) -> None:
    """Propagate a model failure and leave later preparation stages unexecuted.

    Args:
        tmp_path:
            Isolated test directory.

    """
    root, environment = bootstrap_fixture(tmp_path, True)
    environment["FAIL_STAGE"] = "tools.prepare_stt"
    assert _BASH is not None
    result = subprocess.run(
        [_BASH, str(root / "prepare.sh")],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 7 and "diagnostics:" in result.stderr
    records = [
        json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()
    ]
    assert "tools.prepare_stt" in records[-1]["args"]
    assert not any("tools.prepare_tts" in row["args"] for row in records)


@pytest.mark.parametrize("flag,status", [("--help", 0), ("--unknown", 2)])
def test_bootstrap_argument_handling(tmp_path: Path, flag: str, status: int) -> None:
    """Handle help and invalid arguments before installing or preparing anything.

    Args:
        tmp_path:
            Isolated script and executable fixtures.

        flag:
            Help or invalid command-line option.

        status:
            Expected process exit status.

    """
    root, environment = bootstrap_fixture(tmp_path, False)
    assert _BASH is not None
    result = subprocess.run(
        [_BASH, str(root / "prepare.sh"), flag],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == status
    assert not (tmp_path / "trace.jsonl").exists()


def test_chinese_publication_is_verified_and_fail_closed(tmp_path: Path) -> None:
    """Reuse verified exports and remove readiness if publication is interrupted.

    Args:
        tmp_path:
            Minimal synthetic staged artifacts.

    """
    staging, output = tmp_path / "staging", tmp_path / "output"
    (staging / "voices").mkdir(parents=True)
    for name in (
        "model.xml",
        "model.bin",
        "config.json",
        "voices/zf_001.npy",
        "validation.wav",
    ):
        (staging / name).write_bytes(b"fixture")
    manifest = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "voice": "zf_001",
        "checkpoint_sha256": WEIGHT_SHA256,
        "voice_index": "phoneme_count_minus_one",
        "g2p_version": "1.1",
    }
    publish(staging, output, manifest)
    assert cached(output, "zf_001")
    (output / "model.bin").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        cached(output, "zf_001")
    for name in (
        "model.xml",
        "model.bin",
        "config.json",
        "voices/zf_001.npy",
        "validation.wav",
    ):
        (staging / name).write_bytes(b"replacement")
    with (
        patch.object(Path, "replace", side_effect=OSError("publication interrupted")),
        pytest.raises(OSError, match="publication interrupted"),
    ):
        publish(staging, output, manifest)
    assert not (output / "manifest.json").exists()


def test_guard_interrupt_stops_owned_scope(tmp_path: Path) -> None:
    """An interrupted supervisor must not leave its detached model process running.

    Args:
        tmp_path:
            Isolated durable child log path.

    """
    with (
        patch(
            "tools.guarded_run.psutil.virtual_memory",
            return_value=SimpleNamespace(available=16 * 1024**3),
        ),
        patch("tools.guarded_run.os.sched_getaffinity", return_value={0, 1}),
        patch("tools.guarded_run.os.sched_setaffinity"),
        patch("tools.guarded_run.subprocess.Popen") as launch,
        patch("tools.guarded_run._stop_scope") as stop,
    ):
        launch.return_value.poll.side_effect = KeyboardInterrupt
        with pytest.raises(KeyboardInterrupt):
            run_guarded(["fixture"], 2, 4, 3, 60, tmp_path / "guard.log")
        stop.assert_called_once()
        launch.return_value.wait.assert_called_once()


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_guard_rejects_nonfinite_timeout(tmp_path: Path, value: float) -> None:
    """Reject timeout values that would otherwise disable the watchdog comparison.

    Args:
        tmp_path:
            Unused log destination; validation occurs before creating it.

        value:
            Invalid nonfinite timeout.

    """
    with pytest.raises(ValueError, match="resource budget"):
        run_guarded(["fixture"], 2, 4, 3, value, tmp_path / "guard.log")
