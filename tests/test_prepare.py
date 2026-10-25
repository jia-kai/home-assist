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
_PYTHON = r"""import json, os, pathlib, sys
args = sys.argv[1:]
with open(os.environ["TRACE"], "a") as log:
    log.write(json.dumps({"tool": "python", "args": args, "cwd": os.getcwd()}) + "\n")
if os.environ.get("FAIL_STAGE") in args:
    path = pathlib.Path(args[args.index("--log-file") + 1])
    path.write_text("simulated model failure\n")
    sys.exit(7)
if args[:2] == ["-m", "tools.guarded_run"] and "tools.fetch_speech2" in args:
    output = pathlib.Path(args[args.index("--output") + 1])
    output.write_text(json.dumps({
        "stt_source": "/fixture/stt", "melo_source": "/fixture/melo",
        "assets_manifest": "/fixture/assets.json", "g2p_resource_dir": "/fixture/g2p",
        "nltk_resource_dir": "/fixture/nltk", "community_source": None,
        "license_dir": "/fixture/licenses",
    }))
elif args[0] == "-c" and len(args) > 2:
    inputs = json.loads(pathlib.Path(args[2]).read_text())
    for key in ("stt_source", "melo_source", "assets_manifest", "g2p_resource_dir", "nltk_resource_dir", "community_source", "license_dir"):
        sys.stdout.buffer.write((inputs[key] or "").encode() + b"\0")
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
    for name in ("dirname", "date", "mkdir", "sh", "cp", "chmod", "cat", "tee"):
        source = shutil.which(name)
        assert source is not None
        (bin_dir / name).symlink_to(source)
    for name in ("c++", "systemd-run", "systemctl", "taskset", "git"):
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
    stages = [row for row in records if row["args"][:2] == ["-m", "tools.guarded_run"]]
    names = [
        Path(row["args"][row["args"].index("--log-file") + 1]).stem for row in stages
    ]
    expected = [
        "vad",
        "speech-inputs",
        "lfm-download",
        "lfm-gpu",
        "lfm-cpu",
        "stt2",
        "tts2-EN",
        "tts2-ZH",
    ]
    if extra:
        expected += ["functiongemma-download", "functiongemma-export"]
    assert names == expected
    for row in stages:
        args = row["args"]
        assert args[:2] == ["-m", "tools.guarded_run"]
        assert args[args.index("--threads") + 1] in ("1", "2")
        assert args[args.index("--memory-gib") + 1] == "4"
    assert "tools.prepare_vad" in stages[0]["args"]
    assert "--offline" not in stages[0]["args"]
    assert "--download" in stages[1]["args"]
    assert "tools.prepare_stt2" in stages[5]["args"]
    assert "--license-dir" in stages[5]["args"]
    assert stages[5]["args"][stages[5]["args"].index("--output") + 1] == "models/stt2"
    for index, language in ((6, "EN"), (7, "ZH")):
        args = stages[index]["args"]
        assert "tools.prepare_tts2" in args
        assert args[args.index("--language") + 1] == language
        assert args[args.index("--precision") + 1] == "FP16"
        assert args[args.index("--fp16-stages") + 1] == "decoder"
        assert args[args.index("--output") + 1] == "models/tts2"
        assert "--verify-original" in args
    assert "tools.check_speech2" in result.stdout
    assert not any("tools.check_speech2" in row["args"] for row in stages)
    assert (root / "config.toml").read_text() == "keep this configuration\n"
    assert (root / ".env").read_text() == "FIXTURE_TOKEN=keep-this-value\n"


def test_bootstrap_offline_vad(tmp_path: Path) -> None:
    """Use the verified local VAD asset without invoking any network acquisition.

    Args:
        tmp_path:
            Isolated script and executable fixtures.

    """
    root, environment = bootstrap_fixture(tmp_path, True)
    selection = root / "models/lfm/model.json"
    selection.parent.mkdir(parents=True)
    selection.write_text("{}")
    assert _BASH is not None
    result = subprocess.run(
        [_BASH, str(root / "prepare.sh"), "--offline"],
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
    vad = next(row for row in records if "tools.prepare_vad" in row["args"])
    assert "--offline" in vad["args"]
    assert not any("download" in row["args"] for row in records)


def test_bootstrap_stops_after_failure(tmp_path: Path) -> None:
    """Propagate a model failure and leave later preparation stages unexecuted.

    Args:
        tmp_path:
            Isolated test directory.

    """
    root, environment = bootstrap_fixture(tmp_path, True)
    environment["FAIL_STAGE"] = "tools.prepare_stt2"
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
    assert "tools.prepare_stt2" in records[-1]["args"]
    assert not any("tools.prepare_tts2" in row["args"] for row in records)


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
