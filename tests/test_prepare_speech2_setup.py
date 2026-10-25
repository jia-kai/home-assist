"""Bootstrap CLI/routing fixtures: no downloads, installs, exports or inference."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "prepare.sh"
FAKE_PYTHON = r"""
import json
import os
import sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ["COMMAND_LOG"], "a", encoding="utf-8") as stream:
    stream.write(json.dumps(args) + "\n")
if args[:2] == ["-m", "tools.guarded_run"]:
    payload = args[args.index("--") + 1:]
    if "tools.fetch_speech2" in payload:
        output = Path(payload[payload.index("--output") + 1])
        output.write_text(json.dumps({
            "stt_source": "/fixture/stt", "melo_source": "/fixture/melo",
            "assets_manifest": "/fixture/assets.json",
            "g2p_resource_dir": "/fixture/g2p",
            "nltk_resource_dir": "/fixture/nltk",
            "community_source": "/fixture/community" if "--with-community" in payload else None,
            "license_dir": "/fixture/licenses",
        }))
elif args[0] == "-c" and len(args) > 2:
    inputs = json.loads(Path(args[2]).read_text())
    for key in ("stt_source", "melo_source", "assets_manifest", "g2p_resource_dir", "nltk_resource_dir", "community_source", "license_dir"):
        sys.stdout.buffer.write((inputs[key] or "").encode() + b"\0")
"""


def fixture_checkout(tmp_path: Path) -> tuple[Path, dict[str, str], Path]:
    """Build command stubs that record routing without performing preparation.

    Args:
        tmp_path:
            Isolated checkout fixture with no real models or environments.

    """
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    shutil.copyfile(SCRIPT, checkout / "prepare.sh")
    (checkout / ".python-version").write_text("3.14\n")
    selection = checkout / "models/lfm/model.json"
    selection.parent.mkdir(parents=True)
    selection.write_text("{}\n")
    commands = tmp_path / "commands.jsonl"
    binary = tmp_path / "bin"
    binary.mkdir()
    for name in ("uv", "c++", "systemd-run", "systemctl", "taskset", "git"):
        stub = binary / name
        stub.write_text("#!/bin/sh\nexit 0\n")
        stub.chmod(0o755)
    python = checkout / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text(f"#!{sys.executable}\n" + FAKE_PYTHON)
    python.chmod(0o755)
    environment = dict(os.environ)
    environment["PATH"] = str(binary) + os.pathsep + os.defpath
    environment["COMMAND_LOG"] = str(commands)
    return checkout, environment, commands


def test_help_and_invalid_policy_do_not_provision(tmp_path: Path) -> None:
    """CLI validation exits before creating diagnostic/cache/environment files.

    Args:
        tmp_path:
            Bare script fixture without a valid Python environment.

    """
    script = tmp_path / "prepare.sh"
    shutil.copyfile(SCRIPT, script)
    help_result = subprocess.run(
        ["bash", str(script), "--help"], capture_output=True, text=True, check=True
    )
    assert "Python 3.14" in help_result.stdout
    assert "--offline" in help_result.stdout
    for arguments in (
        ["--threads", "3"],
        ["--stt-backend", "ort"],
        ["--stt-source"],
        ["--community-source", "x"],
    ):
        result = subprocess.run(
            ["bash", str(script), *arguments],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 2
    assert not (tmp_path / ".cache").exists()


@pytest.mark.parametrize("precision,backend", [("FP32", "cpu"), ("UINT8", "ort")])
def test_native_speech_command_routing(
    tmp_path: Path, precision: str, backend: str
) -> None:
    """Both artifact formats route native commands and preserve unrelated LFM steps.

    Args:
        tmp_path:
            Isolated checkout and recording executables.

        precision:
            Selected official FP32 or community UINT8 artifact storage.

        backend:
            Selected OpenVINO CPU or ONNX Runtime backend.

    """
    checkout, environment, commands = fixture_checkout(tmp_path)
    result = subprocess.run(
        [
            "bash",
            str(checkout / "prepare.sh"),
            "--offline",
            "--stt-precision",
            precision,
            "--stt-backend",
            backend,
            "--tts-precision",
            "FP16-decoder",
            "--threads",
            "1",
            "--stt-output",
            "new-stt",
            "--tts-output",
            "new-tts",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    recorded = [json.loads(line) for line in commands.read_text().splitlines()]
    payloads = [
        args[args.index("--") + 1 :]
        for args in recorded
        if args[:2] == ["-m", "tools.guarded_run"]
    ]
    modules = [args[args.index("-m") + 1] for args in payloads]
    assert modules.count("tools.prepare_tts2") == 2
    assert "tools.prepare_stt" not in modules
    assert "tools.prepare_tts" not in modules
    assert "tools.check_speech" not in modules
    assert modules.count("tools.prepare_llm") == 2
    assert all("download" not in args for args in payloads)
    inputs = next(args for args in payloads if "tools.fetch_speech2" in args)
    assert "--offline" in inputs
    assert inputs[inputs.index("--root") + 1] == "models/speech2-inputs"
    vad = next(args for args in payloads if "tools.prepare_vad" in args)
    assert "--offline" in vad
    assert inputs[inputs.index("--output") + 1] == "models/speech2-inputs/inputs.json"
    for args in payloads:
        if "tools.prepare_tts2" in args:
            assert args[args.index("--precision") + 1] == "FP16"
            assert args[args.index("--fp16-stages") + 1] == "decoder"
            assert "--verify-original" in args
            assert args[args.index("--g2p-resource-dir") + 1] == "/fixture/g2p"
    if precision == "UINT8":
        assert "--with-community" in inputs
        stt = next(args for args in payloads if "tools.prepare_stt2_community" in args)
        assert stt[stt.index("--backend") + 1] == "onnxruntime"
    else:
        assert "tools.prepare_stt2" in modules
        stt = next(args for args in payloads if "tools.prepare_stt2" in args)
    assert stt[stt.index("--license-dir") + 1] == "/fixture/licenses"
    assert "tools.check_speech2" in result.stdout
    assert "independent speech qualification" in result.stdout
    assert not (checkout / "config.toml").exists()


def test_fp32_melo_selects_separate_default_root(tmp_path: Path) -> None:
    """FP32 comparison artifacts cannot implicitly replace the mixed default root.

    Args:
        tmp_path:
            Isolated recording checkout and cached LFM selection.

    """
    checkout, environment, commands = fixture_checkout(tmp_path)
    result = subprocess.run(
        ["bash", str(checkout / "prepare.sh"), "--offline", "--tts-precision", "FP32"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    recorded = [json.loads(line) for line in commands.read_text().splitlines()]
    tts = [args for args in recorded if "tools.prepare_tts2" in args]
    assert len(tts) == 2
    for args in tts:
        assert args[args.index("--precision") + 1] == "FP32"
        assert args[args.index("--output") + 1] == "models/tts2-fp32"
        assert "--fp16-stages" not in args


def test_existing_output_fails_before_model_work(tmp_path: Path) -> None:
    """Existing artifacts survive and block bootstrap before any Python command.

    Args:
        tmp_path:
            Fixture checkout containing a prior STT artifact.

    """
    checkout, environment, commands = fixture_checkout(tmp_path)
    existing = checkout / "models/stt2"
    existing.mkdir(parents=True)
    prior = existing / "prior"
    prior.write_bytes(b"keep")
    result = subprocess.run(
        ["bash", str(checkout / "prepare.sh"), "--offline"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "Refusing to replace" in result.stderr
    assert prior.read_bytes() == b"keep"
    assert not commands.exists()
    assert list((checkout / ".cache/hoast/diagnostics/prepare").glob("*/bootstrap.log"))


def test_offline_requires_permanent_lfm_selection(tmp_path: Path) -> None:
    """Reject a cache-only selection before installing or preparing any model.

    Args:
        tmp_path:
            Fixture checkout with a disposable selection but no permanent manifest.

    """
    checkout, environment, commands = fixture_checkout(tmp_path)
    (checkout / "models/lfm/model.json").unlink()
    cached = checkout / ".cache/hoast/lfm/model.json"
    cached.parent.mkdir(parents=True)
    cached.write_text("{}")
    result = subprocess.run(
        ["bash", str(checkout / "prepare.sh"), "--offline"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "models/lfm/model.json" in result.stderr
    assert not commands.exists()
