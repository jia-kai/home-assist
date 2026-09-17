"""Offline bilingual routing, model-contract and worker lifecycle regressions."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hoast.chinese import contains_han, phoneme_batches
from hoast.chinese_g2p import ChineseG2P
from hoast.tts import TTS, TTSConfig

_WORKER = """
import json, sys
print(json.dumps({'ready': True, 'protocol': 1}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    if request['text'] == 'exit':
        sys.exit(2)
    identifier = request['id'] + (request['text'] == 'bad-id')
    print(json.dumps({'id': identifier, 'phonemes': request['text']}), flush=True)
"""


def test_han_and_lossless_bounded_syllables() -> None:
    """Recognize extended Han and retain tone/word boundaries across long batches."""
    for text in ("你好", "Hello. 你好。", "〇", "\U00020000", "\uf900"):
        assert contains_han(text)
    for text in ("Hello!", "123", "ㄋㄧ3", "かな", "한글", ""):
        assert not contains_han(text)
    for text in ("ㄋㄧ3/ㄏㄠ3 " * 100, "ㄋㄧ3" * 200):
        batches = phoneme_batches(text)
        assert all(0 < len(batch) <= 120 for batch in batches)
        assert "".join(batches).replace(" ", "") == text.replace(" ", "")
        assert all(batch[-1] in "12345/" for batch in batches)
    with pytest.raises(ValueError, match="syllable boundary"):
        phoneme_batches("a" * 451)


def test_lazy_routing_cache_voice_index_and_close(tmp_path: Path) -> None:
    """Load Chinese only on demand, preserve mixed input and index style by N-1.

    Args:
        tmp_path:
            Minimal synthetic local model/voice artifacts.

    """
    for name in ("model.xml", "model.bin", "english.onnx", "english.bin"):
        (tmp_path / name).touch()
    (tmp_path / "voices").mkdir()
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "repository": "hexgrad/Kokoro-82M-v1.1-zh",
                "voice_index": "phoneme_count_minus_one",
            }
        )
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "vocab": {
                    character: index for index, character in enumerate("ㄋㄧ3/ ", 1)
                },
            }
        )
    )
    voice = np.broadcast_to(
        np.arange(450, dtype=np.float32)[:, None, None], (450, 1, 256)
    )
    np.save(tmp_path / "voices" / "zf_001.npy", voice)
    english = MagicMock()
    english.get_voices.return_value = ["af_heart"]
    english.tokenizer.phonemize.return_value = "hello"
    english.create.return_value = (np.zeros(24, dtype=np.float32), 24000)
    with (
        patch("hoast.tts._OpenVINOSession") as session,
        patch("hoast.tts.SharedGPU") as gpu,
        patch("hoast.tts.Kokoro.from_session", return_value=english),
        patch("hoast.tts.ChineseG2P") as worker,
    ):
        session.return_value.run.return_value = [np.ones(24, dtype=np.float32)]
        worker.return_value.phonemize.return_value = "ㄋㄧ3/ㄋㄧ3"
        engine = TTS(
            TTSConfig(
                model_path=tmp_path / "english.onnx",
                voices_path=tmp_path / "english.bin",
                chinese_model_dir=tmp_path,
            )
        )
        engine.synthesize("Hello.")
        worker.assert_not_called()
        assert session.call_count == 1
        mixed = "Hello world. 你好，欢迎回家。"
        samples, rate = engine.synthesize(mixed)
        worker.return_value.phonemize.assert_called_once_with(mixed)
        assert samples.shape == (24,) and rate == 24000
        feed = session.return_value.run.call_args.args[1]
        np.testing.assert_array_equal(
            feed["style"], np.full((1, 256), 6, dtype=np.float32)
        )
        assert feed["tokens"].shape == (1, 9)
        engine.synthesize("你好")
        assert session.call_count == 2 and worker.call_count == 1
        assert gpu.call_count == 1
        assert all(
            call.kwargs["gpu"] is gpu.return_value for call in session.call_args_list
        )
        worker.return_value.phonemize.return_value = "ㄋㄧ3/" * 100
        samples, _ = engine.synthesize("你好" * 100)
        assert samples.size > 24
        worker.return_value.phonemize.return_value = "❓"
        with pytest.raises(ValueError, match="unsupported phonemes"):
            engine.synthesize("你好")
        engine.close()
        engine.close()
        worker.return_value.close.assert_called_once()
        assert engine._chinese is None


@pytest.mark.parametrize("file_logging", [False, True])
def test_worker_large_requests_failure_recovery_and_cleanup(
    tmp_path: Path, file_logging: bool
) -> None:
    """Exercise real pipes, recovery, and cleanup with console or opt-in file logging.

    Args:
        tmp_path:
            Isolated durable worker log location.

        file_logging:
            Whether to explicitly enable persistent worker diagnostics.

    """
    original = subprocess.Popen

    def launch(command: list[str], **kwargs: Any) -> subprocess.Popen[bytes]:
        """Substitute a tiny protocol worker without optional language dependencies.

        Args:
            command:
                Production worker command, replaced for this offline fixture.

            kwargs:
                Original binary pipe and environment settings.

        """
        assert ("--log-file" in command) is file_logging
        assert (kwargs["stderr"] is not None) is file_logging
        return original(
            [sys.executable, "-u", "-c", _WORKER],
            stdin=kwargs["stdin"],
            stdout=kwargs["stdout"],
            stderr=kwargs["stderr"],
            bufsize=kwargs["bufsize"],
            env=kwargs["env"],
        )

    worker = (
        ChineseG2P(Path(sys.executable), log_file=tmp_path / "worker.log")
        if file_logging
        else ChineseG2P(Path(sys.executable))
    )
    with patch("hoast.chinese_g2p.subprocess.Popen", side_effect=launch):
        try:
            assert worker._process is None
            text = "你好" * 20000
            assert worker.phonemize(text) == text
            process = worker._process
            assert worker.phonemize("hello") == "hello"
            assert worker._process is process
            for failure in ("bad-id", "exit"):
                with pytest.raises(RuntimeError):
                    worker.phonemize(failure)
                assert worker._process is None
                assert worker.phonemize("recovered") == "recovered"
            process = worker._process
        finally:
            worker.close()
        assert process is not None and process.poll() is not None
    assert (tmp_path / "worker.log").exists() is file_logging


def test_worker_limits_deadlines_and_bad_handshake(tmp_path: Path) -> None:
    """Reject oversized input before startup and handle stalled pipes and handshakes.

    Args:
        tmp_path:
            Isolated log directory.

    """
    worker = ChineseG2P(Path(sys.executable), log_file=tmp_path / "worker.log")
    with patch("hoast.chinese_g2p.subprocess.Popen") as launch:
        with pytest.raises(ValueError, match="size limit"):
            worker.phonemize("你" * 400000)
        launch.assert_not_called()
    with (
        patch("hoast.chinese_g2p.subprocess.Popen") as launch,
        patch("hoast.chinese_g2p.os.set_blocking"),
        patch.object(ChineseG2P, "_read", return_value={"ready": False}),
    ):
        with pytest.raises(RuntimeError, match="protocol mismatch"):
            worker.phonemize("你好")
        launch.return_value.wait.assert_called_once()
        assert worker._process is None
    with (
        patch("hoast.chinese_g2p.subprocess.Popen") as launch,
        patch("hoast.chinese_g2p.os.set_blocking"),
        patch.object(ChineseG2P, "_read", side_effect=KeyboardInterrupt),
    ):
        with pytest.raises(KeyboardInterrupt):
            worker.phonemize("你好")
        launch.return_value.wait.assert_called_once()
        assert worker._process is None
    worker._process = MagicMock()
    with patch("hoast.chinese_g2p.select.select", return_value=([], [], [])):
        with pytest.raises(TimeoutError):
            worker._write(b"request\n")
        with pytest.raises(TimeoutError):
            worker._read()
    worker.close()
    for timeout in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="timeout"):
            ChineseG2P(timeout=timeout)
