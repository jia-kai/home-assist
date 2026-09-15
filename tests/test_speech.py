"""Offline speech boundary tests using generated samples and mocked inference."""

import hashlib
import io
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import openvino as ov
import pytest
import soundfile as sf

from hoast.stt import STT, STTConfig, _WindowedWhisper
from hoast.tts import TTS, TTSConfig, phoneme_batches
from tools.prepare_stt import REVISION
from tools.prepare_stt import prepare as prepare_stt
from tools.prepare_tts import download


def test_missing_artifacts_fail_before_inference(tmp_path: Path) -> None:
    """Missing local files must not trigger backend downloads.

    Args:
        tmp_path:
            Isolated empty model directory.

    """
    with patch("hoast.stt._WindowedWhisper") as whisper:
        with pytest.raises(FileNotFoundError):
            STT(STTConfig(model_path=tmp_path))
        whisper.assert_not_called()
    with patch("hoast.tts.SharedGPU") as session:
        with pytest.raises(FileNotFoundError):
            TTS(TTSConfig(model_path=tmp_path / "missing.onnx"))
        session.assert_not_called()


def test_configuration_validation() -> None:
    """Reject zero thread budgets and invalid speed/language parameters."""
    with pytest.raises(ValueError):
        STTConfig(threads=0)
    with pytest.raises(ValueError):
        STTConfig(beam_size=0)
    with pytest.raises(ValueError):
        STTConfig(language=" ")
    with pytest.raises(ValueError):
        TTSConfig(threads=0)
    for speed in (float("nan"), float("inf"), 0.0, 2.1):
        with pytest.raises(ValueError):
            TTSConfig(speed=speed)
    for context in (float("nan"), 3.0, 31.0):
        with pytest.raises(ValueError):
            STTConfig(encoder_min_seconds=context)


def test_windowed_encoder_only_passes_bounded_prefix() -> None:
    """Pass the requested feature prefix without modifying valid input features."""
    model = _WindowedWhisper.__new__(_WindowedWhisper)
    model.encoder_frame_limit = 800
    features = np.arange(80 * 3000, dtype=np.float32).reshape(80, 3000)
    with patch("hoast.stt.WhisperModel.encode") as encode:
        model.encode(features)
        np.testing.assert_array_equal(encode.call_args.args[0], features[:, :800])
        model.encoder_frame_limit = 3000
        model.encode(features)
        assert encode.call_args.args[0].shape == features.shape


def test_uncertain_short_context_retries_full_encoder(tmp_path: Path) -> None:
    """Retry a low-confidence short-context result before returning text.

    Args:
        tmp_path:
            Isolated dummy checkpoint directory.

    """
    for name in ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt"):
        (tmp_path / name).touch()
    with patch("hoast.stt._WindowedWhisper") as backend:
        recognizer = STT(STTConfig(model_path=tmp_path, encoder_min_seconds=8))
        frames: list[int] = []

        def decode(
            audio: np.ndarray, **options: Any
        ) -> tuple[list[SimpleNamespace], SimpleNamespace]:
            """Return uncertain then confident speech and record encoder limits.

            Args:
                audio:
                    Passed-through test waveform.

                options:
                    Decoder options forwarded by the wrapper.

            """
            assert audio.size == 16000
            assert options["beam_size"] == 1
            frames.append(backend.return_value.encoder_frame_limit)
            first = len(frames) == 1
            return (
                [
                    SimpleNamespace(
                        text="uncertain" if first else "confirmed",
                        avg_logprob=-1.2 if first else -0.1,
                        compression_ratio=1.0,
                    )
                ],
                SimpleNamespace(duration=1.0, duration_after_vad=1.0, language="en"),
            )

        backend.return_value.transcribe.side_effect = decode
        assert (
            recognizer.transcribe_samples(np.zeros(16000, dtype=np.float32), 16000)
            == "confirmed"
        )
        assert frames == [800, 3000]


def test_corrupt_prepared_tts_artifact_is_rejected(tmp_path: Path) -> None:
    """Reject a corrupted cached model rather than loading or overwriting it.

    Args:
        tmp_path:
            Isolated download destination.

    """
    output = tmp_path / "kokoro-v1.0.onnx"
    output.write_bytes(b"incomplete model")
    with patch("tools.prepare_tts.urllib.request.urlopen") as request:
        with pytest.raises(ValueError, match="SHA256 mismatch"):
            download(output.name, output)
        request.assert_not_called()
    assert output.read_bytes() == b"incomplete model"


def test_download_publishes_only_verified_artifact(tmp_path: Path) -> None:
    """Publish a complete valid download and keep a corrupt download unpublished.

    Args:
        tmp_path:
            Isolated destination for a minimal binary fixture.

    """
    data = b"minimal model fixture"
    output = tmp_path / "fixture.bin"
    with patch.dict(
        "tools.prepare_tts.SHA256", {output.name: hashlib.sha256(data).hexdigest()}
    ):
        with (
            patch(
                "tools.prepare_tts.urllib.request.urlopen",
                return_value=io.BytesIO(b"bad"),
            ),
            pytest.raises(ValueError, match="SHA256 mismatch"),
        ):
            download(output.name, output)
        assert not output.exists()
        with patch(
            "tools.prepare_tts.urllib.request.urlopen", return_value=io.BytesIO(data)
        ):
            download(output.name, output)
        assert output.read_bytes() == data
        assert not output.with_suffix(".bin.part").exists()


def test_stt_preparation_pins_complete_offline_artifacts(tmp_path: Path) -> None:
    """Download the pinned small revision with its tokenizer before initialization.

    Args:
        tmp_path:
            Isolated checkpoint destination.

    """
    with (
        patch("tools.prepare_stt.snapshot_download") as snapshot,
        patch("tools.prepare_stt.STT") as recognizer,
    ):
        prepare_stt(tmp_path, 1)
        assert snapshot.call_args.args == ("Systran/faster-whisper-small",)
        assert snapshot.call_args.kwargs["revision"] == REVISION
        assert set(snapshot.call_args.kwargs["allow_patterns"]) == {
            "model.bin",
            "config.json",
            "tokenizer.json",
            "vocabulary.txt",
        }
        assert recognizer.call_args.args[0] == STTConfig(model_path=tmp_path, threads=1)


def test_transcription_consumes_lazy_segments(tmp_path: Path) -> None:
    """Complete lazy inference, preserve empty speech and propagate late errors.

    Args:
        tmp_path:
            Isolated artifact and input-audio directory.

    """
    for name in ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt"):
        (tmp_path / name).touch()
    audio = tmp_path / "silence.wav"
    sf.write(audio, np.zeros(1600, dtype=np.float32), 16000)
    info = SimpleNamespace(duration=0.1, duration_after_vad=0.0, language="en")

    def segments() -> Iterator[SimpleNamespace]:
        """Yield both delayed transcription segments."""
        yield SimpleNamespace(text=" Hello ", avg_logprob=0.0, compression_ratio=1.0)
        yield SimpleNamespace(text=" world. ", avg_logprob=0.0, compression_ratio=1.0)

    def failing_segments() -> Iterator[SimpleNamespace]:
        """Raise after a partial transcript to emulate a deferred backend error."""
        yield SimpleNamespace(text="partial", avg_logprob=0.0, compression_ratio=1.0)
        raise RuntimeError("decoder failed")

    with patch("hoast.stt._WindowedWhisper") as backend:
        recognizer = STT(STTConfig(model_path=tmp_path, threads=1))
        assert backend.call_args.kwargs["compute_type"] == "int8"
        assert backend.call_args.kwargs["device"] == "cpu"
        assert backend.call_args.kwargs["cpu_threads"] == 1
        assert backend.call_args.kwargs["num_workers"] == 1
        assert backend.call_args.kwargs["local_files_only"] is True
        backend.return_value.transcribe.return_value = (segments(), info)
        assert recognizer.transcribe(audio) == "Hello world."
        assert (
            backend.return_value.transcribe.call_args.kwargs["without_timestamps"]
            is True
        )
        backend.return_value.transcribe.return_value = (iter(()), info)
        assert recognizer.transcribe(audio) == ""
        backend.return_value.transcribe.return_value = (failing_segments(), info)
        with pytest.raises(RuntimeError, match="decoder failed"):
            recognizer.transcribe(audio)


def test_phoneme_batches_preserve_long_unpunctuated_text() -> None:
    """Retain every phoneme even when one word exceeds the model context."""
    for text in ("word " * 600, "a" * 1200, "abc. def? " * 100):
        batches = phoneme_batches(text)
        assert all(0 < len(batch) <= 450 for batch in batches)
        assert "".join(batches).replace(" ", "") == text.replace(" ", "")
    assert phoneme_batches(" ") == []


def test_in_memory_audio_resampling_and_validation(tmp_path: Path) -> None:
    """Preserve pitch/duration in 24-to-16 kHz handoff and reject invalid samples.

    Args:
        tmp_path:
            Isolated dummy checkpoint directory.

    """
    for name in ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt"):
        (tmp_path / name).touch()
    with patch("hoast.stt._WindowedWhisper") as backend:
        recognizer = STT(STTConfig(model_path=tmp_path))
        backend.return_value.transcribe.return_value = (
            (),
            SimpleNamespace(duration=0.1, duration_after_vad=0.0, language="en"),
        )
        waveform = np.sin(2 * np.pi * 1000 * np.arange(2400) / 24000).astype(np.float32)
        assert recognizer.transcribe_samples(waveform, 24000) == ""
        resampled = backend.return_value.transcribe.call_args.args[0]
        assert resampled.shape == (1600,)
        assert resampled.dtype == np.float32
        spectrum = np.abs(np.fft.rfft(resampled))
        assert np.argmax(spectrum) * 16000 / resampled.size == 1000
        recognizer.transcribe_samples(resampled, 16000)
        assert backend.return_value.transcribe.call_args.args[0] is resampled
        backend.return_value.transcribe.reset_mock()
        invalid_inputs: tuple[Any, ...] = (
            np.zeros(0, dtype=np.float32),
            np.zeros((2, 10), dtype=np.float32),
            np.array([np.nan], dtype=np.float32),
            np.array([np.inf], dtype=np.float32),
            np.zeros(10, dtype=np.float64),
        )
        for invalid in invalid_inputs:
            with pytest.raises(ValueError):
                recognizer.transcribe_samples(invalid, 24000)
        with pytest.raises(ValueError, match="sample_rate"):
            recognizer.transcribe_samples(waveform, 0)
        backend.return_value.transcribe.assert_not_called()


def test_synthesis_cpu_budget_and_wav_contract(tmp_path: Path) -> None:
    """Use the shared hybrid session and save all long-text samples as PCM16 mono WAV.

    Args:
        tmp_path:
            Isolated dummy artifacts and WAV destination.

    """
    model_path = tmp_path / "model.onnx"
    voices_path = tmp_path / "voices.bin"
    model_path.touch()
    voices_path.touch()
    backend = MagicMock()
    backend.get_voices.return_value = ["af_heart"]
    backend.tokenizer.phonemize.return_value = "a" * 1000
    backend.create.return_value = (np.zeros(2400, dtype=np.float32), 24000)
    with (
        patch("hoast.tts._OpenVINOSession") as session,
        patch("hoast.tts.SharedGPU") as gpu,
        patch("hoast.tts.Kokoro.from_session", return_value=backend),
    ):
        engine = TTS(TTSConfig(model_path, voices_path, threads=2))
        assert session.call_args.args[1] == 2
        assert session.call_args.kwargs["gpu"] is gpu.return_value
        output = tmp_path / "speech.wav"
        engine.write_wav("long utterance", output)
        info = sf.info(output)
        assert (info.samplerate, info.channels, info.frames, info.subtype) == (
            24000,
            1,
            7200,
            "PCM_16",
        )
        assert backend.create.call_count == 3
        with pytest.raises(ValueError, match="nonempty"):
            engine.synthesize(" ")
        backend.tokenizer.phonemize.return_value = ""
        with pytest.raises(ValueError, match="no speakable"):
            engine.synthesize("!!!")
        backend.tokenizer.phonemize.return_value = "test"
        backend.create.return_value = (np.zeros((2, 5), dtype=np.float32), 24000)
        with pytest.raises(RuntimeError, match="invalid mono"):
            engine.synthesize("test")
        engine.close()
        engine.close()
        session.return_value.close.assert_called_once()
        gpu.return_value.close.assert_called_once()
        with pytest.raises(RuntimeError, match="closed"):
            engine.synthesize("test")
        backend.get_voices.return_value = []
        with pytest.raises(ValueError, match="Unknown Kokoro voice"):
            TTS(TTSConfig(model_path, voices_path))


def test_hybrid_selection_and_output_ownership(tmp_path: Path) -> None:
    """Offload the decoder, retain bounded CPU execution and own copied audio outputs.

    Args:
        tmp_path:
            Isolated dummy model and voice files.

    """
    model_path = tmp_path / "model.onnx"
    voices_path = tmp_path / "voices.bin"
    model_path.touch()
    voices_path.touch()
    port = MagicMock()
    port.get_any_name.return_value = "tokens"
    port.get_element_type.return_value = ov.Type.i64
    backend = MagicMock()
    backend.get_voices.return_value = ["af_heart"]
    with (
        patch("hoast.tts.ov.Core") as core,
        patch("hoast.tts.fuse_cpu_activations"),
        patch("hoast.tts.SharedGPU") as gpu,
        patch("hoast.tts.replace_convolutions") as offload,
        patch("hoast.tts.Kokoro.from_session", return_value=backend) as frontend,
    ):
        compiled = core.return_value.compile_model.return_value
        compiled.inputs = [port]
        compiled.outputs = [port]
        buffer = np.ones(24, dtype=np.float32)
        compiled.create_infer_request.return_value.infer.return_value = {port: buffer}
        config = TTSConfig(model_path, voices_path, threads=2)
        TTS(config)
        offload.assert_called_once_with(
            core.return_value.read_model.return_value, gpu.return_value
        )
        assert core.return_value.compile_model.call_args.args[1] == "CPU"
        settings = core.return_value.compile_model.call_args.args[2]
        assert settings["INFERENCE_NUM_THREADS"] == 2
        assert settings["NUM_STREAMS"] == "1"
        assert settings["INFERENCE_PRECISION_HINT"] == "f32"
        adapter = frontend.call_args.args[0]
        assert adapter.get_inputs()[0].name == "tokens"
        samples = adapter.run(None, {})[0]
        buffer[:] = 0
        np.testing.assert_array_equal(samples, np.ones(24, dtype=np.float32))
        with pytest.raises(ValueError, match="all outputs"):
            adapter.run(["output"], {})
        core.return_value.compile_model.side_effect = RuntimeError("compile failed")
        with pytest.raises(RuntimeError, match="compile failed"):
            TTS(config)


def test_native_openvino_inputs_preserve_fractional_speed(tmp_path: Path) -> None:
    """Map native IR input names while preserving float speed and voice tensors.

    Args:
        tmp_path:
            Isolated dummy model and voices directory.

    """
    model_path = tmp_path / "model.xml"
    voices_path = tmp_path / "voices.bin"
    model_path.touch()
    voices_path.touch()
    ports = [MagicMock() for _ in range(3)]
    for port, name in zip(ports, ("input_ids", "ref_s", "speed"), strict=True):
        port.get_any_name.return_value = name
        port.get_element_type.return_value = (
            ov.Type.i64 if name == "input_ids" else ov.Type.f32
        )
    backend = MagicMock()
    backend.get_voices.return_value = ["af_heart"]
    with (
        patch("hoast.tts.ov.Core") as core,
        patch("hoast.tts.fuse_cpu_activations"),
        patch("hoast.tts.SharedGPU"),
        patch("hoast.tts.replace_convolutions"),
        patch("hoast.tts.Kokoro.from_session", return_value=backend) as frontend,
    ):
        compiled = core.return_value.compile_model.return_value
        compiled.inputs = ports
        compiled.outputs = [ports[0]]
        request = compiled.create_infer_request.return_value
        request.infer.return_value = {ports[0]: np.zeros(24, dtype=np.float32)}
        TTS(TTSConfig(model_path, voices_path, speed=1.25))
        adapter = frontend.call_args.args[0]
        assert [port.name for port in adapter.get_inputs()] == [
            "tokens",
            "style",
            "speed",
        ]
        style = np.zeros((1, 256), dtype=np.float32)
        adapter.run(
            None,
            {
                "tokens": [[0, 1, 0]],
                "style": style,
                "speed": np.array([1.25], dtype=np.float32),
            },
        )
        inputs = request.infer.call_args.args[0]
        assert set(inputs) == {"input_ids", "ref_s", "speed"}
        assert inputs["ref_s"] is style
        assert inputs["speed"].dtype == np.float32
        assert inputs["speed"][0] == 1.25
