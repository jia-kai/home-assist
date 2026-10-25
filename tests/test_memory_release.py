"""Offline ownership and failure cleanup checks for speech intermediate release."""

import threading
from unittest.mock import Mock

import numpy as np
import openvino as ov
import pytest
from numpy.typing import NDArray
from openvino import opset13 as ops

from hoast.stt2 import STT, STTConfig
from hoast.stt2.engine import OpenVINOEngine
from hoast.stt2.ort_engine import ORTEngine
from hoast.tts2 import TTS, TTSConfig
from hoast.tts2.engine import Resources, Stage
from hoast.tts2.frontend import TextInputs


def recognizer(release: bool) -> STT:
    """Create an unloaded fixture whose transcription helper can be replaced.

    Args:
        release:
            Whether the fixture must release intermediate memory on call exit.

    """
    result = object.__new__(STT)
    result.config = STTConfig(release_after_call=release)
    result._lock = threading.RLock()
    result._closed = False
    result.engine = Mock()
    result.last_timings = {}
    return result


@pytest.mark.parametrize("release", [False, True])
def test_stt_call_cleanup(monkeypatch: pytest.MonkeyPatch, release: bool) -> None:
    """Include optional cleanup in timing without altering the decoded response.

    Args:
        monkeypatch:
            Replaces model work with a fixture free of external model artifacts.

        release:
            Explicit opt-in or opt-out policy under test.

    """
    stt = recognizer(release)
    work = Mock(return_value="hello")
    monkeypatch.setattr(STT, "_transcribe_samples", work)
    assert stt.transcribe_samples(np.zeros(400, dtype=np.float32), 16000) == "hello"
    assert isinstance(stt.engine, Mock)
    assert stt.engine.release_memory.call_count == int(release)
    assert ("release_memory" in stt.last_timings) == release
    assert stt.last_timings["total"] >= stt.last_timings.get("release_memory", 0)


def test_stt_failure_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Release intermediates after a failed call and propagate the original failure.

    Args:
        monkeypatch:
            Makes the fixture model fail synchronously.

    """
    stt = recognizer(True)
    monkeypatch.setattr(STT, "_transcribe_samples", Mock(side_effect=ValueError("bad")))
    with pytest.raises(ValueError, match="bad"):
        stt.transcribe_samples(np.zeros(400, dtype=np.float32), 16000)
    assert isinstance(stt.engine, Mock)
    stt.engine.release_memory.assert_called_once()


def synthesizer(release: bool) -> TTS:
    """Create independent native-stage mocks for both loaded languages.

    Args:
        release:
            Whether synthesis automatically releases all loaded stages on exit.

    """
    tts = TTS(TTSConfig(release_after_call=release), playback_only=True)
    tts._playback_only = False
    tts._resources = {
        language: Resources(
            Mock(),
            {
                name: Mock(
                    spec=Stage, compiled=Mock(), device="CPU", precision_hint="f32"
                )
                for name in ("bert", "encoder", "flow", "decoder")
            },
            {"EN-Default": 0, "ZH": 1},
            44100,
            2,
            {"precision": "FP32"},
        )
        for language in ("EN", "ZH")
    }
    return tts


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("release_failures", [0, 1, 2])
def test_tts_cleanup_attempts_every_stage(
    monkeypatch: pytest.MonkeyPatch, failure: bool, release_failures: int
) -> None:
    """Attempt all stages and preserve cleanup identities and synthesis context.

    Args:
        monkeypatch:
            Replaces synthesis work while retaining the real cleanup wrapper.

        failure:
            Whether synthesis fails before cleanup begins.

        release_failures:
            Number of distinct native release failures across the two languages.

    """
    tts = synthesizer(True)
    synthesis_error = ValueError("synthesis failed")
    targets = [("EN", "bert"), ("ZH", "flow")][:release_failures]
    errors = [
        RuntimeError(f"native release failed {index}")
        for index in range(release_failures)
    ]
    for (language, name), error in zip(targets, errors, strict=True):
        compiled = tts._resources[language].stages[name].compiled
        assert isinstance(compiled, Mock)
        compiled.release_memory.side_effect = error
    work = Mock(
        side_effect=synthesis_error if failure else None,
        return_value=(np.arange(8, dtype=np.float32), 44100),
    )
    monkeypatch.setattr(TTS, "_synthesize", work)
    if release_failures:
        with pytest.raises(
            RuntimeError if release_failures == 1 else ExceptionGroup
        ) as caught:
            tts.synthesize("Hello")
        if release_failures == 1:
            assert caught.value is errors[0]
        else:
            assert isinstance(caught.value, ExceptionGroup)
            assert caught.value.exceptions == tuple(errors)
        assert caught.value.__context__ is (synthesis_error if failure else None)
        for (language, name), error in zip(targets, errors, strict=True):
            assert error.__notes__ == [
                f"TTS release_memory language={language} stage={name}"
            ]
    elif failure:
        with pytest.raises(ValueError) as caught_synthesis:
            tts.synthesize("Hello")
        assert caught_synthesis.value is synthesis_error
    else:
        audio, rate = tts.synthesize("Hello")
        np.testing.assert_array_equal(audio, np.arange(8, dtype=np.float32))
        assert rate == 44100
        assert tts.last_timings["total"] >= tts.last_timings["release_memory"]
    for resources in tts._resources.values():
        for stage in resources.stages.values():
            assert isinstance(stage.compiled, Mock)
            stage.compiled.release_memory.assert_called_once()


@pytest.mark.parametrize("release", [False, True])
def test_tts_multisegment_output_lifetime(release: bool) -> None:
    """Own each decoder segment before buffer reuse, subsequent calls and release.

    Args:
        release:
            Whether to release automatically as well as explicitly after calls.

    """
    tts = synthesizer(release)
    resources = tts._resources["EN"]
    assert isinstance(resources.frontend, Mock)
    resources.frontend.split.return_value = ["Hello.", "Again."]
    resources.frontend.prepare.return_value = TextInputs(
        phones=np.zeros((1, 2), dtype=np.int64),
        tones=np.zeros((1, 2), dtype=np.int64),
        languages=np.zeros((1, 2), dtype=np.int64),
        input_ids=np.zeros((1, 2), dtype=np.int64),
        word2ph=np.ones(2, dtype=np.int64),
        normalized_text="Hello.",
    )
    bert, encoder, flow, decoder = (
        resources.stages[name] for name in ("bert", "encoder", "flow", "decoder")
    )
    assert isinstance(bert, Mock) and isinstance(encoder, Mock)
    assert isinstance(flow, Mock) and isinstance(decoder, Mock)
    bert.inputs = {"input_ids"}
    bert.run.return_value = [np.zeros((1, 2, 768), dtype=np.float32)]
    encoder.run.return_value = [
        np.zeros((1, 192, 2), dtype=np.float32),
        np.zeros((1, 192, 2), dtype=np.float32),
        np.zeros((1, 1, 2), dtype=np.float32),
        np.zeros((1, 1, 2), dtype=np.float32),
        np.zeros((1, 1, 1), dtype=np.float32),
    ]
    flow.run.return_value = [np.zeros((1, 192, 2), dtype=np.float32)]
    borrowed = np.zeros((1, 1, 4), dtype=np.float32)

    def decode(values: dict[str, NDArray[np.float32]]) -> list[NDArray[np.float32]]:
        """Overwrite a single native-like waveform buffer on every segment.

        Args:
            values:
                Latent and speaker tensors passed by the actual synthesis pipeline.

        """
        assert values["latent"].shape == (1, 192, 2)
        borrowed.fill(decoder.run.call_count)
        return [borrowed]

    decoder.run.side_effect = decode
    assert isinstance(decoder.compiled, Mock)
    decoder.compiled.release_memory.side_effect = lambda: borrowed.fill(-1)
    gap = np.zeros(int(resources.rate * 0.05), dtype=np.float32)
    expected = np.concatenate(
        (np.ones(4, dtype=np.float32), gap, np.full(4, 2, dtype=np.float32), gap)
    )
    first, rate = tts.synthesize("Hello. Again.")
    assert rate == resources.rate and first.dtype == np.float32
    np.testing.assert_array_equal(first, expected)
    assert not np.shares_memory(first, borrowed)
    second, _ = tts.synthesize("Hello. Again.")
    np.testing.assert_array_equal(
        second,
        np.concatenate(
            (np.full(4, 3, dtype=np.float32), gap, np.full(4, 4, dtype=np.float32), gap)
        ),
    )
    np.testing.assert_array_equal(first, expected)
    tts.release_memory()
    np.testing.assert_array_equal(borrowed, -np.ones_like(borrowed))
    np.testing.assert_array_equal(first, expected)
    assert decoder.run.call_count == 4


def test_release_error_is_not_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    """Native cleanup failures propagate instead of reporting successful release.

    Args:
        monkeypatch:
            Replaces speech work with a successful decode.

    """
    stt = recognizer(True)
    assert isinstance(stt.engine, Mock)
    stt.engine.release_memory.side_effect = RuntimeError("native release failed")
    monkeypatch.setattr(STT, "_transcribe_samples", Mock(return_value="hello"))
    with pytest.raises(RuntimeError, match="native release failed"):
        stt.transcribe_samples(np.zeros(400, dtype=np.float32), 16000)


def test_release_defaults_require_explicit_opt_in() -> None:
    """Retain intermediates unless a caller explicitly chooses an experimental policy."""
    assert not STTConfig().release_after_call
    assert not TTSConfig().release_after_call


def test_unsupported_and_closed_release() -> None:
    """Reject unsupported ORT policies and never resurrect closed speech engines."""
    with pytest.raises(ValueError, match="OpenVINO"):
        STTConfig(backend="ort", release_after_call=True)
    with pytest.raises(TypeError, match="boolean"):
        STTConfig(release_after_call=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="boolean"):
        TTSConfig(release_after_call=1)  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError):
        object.__new__(ORTEngine).release_memory()
    stt = recognizer(True)
    stt._closed = True
    with pytest.raises(RuntimeError, match="closed"):
        stt.release_memory()
    engine = object.__new__(OpenVINOEngine)
    engine.request = None
    engine.compiled_model = None
    with pytest.raises(RuntimeError, match="closed"):
        engine.release_memory()


def test_real_request_reusable_after_release() -> None:
    """Real engine inference owns IDs across same/different shapes and releases."""
    speech = ops.parameter([1, -1, 560], ov.Type.f32, name="speech")
    language = ops.parameter([1], ov.Type.i64, name="language")
    textnorm = ops.parameter([1], ov.Type.i64, name="textnorm")
    ids = ops.convert(
        ops.reduce_sum(speech, np.array([2], dtype=np.int64), False), "i64"
    )
    controls = ops.unsqueeze(ops.add(language, textnorm), np.array([0], dtype=np.int64))
    output = ops.concat(
        [ops.constant(np.zeros((1, 4), dtype=np.int64)), ops.add(ids, controls)], 1
    )
    output.output(0).get_tensor().set_names({"frame_ids"})
    model = ov.Model([output], [speech, language, textnorm])  # pyright: ignore[reportGeneralTypeIssues]
    engine = object.__new__(OpenVINOEngine)
    engine.compiled_model = ov.Core().compile_model(
        model,
        "CPU",
        {"NUM_STREAMS": 1, "INFERENCE_NUM_THREADS": 1, "ENABLE_CPU_PINNING": False},
    )
    engine.request = engine.compiled_model.create_infer_request()
    saved = engine.infer(np.ones((2, 560), dtype=np.float32), 3, 14)
    expected = np.array([0, 0, 0, 0, 577, 577], dtype=np.int64)
    np.testing.assert_array_equal(saved, expected)
    second = engine.infer(np.zeros((2, 560), dtype=np.float32), 0, 15)
    np.testing.assert_array_equal(second, [0, 0, 0, 0, 15, 15])
    np.testing.assert_array_equal(saved, expected)
    assert not np.shares_memory(saved, engine.request.get_tensor("frame_ids").data)
    engine.release_memory()
    np.testing.assert_array_equal(saved, expected)
    third = engine.infer(np.zeros((3, 560), dtype=np.float32), 4, 14)
    np.testing.assert_array_equal(third, [0, 0, 0, 0, 18, 18, 18])
    engine.release_memory()
    np.testing.assert_array_equal(saved, expected)
    assert engine.request.get_tensor("frame_ids").shape == ov.Shape([1, 7])
    engine.close()
