"""Small offline fixtures for inclusive speech input and generated-output budgets."""

import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
from numpy.typing import NDArray

from hoast import stt2
from hoast.stt2 import STT, STTConfig
from hoast.tts2 import TTS, TTSConfig
from hoast.tts2 import engine as tts_engine
from hoast.tts2.engine import Resources, Stage
from hoast.tts2.frontend import TextInputs


def _recognizer(
    config: STTConfig,
    raw: str = "ok",
    ids: list[int] | None = None,
) -> STT:
    """Build tiny mock stages without artifacts, CPU-policy changes or inference.

    Args:
        config:
            Validated request budgets and optional cleanup policy.

        raw:
            Complete string returned by the mock tokenizer.

        ids:
            Eight frame IDs; None emits three collapsed nonblank tokens.

    """
    result = object.__new__(STT)
    result.config = config
    result._lock = threading.RLock()
    result._closed = False
    result.frontend = Mock()
    result.frontend.extract.return_value = np.zeros((4, 560), dtype=np.float32)
    result.engine = Mock()
    result.engine.infer.return_value = np.array(
        [0, 1, 1, 0, 1, 2, 2, 0] if ids is None else ids, dtype=np.int64
    )
    result.tokenizer = Mock()
    result.tokenizer.decode.return_value = raw
    result.last_raw_text = "prior request"
    result.last_timings = {"total": 1.0}
    result.last_profile = []
    return result


def _synthesizer(config: TTSConfig, segments: int = 1) -> TTS:
    """Build a 100-Hz fixture generating two frames and one gap per segment.

    Args:
        config:
            Validated speech budgets; no real model is constructed.

        segments:
            Number of identical two-phone sentence chunks to synthesize.

    """
    result = TTS(config, playback_only=True)
    result._playback_only = False
    frontend = Mock()
    frontend.split.return_value = ["Hi."] * segments
    frontend.prepare.return_value = TextInputs(
        phones=np.zeros((1, 2), dtype=np.int64),
        tones=np.zeros((1, 2), dtype=np.int64),
        languages=np.zeros((1, 2), dtype=np.int64),
        input_ids=np.zeros((1, 2), dtype=np.int64),
        word2ph=np.ones(2, dtype=np.int64),
        normalized_text="Hi.",
    )
    stages = {
        name: Mock(spec=Stage, device="CPU", precision_hint="f32", compiled=Mock())
        for name in ("bert", "encoder", "flow", "decoder")
    }
    stages["bert"].inputs = {"input_ids"}
    stages["bert"].run.return_value = [np.zeros((1, 2, 768), dtype=np.float32)]
    stages["encoder"].run.return_value = [
        np.zeros((1, 192, 2), dtype=np.float32),
        np.zeros((1, 192, 2), dtype=np.float32),
        np.zeros((1, 1, 2), dtype=np.float32),
        np.zeros((1, 1, 2), dtype=np.float32),
        np.zeros((1, 1, 1), dtype=np.float32),
    ]
    stages["flow"].run.side_effect = lambda values: [values["latent"]]
    stages["decoder"].run.side_effect = lambda values: [
        np.zeros((1, 1, values["latent"].shape[-1] * 2), dtype=np.float32)
    ]
    result._resources["EN"] = Resources(
        frontend,
        {name: stage for name, stage in stages.items()},
        {"EN-Default": 0},
        100,
        2,
        {"precision": "FP32"},
    )
    return result


@pytest.mark.parametrize(
    "name", ["max_input_samples", "max_output_tokens", "max_output_chars"]
)
@pytest.mark.parametrize("value", [True, False, 0, -1, 1.0, "1", None])
def test_stt_exact_integer_limits(name: str, value: Any) -> None:
    """Reject invalid STT integer budgets without constructing model resources.

    Args:
        name:
            Configured integer budget under test.

        value:
            Invalid value, including bool and integral float lookalikes.

    """
    with pytest.raises(ValueError, match=f"{name} must be a positive integer"):
        STTConfig(**{name: value})


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.0, "1", None])
def test_tts_exact_integer_limit(value: Any) -> None:
    """Reject non-exact positive integers for the raw text budget.

    Args:
        value:
            Invalid input-character budget.

    """
    with pytest.raises(ValueError, match="max_input_chars must be a positive integer"):
        TTSConfig(max_input_chars=value)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), -float("inf")])
def test_tts_finite_audio_limit(value: float) -> None:
    """Reject nonpositive and nonfinite audio duration budgets.

    Args:
        value:
            Invalid duration in seconds.

    """
    with pytest.raises(ValueError, match="finite and positive"):
        TTSConfig(max_audio_seconds=value)


@pytest.mark.parametrize("value", [True, "30", None])
def test_tts_numeric_audio_limit(value: Any) -> None:
    """Reject bool/string/missing audio durations without coercion.

    Args:
        value:
            Invalid duration type.

    """
    with pytest.raises(TypeError, match="must be a number"):
        TTSConfig(max_audio_seconds=value)


def test_default_limits_and_release_policy() -> None:
    """Keep library durations, native speed and explicit release opt-in defaults."""
    stt, tts = STTConfig(), TTSConfig()
    assert (stt.max_input_samples, stt.max_output_tokens, stt.max_output_chars) == (
        2_880_000,
        1024,
        4096,
    )
    assert stt.max_audio_seconds == 60.0
    assert (tts.max_input_chars, tts.max_audio_seconds, tts.speed) == (4000, 30.0, 1.0)
    assert not stt.release_after_call and not tts.release_after_call


@pytest.mark.parametrize("sample_rate", [16000, 24000])
def test_stt_source_sample_boundary(
    monkeypatch: pytest.MonkeyPatch, sample_rate: int
) -> None:
    """Reject one extra source sample before finite scans/resampling/frontend.

    Args:
        monkeypatch:
            Replaces expensive work with mocks after the exact-boundary call.

        sample_rate:
            Source rate in Hz, native or requiring resampling.

    """
    stt = _recognizer(STTConfig(max_input_samples=600))
    assert stt.transcribe_samples(np.zeros(600, np.float32), sample_rate) == "ok"
    assert isinstance(stt.frontend, Mock) and isinstance(stt.engine, Mock)
    stt.frontend.extract.reset_mock()
    stt.engine.infer.reset_mock()
    scan, resample = Mock(), Mock()
    monkeypatch.setattr(stt2.np, "isfinite", scan)
    monkeypatch.setattr(stt2, "resample_poly", resample)
    with pytest.raises(ValueError, match="max_input_samples"):
        stt.transcribe_samples(np.zeros(601, np.float32), sample_rate)
    scan.assert_not_called()
    resample.assert_not_called()
    stt.frontend.extract.assert_not_called()
    stt.engine.infer.assert_not_called()
    assert (
        stt.last_raw_text is None and stt.last_timings == {} and stt.last_profile == []
    )
    stt.engine.release_memory.assert_not_called()


@pytest.mark.parametrize("sample_rate", [8000, 16000, 44100, 48000, 192000, 15996])
def test_stt_supported_resampling_boundaries(
    monkeypatch: pytest.MonkeyPatch, sample_rate: int
) -> None:
    """Accept rate endpoints, common rates and an exact reduced factor of 4000.

    Args:
        monkeypatch:
            Records real bounded resampling without loading model artifacts.

        sample_rate:
            Supported source rate in Hz; 15996 reduces to factors 4000/3999.

    """
    stt = _recognizer(STTConfig())
    resample = Mock(wraps=stt2.resample_poly)
    monkeypatch.setattr(stt2, "resample_poly", resample)
    samples = np.zeros(sample_rate // 40, np.float32)
    assert stt.transcribe_samples(samples, sample_rate) == "ok"
    if sample_rate == 16000:
        resample.assert_not_called()
    else:
        resample.assert_called_once()
        assert max(resample.call_args.args[1:]) <= 4000
        if sample_rate == 15996:
            assert resample.call_args.args[1:] == (4000, 3999)


@pytest.mark.parametrize("sample_rate", [1, 7999, 192001, 1_000_000_007, 16004, 16001])
def test_stt_unsupported_resampling_before_array_work(
    monkeypatch: pytest.MonkeyPatch, sample_rate: int
) -> None:
    """Reject unsupported rates/factors before scanning or resampling tiny arrays.

    Args:
        monkeypatch:
            Records scans, resampling and mock neural stages.

        sample_rate:
            Invalid range, huge coprime rate, or excessive reduced factor;
            16004 reduces to 4000/4001, one beyond the factor budget.

    """
    stt = _recognizer(STTConfig())
    scan, resample = Mock(), Mock()
    monkeypatch.setattr(stt2.np, "isfinite", scan)
    monkeypatch.setattr(stt2, "resample_poly", resample)
    with pytest.raises(ValueError, match="sample_rate|resampling factors"):
        stt.transcribe_samples(np.zeros(1, np.float32), sample_rate)
    scan.assert_not_called()
    resample.assert_not_called()
    assert isinstance(stt.frontend, Mock) and isinstance(stt.engine, Mock)
    stt.frontend.extract.assert_not_called()
    stt.engine.infer.assert_not_called()
    assert stt.last_raw_text is None and stt.last_timings == {}


@pytest.mark.parametrize("sample_rate", [1, 7999, 192001, 1_000_000_007, 16004, 16001])
def test_stt_unsupported_file_rate_before_decode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sample_rate: int
) -> None:
    """Reject unsupported file rates/factors from metadata before waveform decode.

    Args:
        monkeypatch:
            Supplies tiny metadata and records waveform decoding/resampling.

        tmp_path:
            Isolated existing path whose bytes are never decoded.

        sample_rate:
            Unsupported source sample rate in Hz, including one-over boundaries.

    """
    stt = _recognizer(STTConfig())
    path = tmp_path / "unsupported-rate.wav"
    path.touch()
    monkeypatch.setattr(
        stt2.sf,
        "info",
        Mock(
            return_value=SimpleNamespace(frames=1, samplerate=sample_rate, channels=1)
        ),
    )
    read, resample, scan = Mock(), Mock(), Mock()
    monkeypatch.setattr(stt2.sf, "read", read)
    monkeypatch.setattr(stt2, "resample_poly", resample)
    monkeypatch.setattr(stt2.np, "isfinite", scan)
    with pytest.raises(ValueError, match="sample_rate|resampling factors"):
        stt.transcribe(path)
    read.assert_not_called()
    resample.assert_not_called()
    scan.assert_not_called()
    assert stt.last_raw_text is None and stt.last_timings == {}


@pytest.mark.parametrize("channels", [1, 2, 3])
def test_stt_file_scalar_sample_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, channels: int
) -> None:
    """Count every channel before decoding and permit exact scalar-sample totals.

    Args:
        monkeypatch:
            Supplies tiny in-memory soundfile metadata and decoded PCM.

        tmp_path:
            Isolated existing path whose contents are not decoded.

        channels:
            Number of source channels contributing to the allocation budget.

    """
    stt = _recognizer(STTConfig(max_input_samples=600 * channels))
    path = tmp_path / "audio.wav"
    path.touch()
    metadata = SimpleNamespace(frames=600, samplerate=16000, channels=channels)
    monkeypatch.setattr(stt2.sf, "info", Mock(return_value=metadata))
    read = Mock(return_value=(np.zeros((600, channels), np.float32), 16000))
    monkeypatch.setattr(stt2.sf, "read", read)
    assert stt.transcribe(path) == "ok"
    read.reset_mock()
    metadata.frames += 1
    with pytest.raises(ValueError, match="max_input_samples"):
        stt.transcribe(path)
    read.assert_not_called()
    assert stt.last_raw_text is None and stt.last_timings == {}


@pytest.mark.parametrize("channels", [0, -1, True, 2.0, 601])
def test_stt_invalid_file_channels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, channels: Any
) -> None:
    """Invalid or unbounded channel metadata never reaches waveform allocation.

    Args:
        monkeypatch:
            Supplies invalid metadata and records decoder requests.

        tmp_path:
            Existing fixture path without real audio bytes.

        channels:
            Invalid channel metadata, including a huge zero-frame layout.

    """
    stt = _recognizer(STTConfig(max_input_samples=600))
    path = tmp_path / "audio.wav"
    path.touch()
    monkeypatch.setattr(
        stt2.sf,
        "info",
        Mock(
            return_value=SimpleNamespace(frames=0, samplerate=16000, channels=channels)
        ),
    )
    read = Mock()
    monkeypatch.setattr(stt2.sf, "read", read)
    with pytest.raises(ValueError, match="channels"):
        stt.transcribe(path)
    read.assert_not_called()


@pytest.mark.parametrize(
    "shape,rate", [((601, 1), 16000), ((600, 2), 16000), ((600, 1), 24000)]
)
def test_stt_decoded_metadata_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, shape: tuple[int, int], rate: int
) -> None:
    """Reject inconsistent decoded layouts/rates before downmix or model stages.

    Args:
        monkeypatch:
            Supplies metadata and a deliberately inconsistent decoded result.

        tmp_path:
            Isolated existing fixture path.

        shape:
            Returned decoded waveform shape (frames, channels).

        rate:
            Returned decoded sample rate in Hz.

    """
    stt = _recognizer(STTConfig(max_input_samples=600))
    path = tmp_path / "audio.wav"
    path.touch()
    monkeypatch.setattr(
        stt2.sf,
        "info",
        Mock(return_value=SimpleNamespace(frames=600, samplerate=16000, channels=1)),
    )
    monkeypatch.setattr(
        stt2.sf, "read", Mock(return_value=(np.zeros(shape, np.float32), rate))
    )
    with pytest.raises(ValueError, match="does not match file metadata"):
        stt.transcribe(path)
    assert isinstance(stt.frontend, Mock)
    stt.frontend.extract.assert_not_called()


def test_stt_collapsed_token_boundary() -> None:
    """Count collapsed nonblank tokens, including blank-separated repeat IDs."""
    stt = _recognizer(STTConfig(max_output_tokens=3))
    assert stt.transcribe_samples(np.zeros(400, np.float32), 16000) == "ok"
    assert isinstance(stt.tokenizer, Mock)
    stt.tokenizer.decode.assert_called_once_with([1, 1, 2])
    stt.config = STTConfig(max_output_tokens=2)
    stt.tokenizer.decode.reset_mock()
    with pytest.raises(ValueError, match="max_output_tokens"):
        stt.transcribe_samples(np.zeros(400, np.float32), 16000)
    stt.tokenizer.decode.assert_not_called()
    assert stt.last_raw_text is None and stt.last_timings == {}


def test_stt_raw_character_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bound raw metadata-rich decode before canonicalization or audit retention.

    Args:
        monkeypatch:
            Records canonicalization to prove over-limit raw text skips it.

    """
    stt = _recognizer(STTConfig(max_output_chars=7), raw="<|en|>x")
    assert stt.transcribe_samples(np.zeros(400, np.float32), 16000) == "x"
    assert stt.last_raw_text == "<|en|>x"
    stt.config = STTConfig(max_output_chars=6)
    canonical = Mock()
    monkeypatch.setattr(stt2, "canonical_text", canonical)
    with pytest.raises(ValueError, match="raw output exceeds max_output_chars"):
        stt.transcribe_samples(np.zeros(400, np.float32), 16000)
    canonical.assert_not_called()
    assert stt.last_raw_text is None and stt.last_timings == {}


def test_stt_canonical_character_boundary() -> None:
    """Recheck canonical length because Unicode compatibility expansion adds chars."""
    stt = _recognizer(STTConfig(max_output_chars=3), raw="ﬃ")
    assert stt.transcribe_samples(np.zeros(400, np.float32), 16000) == "ffi"
    stt.config = STTConfig(max_output_chars=2)
    with pytest.raises(ValueError, match="canonical output exceeds max_output_chars"):
        stt.transcribe_samples(np.zeros(400, np.float32), 16000)
    assert stt.last_raw_text is None and stt.last_timings == {}


def test_stt_decode_failure_clears_audit() -> None:
    """Tokenizer exceptions propagate without retaining prior request audit text."""
    stt = _recognizer(STTConfig())
    assert isinstance(stt.tokenizer, Mock)
    stt.tokenizer.decode.side_effect = RuntimeError("decode failed")
    with pytest.raises(RuntimeError, match="decode failed"):
        stt.transcribe_samples(np.zeros(400, np.float32), 16000)
    assert stt.last_raw_text is None and stt.last_timings == {}


def test_stt_missing_file_clears_audit(tmp_path: Path) -> None:
    """A new file request clears prior audit text even when the path is absent.

    Args:
        tmp_path:
            Isolated directory containing no requested waveform.

    """
    stt = _recognizer(STTConfig())
    with pytest.raises(FileNotFoundError):
        stt.transcribe(tmp_path / "missing.wav")
    assert stt.last_raw_text is None and stt.last_timings == {}


class _UnstrippableText(str):
    """Raw text fixture proving a rejected request never strips its contents."""

    def strip(self, chars: str | None = None) -> str:
        """Fail if stripping happens before the raw character-budget rejection.

        Args:
            chars:
                Optional characters to strip, unused because stripping must fail.

        """
        raise AssertionError("Over-limit raw text was stripped")


def test_tts_raw_character_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    """Count whitespace before stripping, preparation and loading any language.

    Args:
        monkeypatch:
            Records text preparation and model loading after an exact-limit call.

    """
    tts = _synthesizer(TTSConfig(max_input_chars=4))
    assert tts.synthesize(" Hi ")[0].size == 9
    prepare, load = Mock(), Mock()
    monkeypatch.setattr(tts_engine, "prepare_speech_text", prepare)
    monkeypatch.setattr(TTS, "_load", load)
    with pytest.raises(ValueError, match="max_input_chars"):
        tts.synthesize(_UnstrippableText(" Hi  "))
    prepare.assert_not_called()
    load.assert_not_called()
    assert tts.last_timings == {}
    resources = tts._resources["EN"]
    for stage in resources.stages.values():
        assert isinstance(stage.compiled, Mock)
        stage.compiled.release_memory.assert_not_called()


@pytest.mark.parametrize("speed,expected", [(0.5, 18), (1.0, 9), (2.0, 6)])
def test_tts_audio_duration_boundary(
    monkeypatch: pytest.MonkeyPatch, speed: float, expected: int
) -> None:
    """Permit exact sample duration including speed-adjusted trailing gaps.

    Args:
        monkeypatch:
            Records expansion and downstream neural stages after the valid call.

        speed:
            Inverse duration scale affecting both phone durations and gaps.

        expected:
            Exact total waveform scalar samples at the fixture's 100-Hz rate.

    """
    tts = _synthesizer(TTSConfig(speed=speed, max_audio_seconds=expected / 100))
    audio, rate = tts.synthesize("Hi")
    assert rate == 100 and audio.shape == (expected,)
    stages = tts._resources["EN"].stages
    for stage in stages.values():
        assert isinstance(stage, Mock)
        stage.run.reset_mock()
    expand = Mock()
    monkeypatch.setattr(tts_engine, "expand_prior", expand)
    tts.config = TTSConfig(speed=speed, max_audio_seconds=(expected - 1) / 100)
    with pytest.raises(ValueError, match="max_audio_seconds"):
        tts.synthesize("Hi")
    expand.assert_not_called()
    for name in ("flow", "decoder"):
        stage = stages[name]
        assert isinstance(stage, Mock)
        stage.run.assert_not_called()
    assert tts.last_timings == {}


@pytest.mark.parametrize("budget", ["max_total_frames", "max_audio_seconds"])
def test_tts_cumulative_budget_before_expansion(
    monkeypatch: pytest.MonkeyPatch, budget: str
) -> None:
    """Reject the second segment before its expansion/flow/decoder/concatenation.

    Args:
        monkeypatch:
            Records real sparse expansions and final waveform concatenation.

        budget:
            Cumulative frame or complete waveform duration budget to exhaust.

    """
    config = TTSConfig(max_total_frames=4, max_audio_seconds=0.18)
    tts = _synthesizer(config, segments=2)
    assert tts.synthesize("Hi. Hi.")[0].size == 18
    tts.config = (
        TTSConfig(max_total_frames=3)
        if budget == "max_total_frames"
        else TTSConfig(max_audio_seconds=0.17)
    )
    stages = tts._resources["EN"].stages
    for stage in stages.values():
        assert isinstance(stage, Mock)
        stage.run.reset_mock()
    expand = Mock(wraps=tts_engine.expand_prior)
    concatenate = Mock()
    rng = np.random.default_rng(0)
    monkeypatch.setattr(tts_engine.np.random, "default_rng", Mock(return_value=rng))
    monkeypatch.setattr(tts_engine, "expand_prior", expand)
    monkeypatch.setattr(tts_engine.np, "concatenate", concatenate)
    with pytest.raises(ValueError, match="total-frame|max_audio_seconds"):
        tts.synthesize("Hi. Hi.")
    assert expand.call_count == 1
    concatenate.assert_not_called()
    for name in ("flow", "decoder"):
        stage = stages[name]
        assert isinstance(stage, Mock)
        assert stage.run.call_count == 1


def test_tts_segment_frame_limit_before_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retain the per-segment frame bound and reject before sparse expansion.

    Args:
        monkeypatch:
            Records expansion for exact and one-over-frame limits.

    """
    tts = _synthesizer(TTSConfig(max_frames=2))
    assert tts.synthesize("Hi")[0].size == 9
    tts.config = TTSConfig(max_frames=1)
    expand = Mock()
    monkeypatch.setattr(tts_engine, "expand_prior", expand)
    with pytest.raises(ValueError, match="Generated frames"):
        tts.synthesize("Hi")
    expand.assert_not_called()


class _NoCopyArray(np.ndarray):
    """Waveform fixture that fails if an invalid decoder output is copied."""

    def copy(self, order: str = "C") -> NDArray[Any]:
        """Fail on copying malformed decoder output before shape rejection.

        Args:
            order:
                Requested NumPy memory layout, unused because copying must fail.

        """
        raise AssertionError("Oversized decoder output was copied")


def test_tts_oversized_decoder_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    """Malformed decoder waveforms are rejected before copying or concatenation.

    Args:
        monkeypatch:
            Records final waveform concatenation without allocating model data.

    """
    tts = _synthesizer(TTSConfig(max_audio_seconds=0.09))
    decoder = tts._resources["EN"].stages["decoder"]
    assert isinstance(decoder, Mock)
    decoder.run.side_effect = None
    decoder.run.return_value = [np.zeros((1, 1, 5), np.float32).view(_NoCopyArray)]
    concatenate = Mock()
    rng = np.random.default_rng(0)
    monkeypatch.setattr(tts_engine.np.random, "default_rng", Mock(return_value=rng))
    monkeypatch.setattr(tts_engine.np, "concatenate", concatenate)
    with pytest.raises(RuntimeError, match="Unexpected decoder waveform shape"):
        tts.synthesize("Hi")
    concatenate.assert_not_called()
    assert tts.last_timings == {}


def test_tts_decoder_failure_keeps_default_release_disabled() -> None:
    """Propagate model failure, clear audit metrics and retain explicit release policy."""
    tts = _synthesizer(TTSConfig())
    tts.last_timings = {"prior": "request"}
    decoder = tts._resources["EN"].stages["decoder"]
    assert isinstance(decoder, Mock)
    decoder.run.side_effect = RuntimeError("decoder failed")
    with pytest.raises(RuntimeError, match="decoder failed"):
        tts.synthesize("Hi")
    assert tts.last_timings == {}
    for stage in tts._resources["EN"].stages.values():
        assert isinstance(stage.compiled, Mock)
        stage.compiled.release_memory.assert_not_called()


def test_stt_new_limit_failure_preserves_explicit_release_opt_in() -> None:
    """An explicitly enabled release still runs when a new source budget rejects."""
    stt = _recognizer(STTConfig(max_input_samples=400, release_after_call=True))
    with pytest.raises(ValueError, match="max_input_samples"):
        stt.transcribe_samples(np.zeros(401, np.float32), 16000)
    assert isinstance(stt.engine, Mock)
    stt.engine.release_memory.assert_called_once()
    stt.engine.infer.assert_not_called()
    assert stt.last_raw_text is None and stt.last_timings == {}


def test_tts_new_limit_failure_preserves_explicit_release_opt_in() -> None:
    """An explicitly enabled release still runs when raw character validation fails."""
    tts = _synthesizer(TTSConfig(max_input_chars=1, release_after_call=True))
    with pytest.raises(ValueError, match="max_input_chars"):
        tts.synthesize("Hi")
    for stage in tts._resources["EN"].stages.values():
        assert isinstance(stage, Mock)
        stage.run.assert_not_called()
        stage.compiled.release_memory.assert_called_once()
    assert tts.last_timings == {}


def test_tts_invalid_duration_before_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid generated durations fail before expansion or waveform generation.

    Args:
        monkeypatch:
            Records expansion requests for invalid encoder duration outputs.

    """
    tts = _synthesizer(TTSConfig())
    stages = tts._resources["EN"].stages
    encoder = stages["encoder"]
    assert isinstance(encoder, Mock)
    encoder.run.return_value[2].fill(np.nan)
    expand = Mock()
    monkeypatch.setattr(tts_engine, "expand_prior", expand)
    with pytest.raises(RuntimeError, match="Invalid generated duration"):
        tts.synthesize("Hi")
    expand.assert_not_called()
    for name in ("flow", "decoder"):
        stage = stages[name]
        assert isinstance(stage, Mock)
        stage.run.assert_not_called()
