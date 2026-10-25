# Local speech

Hoast supports two speech engines:

- `hoast.stt2.STT`: SenseVoiceSmall with a repository-owned NumPy/SciPy
  FBANK/LFR/CMVN frontend and local OpenVINO artifacts. The pinned community
  ONNX Runtime candidate is an explicit opt-in backend.
- `hoast.tts2.TTS`: English/Chinese Melo with native normalization, G2P and
  POS tagging, and four resident OpenVINO stages. Runtime synthesis does not
  launch an alternate interpreter or import Melo's upstream frontend.

The qualified application path is CPU speech with a shared two-thread budget:
FP32 SenseVoice and decoder-only FP16 Melo. Language routing and playback use
shared model-independent helpers. LFM GPU selection does not move speech to GPU.

Voice startup speech and assistant replies use `TTSConfig(speed=0.85)`, about
15% slower than the standalone API/benchmark default of `1.0`. Speed controls
generated phone durations, not playback sample rate. English and Chinese sentence
splitting preserves decimal points and numeric grouping commas for subsequent
number normalization; English currency expansion leaves sentence punctuation
outside the amount.

## Preparation and use

Use [`prepare.sh`](../prepare.sh) for pinned input acquisition and serial export.
`--offline` validates local pinned inputs without downloads. Preparation refuses
existing export targets; provide fresh output roots rather than replacing models
in use. See the [native architecture and preparation guide](speech2-native.md)
for resource pins, legal notices, backend options and explicit conversion commands.

```sh
./prepare.sh --offline
.venv/bin/python -m tools.check_speech2 --threads 2 --chinese
```

`tools.check_speech2` performs real bilingual round-trip inference without
deployment or playback. Supply `--stt-model-dir` and `--tts-model-dir` to select
explicit exports in either that tool or the application. `--help` describes
supported controls. The engines are reusable Python APIs, not standalone
module CLIs:

```python
from pathlib import Path
from hoast.stt2 import STT, STTConfig
from hoast.tts2 import TTS, TTSConfig

with TTS(TTSConfig(threads=2)) as synthesizer:
    synthesizer.write_wav("The weather is sunny.", Path("/tmp/speech.wav"))
with STT(STTConfig(threads=2)) as recognizer:
    text = recognizer.transcribe(Path("/tmp/speech.wav"))
```

## Endpoint detection and dependencies

Streaming endpoint detection uses the separately prepared Silero v6 ONNX model
at `models/vad/silero_vad_v6.onnx`, with per-stream context and recurrent states.
`tools.prepare_vad` verifies pinned model bytes and preserves its license and
provenance. Runtime performs no download. Whisper, CTranslate2 and Kokoro are
not application dependencies. ONNX Runtime also serves the optional STT2
community backend.

## Evidence and development

- [Native architecture, pinned resources and preparation](speech2-native.md)
- [Speech2 benchmark evidence](speech2-benchmarks.md), including historical
  Whisper/Kokoro comparisons. Those baselines are measurements, not runnable
  backends in this repository.
- [Network microphone development](voice-satellite.md)

Permanent model assets and their legal/frontend resources live under
`models/stt2`, `models/tts2` and `models/vad`. Compilation/download caches are
disposable. Keep benchmark evidence outside `.cache` before resetting it.

## Intermediate-memory release

STT2 and TTS2 expose `release_memory()` and the configuration flag
`release_after_call=True` for explicit OpenVINO experiments. Cleanup happens
after the complete utterance's borrowed outputs have been consumed or copied,
including failed calls. Successful call timings include cleanup. ORT cannot
release this storage and rejects automatic release.

Automatic release remains **disabled** in the library and voice application.
The local two-thread CPU benchmark passed every median-latency screen and
preserved exact transcripts/waveforms, but failed the predeclared p95 screens:
SenseVoice Mandarin cases increased about 41 ms (12–20%), and Melo English-long
and Mandarin-short increased about 104–112 ms. There was no consistent idle
RSS/PSS/USS reduction to justify enabling it. These small-sample tail results
are a conservative deployment screen, not proof that release caused every
outlier. See the [methodology and complete results](memory-release-benchmark.md).

This API releases eligible OpenVINO compiled-model intermediates, not model
weights, all allocator/driver pages, or LLM KV pools. OpenVINO GenAI's
`LLMPipeline` does not expose a general `release_memory()` operation. GPU speech
release is not latency-qualified by the CPU measurements.
