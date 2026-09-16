# Local speech

`hoast.tts.TTS` runs Kokoro with **a CPU graph and zero-copy GPU decoder convolution**
on the Intel UHD 630. It is the single maintained TTS backend. Decoder weights are
packed INT8; GPU tap sums use FP16 with FP32 accumulation. The remaining graph uses
OpenVINO CPU, including the verified AVX2 Snake activation where prepared.

`hoast.stt.STT` uses faster-whisper small with CTranslate2 CPU INT8. Both engines
default to two inference threads; Whisper uses one model worker. CLI entrypoints
enforce a shared two-core process affinity
budget. GPU/device failures propagate; inference does not download missing models.

## Prepare

`./prepare.sh` bootstraps uv, prepares all default models, and runs the bilingual
correctness check. For speech-only preparation from the repository root:

```sh
uv sync --locked
uv run python -m tools.prepare_stt
uv run python -m tools.guarded_run --threads 2 --memory-gib 4 --timeout 900 -- .venv/bin/python -m tools.prepare_tts --chinese
```

Omit `--chinese` for English-only preparation. TTS preparation verifies the published
FP32 Kokoro v1.0 ONNX and matching voices, builds the CPU activation library, and
checks the hybrid runtime. The runtime quantizes decoder weights itself; an INT8
ONNX export is not required. Preparation needs a C++17 compiler; inference needs
the Intel OpenCL driver. PyOpenCL is part of the main Python 3.14 environment.

Chinese preparation provisions the pinned `tools/speech_env` Python 3.12 environment
and exports the official Chinese v1.1 model. Only its G2P worker uses Python 3.12;
both neural models run through the same hybrid runtime in the main process.

Chinese exports are validated in a private staging directory before publication.
The readiness manifest is published last and records artifact hashes, allowing
verified exports to be reused on later preparation runs. An interrupted publication
leaves no readiness manifest, so an incomplete artifact set cannot be loaded later.

The export/G2P environment remains on Python 3.12 because Misaki 0.9.4 declares
Python `<3.13`; Kokoro's upstream main branch also excludes Python 3.14. Moving
these components to 3.14 requires validating and updating their dependency support,
not merely changing this project's interpreter setting.

Default artifacts:

- English model: `.cache/hoast/tts/kokoro-v1.0.onnx`
- English voices: `.cache/hoast/tts/voices-v1.0.bin`
- Chinese artifacts: `.cache/hoast/tts/chinese/`
- Chinese G2P interpreter: `.cache/hoast/speech-env/.venv/bin/python`
- STT model: `.cache/hoast/stt/small/`
- CPU activation library: `.cache/hoast/tts/cpu-kernels/`

Historical split-model exports and the profiling environment are not needed.

## Use

```sh
uv run python -m hoast.tts "The weather is sunny." --output /tmp/speech.wav
uv run python -m hoast.stt /tmp/speech.wav
uv run python -m hoast.tts "Hello world. 你好，欢迎回家。" --output /tmp/chinese.wav
uv run python -m hoast.stt /tmp/chinese.wav --language zh
```

TTS writes mono 24 kHz PCM16 WAV. Text may also be piped through stdin. It defaults
to voice `af_heart`, English phonemization `en-us`, speed `1.0`, and two threads.
Speed is bounded to `[0.5, 2.0]`. `--help` displays options and defaults. Diagnostic
details go to `.cache/hoast/diagnostics/`; stdout stays concise.

Han-containing utterances use Chinese voice `zf_001`, including embedded English
spans. The Chinese model, vocabulary, style table and official Misaki G2P worker
initialize on first use and remain cached. Both models share one GPU context and
kernel cache. English-only requests never initialize Chinese resources. Unknown
phonemes fail rather than being discarded; long input is split at bounded phoneme
boundaries. Chinese style lookup follows the official **N−1** convention.

STT accepts PyAV-supported audio files and resamples/downmixes to mono 16 kHz.
English recognition is the default; use `--language zh` or `auto` as appropriate.
The eight-second minimum encoder context avoids unnecessary short-input padding;
uncertain decoding retries the full context. `--encoder-min-seconds 30` requests
the full-padding reference. Silence retains an empty transcript.

## Reuse loaded engines

```python
from hoast.runtime import configure_cpu_budget
from hoast.stt import STT
from hoast.tts import TTS

configure_cpu_budget(2)
tts = TTS()
try:
    stt = STT()
    samples, sample_rate = tts.synthesize("The weather is sunny.")
    transcript = stt.transcribe_samples(samples, sample_rate)
finally:
    tts.close()
```

Samples are mono float32 shaped `(samples,)`. The in-memory handoff uses polyphase
resampling without WAV encoding. TTS serializes synthesis and closure; `close()`
releases compiled models, the Chinese worker and GPU caches and rejects later
synthesis on that instance. Keep an instance alive to amortize initialization.

`tts.play_samples(samples, sample_rate, blocking=False)` accepts raw mono float32
PCM and submits it to the **same ordered queue** as `tts.play(text, blocking=False)`.
Source rates such as 16 kHz are polyphase-resampled to 24 kHz once per complete
submission before queuing; 24 kHz input bypasses conversion. Raw input must be
nonempty and finite. Call `tts.wait_playback()` to drain the queue and close its
device; subsequent submissions can open a new queue. Queue capacity must remain
consistent until drained. An optional `cancelled=threading.Event()` stops new raw
submission between 100 ms blocks and blocking calls drain accepted audio.

For raw playback without synthesis models, use `TTS(playback_only=True)`. It skips
artifact and GPU initialization, supports `play_samples`/`wait_playback`/`close`,
and rejects synthesis. `close()` rejects later raw playback as well as synthesis.

On Linux, playback prefers an output-capable ALSA `pipewire` adapter, then `pulse`,
so desktop audio is mixed with other applications instead of competing for direct
ALSA hardware. If neither adapter is advertised, PortAudio's system default is
used. An open failure is logged and propagated without retrying another route.

`hoast/tts_gpu.py` selects exactly 67 substantial decoder convolutions from the
69-convolution decoder in each full prepared model. Frontend and tiny filters stay
on CPU. The single `hoast/kernels/conv_window.cl` kernel uses SIMD16 and adaptive
nonspilling 32/16/8 time tiles. Lengths are runtime arguments. Shared host imports,
buffer overlap, operation identity and resource release are explicitly checked.

## Compact correctness check

```sh
uv run python -m tools.guarded_run --threads 2 --memory-gib 4 --timeout 240 -- .venv/bin/python -m tools.check_speech --chinese
```

The check uses tiny embedded kernel fixtures with exact INT8-representable weights,
odd channels, stride, dilation and two tail lengths. Relative RMS arithmetic error
must remain below 0.3%. The fixtures exercise their selected tiles, while offline
tests cover tile fallback. Exact fixture weights isolate arithmetic rather than
weight-quantization error. The check then synthesizes and recognizes two English
utterances; `--chinese` adds Mandarin and mixed text plus lazy-cache checks. Transcript matching
normalizes punctuation/case and explicit Simplified/Traditional equivalents without
discarding words. Audio, the latest kernel fixture and complete failures are retained
under `.cache/hoast/diagnostics/speech-check/`.

This check requires prepared local models and the GPU, but no performance-counter
permissions or profiling packages. Offline `pytest` tests use synthetic data and
mocked GPU imports to cover ownership, cleanup, routing and graph contracts.

## Analysis and journey

- [Speech optimization study](speech-study.md): CPU/STT tuning and intermediate comparisons.
- [Kokoro FP16 investigation](kokoro-fp16.md): export correctness, GPU limitations and memory findings.
- [Hybrid convolution investigation](kokoro-hybrid-conv.md): kernel iterations, roofline/ISA/counter analysis and measured gains.

Those reports preserve measurements from their stated configurations. Cleanup
reuses the final kernel on full English/Chinese models and validates correctness
through the public API; it does not relabel historical measurements as new results.
