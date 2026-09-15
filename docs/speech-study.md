# Speech optimization study

This document preserves the investigation's intermediate configurations,
measurements and commands. Benchmark and experiment scripts mentioned here have
been retired. For the maintained hybrid runtime, preparation and compact checks,
see [Local speech](speech.md). The final GPU investigation is documented in
[Kokoro hybrid convolution](kokoro-hybrid-conv.md).

`hoast/stt.py` provides a reusable `STT` recognizer using faster-whisper **small**
with CTranslate2 CPU INT8. `hoast/tts.py` provides a reusable `TTS` synthesizer
using **Kokoro v1.0 FP32 through OpenVINO CPU** by default. ONNX Runtime CPU and
the INT8 export are selectable. Both modules also run as file-based testing CLIs.
Both engines default to two inference threads.

The selected CPU path combines optional verified AVX2 Snake activation fusion
with an eight-second minimum Whisper encoder context. Han-containing utterances
automatically select the official Chinese Kokoro v1.1 bilingual voice, initialized
and cached on demand. See the current results below; subsequent comparison
sections record baseline experiments with full 30-second encoder padding.

## Selected settings and results

Measurements on the i5-8500T/UHD 630 use serial inference and strict one/two-core
affinity. For the 72-character weather response used below:

| CPU cores | TTS median | STT median | Round-trip median |
| --------- | ---------- | ---------- | ----------------- |
| 2         | 2.1645 s   | 0.5570 s   | 2.7478 s          |
| 1         | 4.0967 s   | 0.9411 s   | 5.0508 s          |

Two cores are selected. Stage and total medians need not sum exactly. Results are
in `.cache/hoast/diagnostics/speech-tuning/final-two-threads.log` and
`final-one-thread.log`. Reproduce with `tools.benchmark_speech` and the sentence
“The weather is sunny, twenty degrees, with a ten percent chance of rain.”

Whisper crops padded feature columns conservatively using original audio duration
plus 0.5 seconds, at least eight seconds and at most thirty. Low-confidence,
repetitive, or unexpectedly empty decoding retries the full context. The
16-case comparison (12 generated commands, three noisy variants, and a public
JFK recording) found no transcript regressions versus full padding. A four-second
minimum regressed and was rejected. `--encoder-min-seconds 30` selects the full
reference; default `8` is a latency/accuracy compromise, not a universal guarantee.

The optional OpenVINO CPU extension fuses exactly `x + scale*sin(alpha*x)^2`.
Preparation builds it with a C++17 compiler, AVX2, OpenVINO/TBB and glibc vector
math. Scalar/channel broadcasting and vector/tail paths are numerically checked.
Source/library hashes and the OpenVINO version must match before loading.
Missing optional libraries use native operations; mismatched existing libraries
fail and require `tools.build_speech_kernels`. `--no-fast-activations` disables
fusion in the TTS CLI (`TTSConfig(activation_kernel=None)` in Python).

GPU speech and hybrid partitions did not justify deployment: Whisper GPU and
GPU-encoder/CPU-decoder paths were slower, while Kokoro hybrids paid substantial
new-length specialization costs. The measured CPU path is selected.

An additional [custom-convolution investigation](kokoro-hybrid-conv.md) demonstrates
zero-copy CPU-graph/GPU-convolution execution without those specialization stalls.
Its experimental INT8-weight kernel improves tested TTS latency and CPU time;
the report includes measured hardware counters, accuracy checks and reproduction.

## Chinese and mixed-language speech

```sh
uv run python -m tools.guarded_run --threads 2 --memory-gib 4 --timeout 900 -- .venv/bin/python -m tools.prepare_tts --chinese
uv run python -m hoast.tts "Hello world. 你好，欢迎回家。" --output /tmp/chinese.wav
uv run python -m hoast.stt /tmp/chinese.wav --language zh
uv run python -m tools.benchmark_speech "你好，今天天气很好。" --stt-language zh
```

Chinese artifacts are pinned to `hexgrad/Kokoro-82M-v1.1-zh` revision
`01e7505bd6a7a2ac4975463114c3a7650a9f7218`; default voice is `zf_001`.
They live in `.cache/hoast/tts/chinese/`. The checkpoint SHA256 is
`b1d8410fa44dfb5c15471fd6c4225ea6b4e9ac7fa03c98e8bea47a9928476e2b`.
The locked Python 3.12 environment in `.cache/hoast/speech-env/.venv` runs the
official Misaki v1.1 G2P as a persistent JSON-lines worker. Model inference stays
in the application's Python 3.14 process, through OpenVINO CPU. Preparation
downloads dependencies; runtime stays local. Override locations using
`--chinese-model-dir`, `--chinese-python`, and `--chinese-voice`.

Any Han character, including supplementary/compatibility ideographs and 〇,
routes the whole utterance to one Chinese bilingual voice. English spans use
English phonemes. English-only utterances retain their configured voice. A
tested sentence-by-sentence voice switch lost “Hello world” during recognition;
one bilingual voice preserved it. This is an ASR round-trip check, not a
subjective listening evaluation. Chinese script equivalents are normalized
explicitly; missing words are not accepted.

The Chinese model, vocabulary, style table and G2P worker are loaded lazily and
reused per `TTS` instance. `TTS.close()` releases Chinese resources; subsequent
Chinese synthesis can initialize them again. Worker requests/replies are bounded
to one MiB with separate write/read deadlines; failed requests close the worker.
Payload batches prefer 120 phonemes, preserve tone/word boundaries and use the
official **N−1** style row. Unknown phonemes fail instead of being dropped.

Run the maintained prepared-model check under the resource guard:

```sh
uv run python -m tools.guarded_run --threads 2 --memory-gib 4 --timeout 240 -- .venv/bin/python -m tools.verify_chinese_speech
```

It checks four Mandarin/mixed utterances against both eight- and thirty-second
STT contexts, English-only lazy behavior, cached reuse and long-input waveform
completion. Long-input checking establishes valid bounded synthesis, not exact
recognition accuracy. Audio, results and complete exceptions are retained under
`.cache/hoast/diagnostics/chinese-final/`.

All four cases passed. The first Chinese synthesis, including lazy setup, took
6.11 seconds; the subsequent three short utterances took 1.18–1.64 seconds each
(single calls, not warm medians). A 129-character input produced 27.825 seconds
of valid audio in 13.432 seconds. The guarded verification process tree peaked at
about 2.98 GiB RSS. A separate PCM16 file-CLI round trip retained the mixed-language
words; its short-context confidence check triggered the full-context retry.

The guard uses a systemd user scope with a hard memory limit, no swap, a 3 GiB
available-memory reserve watchdog and a timeout. Run model experiments serially.
The memory limit includes descendants, including the Chinese worker. Full-decoder
activation calibration exceeded host memory and is excluded from the workflow.

## Initialize

Run from the repository root:

```sh
uv sync --locked
uv run python -m tools.prepare_stt
uv run python -m tools.prepare_tts
```

Preparation downloads public artifacts and verifies model loading; TTS also
synthesizes a warmup to check the bundled eSpeak NG phonemizer. No separate
FFmpeg or eSpeak installation is needed with the locked Linux wheels.
STT is pinned to `Systran/faster-whisper-small` revision
`536b0662742c02347bc0e980a01041f333bce120`. It downloads about 461 MiB; CTranslate2
loads the checkpoint with INT8 compute. Kokoro's matched `model-files-v1.0`
release contains an approximately 310 MiB FP32 graph, 88 MiB INT8 graph and
27 MiB voice archive. Preparation defaults to FP32 because it is faster on this
CPU; only the selected graph and shared voice archive are downloaded.

Artifacts live in `.cache/hoast/stt/small/` and `.cache/hoast/tts/`. Preparation
supports `--output` and `--threads`; TTS also accepts `--runtime openvino` or
`--runtime onnxruntime`. Existing downloads are reused; Hugging Face
resumes STT downloads and Kokoro downloads verify pinned SHA256 hashes before
publishing through temporary files. A corrupt existing Kokoro file is rejected;
remove the named corrupt file and rerun preparation to download it again.
Runtime requires local artifacts, including Whisper's tokenizer, and does not
download missing models. Preparation and runtime failures retain full tracebacks
in `.cache/hoast/diagnostics/`.

## Test from the command line

```sh
uv run python -m hoast.tts "The weather is sunny." --output /tmp/speech.wav
uv run python -m hoast.stt /tmp/speech.wav

# Smaller inference thread budget:
uv run python -m hoast.tts "Hello." --threads 1 --output /tmp/speech.wav
uv run python -m hoast.stt /tmp/speech.wav --threads 1

# Read UTF-8 text from stdin:
uv run python -m hoast.tts --output /tmp/speech.wav < README.md

# Optional language detection or a wider decoding beam:
uv run python -m hoast.stt recording.wav --language auto --beam-size 5

# Optional smaller Kokoro INT8 graph (slower on this machine):
uv run python -m tools.prepare_tts --precision int8
uv run python -m hoast.tts "Hello." --output /tmp/speech.wav --model .cache/hoast/tts/kokoro-v1.0.int8.onnx
```

STT accepts formats supported by PyAV, including WAV, FLAC and MP3, downmixing
and resampling to mono 16 kHz. Stdout contains only the transcript and a newline;
silence filtered by VAD produces an empty transcript. English is the default.
Greedy decoding (`--beam-size 1`), temperature zero, no fallback sampling and
independent windows favor short-command latency. Text-only decoding is the
default; `--timestamp-decoding` enables internal timestamp tokens while retaining
text-only stdout. Wider beams cost more CPU.

TTS writes mono 24 kHz PCM16 WAV. It defaults to `--voice af_heart`,
`--language en-us`, and `--speed 1.0`; speed must be between 0.5 and 2.0.
Choose a matching voice and phonemizer language when overriding them; available
voice metadata is in the upstream [Kokoro voice guide](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md).
Long inputs are split into bounded phoneme batches without silently truncating
unpunctuated sentences. Model-generated boundary pauses are retained.

Both CLIs accept `--model` and `--log-file`; TTS also accepts `--voices` and
`--runtime openvino|onnxruntime` (CPU only, without automatic fallback).
Diagnostics and timing stay in log files, while TTS confirms that audio was saved.
The CLIs read/write files; audio-device capture and playback belong to the caller.

## Reuse loaded engines

```python
from hoast.stt import STT, STTConfig
from hoast.tts import TTS, TTSConfig

tts = TTS(TTSConfig(threads=2))
stt = STT(STTConfig(threads=2))
samples, sample_rate = tts.synthesize("The weather is sunny.")
transcript = stt.transcribe_samples(samples, sample_rate)
```

`synthesize` returns a mono float32 NumPy array shaped `(samples,)` and a sample
rate in Hz. Keep engines alive to amortize model loading and first-call setup;
serialize calls on each instance. Thread budgets apply **per engine**, not to
the total process when using the library directly. Call
`hoast.runtime.configure_cpu_budget(2)` before loading models to cap existing
native threads and future workers to the same two-core affinity mask. CLIs do
this automatically. The Chinese worker inherits that mask and selects one core
within it, so it does not add a third effective CPU core.

`transcribe_samples` accepts nonempty, finite mono float32 audio and an explicit
positive integer sample rate. It resamples Kokoro's 24 kHz output to 16 kHz using
SciPy's polyphase resampler; 16 kHz input passes through without a copy. This
avoids WAV encoding, decoding and disk I/O. The external input boundary scans
for NaN/Inf before inference. Do not mutate an input array during transcription.

Test the full resident-model path:

```sh
uv run python -m tools.benchmark_speech "The weather is sunny."
uv run python -m tools.benchmark_speech "The weather is sunny." --runtime onnxruntime
uv run python -m tools.benchmark_speech "The weather is sunny." --threads 1
```

One warmup precedes three measurements (`--repeats` overrides that count). Stdout
contains the final transcript. Per-stage times, warm medians, first-call latency
and model-load times go to `.cache/hoast/diagnostics/roundtrip.log`; `--log-file`
selects another file. This times synthesis plus recognition, not speaker playback
or microphone capture.

## CPU tuning

Target: Intel Core i5-8500T, six physical cores, AVX2, no AVX-512/VNNI or native
FP16 arithmetic. OpenVINO uses CPU, one stream/request, two inference threads,
the latency performance hint, FP32 floating-point hint and disabled CPU pinning.
ONNX Runtime uses its CPU provider, full graph optimizations, sequential graph
execution, one inter-op thread and bounded intra-op threads, with idle spinning
disabled. Whisper uses one INT8 worker; Silero VAD uses one inference thread.
No machine-wide affinity or environment changes are required.

Kokoro also publishes FP32 and FP16 graphs. FP16 is not native arithmetic on
this CPU, so it is not selected just for smaller storage. INT8 and FP32 are
compared using actual synthesis; quantization affects output quality as well as
speed. Timing logs report model load time, synthesis/transcription time, audio
duration and real-time factor (RTF: processing seconds / audio seconds; lower
is better). Whisper timings include complete consumption of lazy segments.

### Baseline ONNX Runtime reference measurements (full encoder padding)

Python 3.14.7, ONNX Runtime 1.30.0, kokoro-onnx 0.4.9, faster-whisper 1.2.1 and
CTranslate2 4.8.2. Each result is the median of three warm calls after one warmup,
with engines run serially. The short input was “The weather is sunny, twenty
degrees, with a ten percent chance of rain.” Kokoro used `af_heart`, speed 1.0,
and generated 4.525 seconds of audio; Whisper transcribed that FP32-generated WAV.

| Engine / precision  | Threads | Warm seconds | RTF   |
| ------------------- | ------- | ------------ | ----- |
| Kokoro FP32         | 1       | 4.438        | 0.981 |
| Kokoro FP32         | 2       | 2.500        | 0.552 |
| Kokoro INT8         | 1       | 8.923        | 1.972 |
| Kokoro INT8         | 2       | 7.425        | 1.641 |
| Whisper small INT8  | 1       | 5.200        | 1.149 |
| Whisper small INT8  | 2       | 2.894        | 0.640 |

For a 216-character response, two-thread Kokoro FP32 generated 13.275 seconds
of audio in 7.382 seconds (RTF 0.556); INT8 took 21.896 seconds for 13.225 seconds
of audio (RTF 1.656). FP32 was about **three times faster than INT8** at two
threads. Fresh engine construction, with files already in the filesystem cache,
took roughly 0.7–0.9 seconds for Kokoro and 0.8–1.0 seconds for Whisper. These
are not uncached disk-startup or end-to-end CLI timings.

ONNX `session.set_denormal_as_zero=1` was also tested in a fresh process. The
two-thread FP32 median was 2.446 seconds versus 2.500 without it, while INT8
remained slow at 7.573 seconds. This did not establish a meaningful gain, so the
runtime retains ONNX's default floating-point behavior. The separate published
FP16 ONNX artifact was not benchmarked; GPU precision-hint trials are below.

The model precisions are **FP32 for Kokoro**, **INT8 for Whisper**. Runtime
comparisons and resident-model defaults are detailed below. Keep models loaded
between requests; a new CLI invocation reloads them.
The synthesized-speech round trip recovered both test utterances, allowing normal
number/punctuation normalization. This is a functional smoke test; human-recorded
speech accuracy and subjective listening quality were not evaluated.

Final verification with network connections blocked passed synthesis,
transcription, stereo 48 kHz silence handling and malformed-audio rejection. A
repeat run measured two-thread Whisper at 2.789 seconds for the short utterance
and 3.163 seconds for the 13.275-second response. Both file CLIs passed a real
WAV round trip. Runtime-extension validation is described below.

Detailed measurements and first-call times are in
`.cache/hoast/diagnostics/speech-benchmark.log` and `speech-denormal.log`.
The local reproduction scripts are `/tmp/opencode/benchmark_speech.py` and
`/tmp/opencode/benchmark_denormal.py`; run them from the project root with
`PYTHONPATH=.` using `uv run python`. The denormal script injects its experimental
session option without changing production defaults.

### OpenVINO CPU and integrated GPU comparison

OpenVINO 2026.3.1 was compared with ONNX Runtime 1.30.0 on the same pinned
FP32/INT8 ONNX artifacts, `af_heart`, speed 1.0 and identical token/style inputs.
Each CPU configuration ran in a fresh process with one warmup and three timed
calls. Timings include the common phonemizer and output copying. The two-thread
round-trip comparison reversed the CPU runtime order as a confirmation.

| Runtime / export         | CPU threads | Short TTS seconds | Audio seconds |
| ------------------------ | ----------- | ----------------- | ------------- |
| ONNX Runtime / FP32      | 1           | 4.017             | 4.525         |
| ONNX Runtime / FP32      | 2           | 2.450             | 4.525         |
| ONNX Runtime / INT8      | 1           | 8.794             | 4.525         |
| ONNX Runtime / INT8      | 2           | 7.486             | 4.525         |
| OpenVINO CPU / FP32      | 1           | 4.425             | 4.525         |
| OpenVINO CPU / FP32      | 2           | 2.340             | 4.525         |
| OpenVINO CPU / INT8      | 1           | 4.957             | 4.575         |
| OpenVINO CPU / INT8      | 2           | 2.745             | 4.575         |
| OpenVINO iGPU / FP32 (*) | —           | 3.860             | 4.775         |

OpenVINO CPU FP32/two threads is the default for a resident engine. Its advantage
over ONNX Runtime FP32 is modest; ONNX Runtime was faster in the one-thread
test. OpenVINO CPU model setup took about two seconds in repeat checks versus
about 0.8 seconds for ONNX Runtime, so warm throughput and fresh-process startup
favor different choices. No persistent compiled-model cache was used.

**Why this INT8 export is slower under ONNX Runtime:** profiling two short calls
showed `ConvInteger` consuming **12.58 of 14.73 seconds of node execution time**,
about 85%. FP32 `Conv` took 2.87 seconds across two calls. Dynamic quantization
itself took only 0.076 seconds across two INT8 calls. The dominant problem is
this export's integer convolution execution path, rather than its smaller file
size or quantization/dequantization overhead alone. OpenVINO handles the INT8
export much faster, but FP32 remains faster in these measurements. “INT8” labels
identify the published export, not a claim that every operation executes in INT8.

#### Integrated GPU compatibility and timing

The device is **Intel UHD Graphics 630**. (*) The stock graph fails GPU compilation
because its 3D `linear_onnx` interpolation is unsupported. The experimental
benchmark lifts two interpolation nodes to equivalent 4D operations and squeezes
the added axis afterward. Its CPU output matched the unmodified graph exactly
for the test utterance. This rewrite is confined to the experimental benchmark.

With that rewrite, GPU FP32 runs and passes transcript checks, but is slower than
CPU. It uses one GPU stream and explicit `GPU.0` execution; CPU thread counts do
not describe GPU parallelism. Compilation took about 15 seconds, and the first
synthesis took about 35 seconds before later calls reached roughly 3.9 seconds.
Those startup/new-shape costs are excluded from warm medians.

For the published ONNX export, the FP16 execution-hint test compiled but failed
inference with incompatible matrix dimensions. The INT8 export failed GPU
compilation with an unsupported
integer-convolution layout under both FP32 and FP16 hints. Neither failed case
has a valid latency result. There was no automatic CPU fallback.

### Baseline TTS-to-STT round-trip latency (full encoder padding)

Both engines remain loaded. These medians include FP32 TTS, 24-to-16 kHz
in-memory resampling and faster-whisper small INT8 recognition with two CPU
threads, greedy text-only decoding and VAD. Short text is the 72-character
weather sentence above; long text is a 216-character weather/music reminder.
CPU audio durations are 4.525 and 13.275 seconds; GPU durations are 4.775 and
13.725 seconds. Complete transcripts matched after punctuation and number-form
normalization; these are synthesized-speech tests, not a human-speech evaluation.

| TTS runtime       | Short TTS + STT | Long TTS + STT |
| ----------------- | --------------- | -------------- |
| ONNX Runtime CPU  | 4.928 s         | 9.965 s        |
| OpenVINO CPU      | 4.867 s         | 9.645 s        |
| OpenVINO iGPU (*) | 6.453 s         | 14.296 s       |

Resampling cost about 3 ms for the short utterance and 8 ms for the long CPU
utterance. Text-only decoding preserved the words and produced only marginal
speed differences (roughly 0–2%); it does not remove Whisper's encoder cost.
The chosen path is **resident OpenVINO CPU FP32 TTS, two threads, direct float32
audio handoff, and resident small INT8 STT with two threads**.

The maintained CLI confirmed short round-trip medians of 4.823 seconds with
OpenVINO and 4.835 seconds with ONNX Runtime: effectively close, rather than a
large runtime win. All 652 offline tests passed, including resampling/pitch,
invalid-audio and OpenVINO output-ownership checks. Ruff and Pyright passed on
the changed Python files. Live production checks covered both CPU backends,
silence, three production utterances, preparation and file CLIs.

Full JSON results, audio, runtime graphs, operator profiles and failure tracebacks
are under `.cache/hoast/diagnostics/kokoro-openvino/`. Reproduction scripts in
`/tmp/opencode/` are `benchmark_kokoro_openvino.py`,
`benchmark_speech_roundtrip.py`, `verify_openvino_speech.py` and
`profile_kokoro_ort.py`; invoke them with `PYTHONPATH=.` from the project root.
This stage used `python -m tools.benchmark_speech` for CPU round trips; that driver
is retired.

### Direct PyTorch FP16 and weight-only INT8 exports

A direct export from the upstream checkpoint with the latest stable export stack
runs successfully in GPU FP16 and preserves the tested round-trip words. The
published ONNX failure is isolated to its dynamic harmonic-mixer matrix layout.
INT8-stored weights with FP16 computation also work, but save little resident GPU
memory despite a much smaller file. See [kokoro-fp16.md](kokoro-fp16.md) for
the diagnosis, current versions, actual kernel profiles, memory measurements and
reproduction commands. Native OpenVINO XML artifacts are supported by the TTS
CPU CLI through `--model`, including their `input_ids`/`ref_s` input convention.
