# Native SenseVoiceSmall STT

`hoast.stt2` uses a deterministic NumPy/SciPy frontend, the checkpoint's actual
native SentencePiece decoder, and an explicit native graph backend: OpenVINO
float32 SANM or the community dynamic-UINT8 ONNX graph on ONNX Runtime CPU. Runtime
does not import Torch or FunASR, download assets, launch subprocesses, or invoke
another Python interpreter.

## Preparation

Run preparation with the project's existing Python 3.14 environment. It only
reads a local source snapshot; there is no implicit download. Defaults pin
`FunAudioLLM/SenseVoiceSmall` commit
`3847d57b6bdf2dd8875cb1508d2af43d80a16bf7` and model.pt SHA256
`833ca2dcfdf8ec91bd4f31cfac36d6124e0c459074d5e909aec9cabe6204a3ea`.

```sh
.venv/bin/python -m tools.fetch_speech2 --download
.venv/bin/python -m tools.prepare_stt2 \
    --source models/speech2-inputs/sensevoice \
    --license-dir models/speech2-inputs/sensevoice/licenses \
    --output models/stt2 --threads 2
```

Native Torch performs a `weights_only=True`, memory-mapped checkpoint load during
preparation. All tensor keys, dimensions and float32 dtypes are validated before
graph construction. OpenVINO serialization retains float32 weights. A sibling
staging directory is atomically renamed only after all required files and their
JSON integrity manifest are complete; preparation refuses an existing output.
Use a different destination for a candidate or update. `--output-logits` adds a
full-logits output for numerical qualification, not normal runtime use.

Preparation does not run model inference. Numerical parity, latency and resource
qualification must be performed separately before deployment. Override both
`--revision` and `--checkpoint-sha256` explicitly to qualify another checkpoint.
The manifest records every source and output file's SHA256 and byte size.

### Community dynamic-UINT8 ONNX Runtime candidate

The community importer verifies the audited 2024-07-17 graph's SHA256
`c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51`, complete token
ordering and CMVN against the official local resources. It rewrites only the
ONNX public contract to `speech/language/textnorm -> frame_ids`: frame length is
derived from speech, prefix IDs cast to the source INT32 arithmetic, and ArgMax
emits int64 frame IDs including four metadata frames. Quantized operators and
weights are retained. No sherpa or FunASR runtime/bridge is involved.

```sh
.venv/bin/python -m tools.fetch_speech2 --download --with-community
.venv/bin/python -m tools.prepare_stt2_community \
    --community models/speech2-inputs/community \
    --official models/speech2-inputs/sensevoice \
    --license-dir models/speech2-inputs/sensevoice/licenses \
    --output models/stt2-community-ort --threads 2 --backend onnxruntime
```

Paths must refer to the extracted community archive and official snapshot.
`--backend onnxruntime` retains transformed `model.onnx` without importing or
converting through OpenVINO; `--backend openvino` (default) creates `model.xml`
and `model.bin`. Both outputs require CMVN, tokenizer and config plus a hashed
manifest, and refuse an existing output directory. The format identifier stays
`hoast-sensevoice-openvino-v1`; explicit manifest `backend: onnxruntime` selects
the ONNX file set. Missing backend selects the OpenVINO file set. Runtime rejects
an artifact/backend mismatch before changing process affinity.

```python
with STT(
    STTConfig(
        model_path=Path("models/stt2-community-ort"), threads=2, backend="ort"
    )
) as recognizer:
    text = recognizer.transcribe(Path("command.wav"))
```

ORT explicitly uses only `CPUExecutionProvider`, sequential execution, one/two
intra-op threads, one inter-op thread and disabled intra/inter-op spinning. It
owns one serialized session and returns copied IDs. Process affinity and native
pool budgeting are identical to the OpenVINO path. There is no runtime/device
fallback. This is an **unqualified candidate**, not a deployment default or a
claim of lower RAM/latency: graph loading, provider kernels, transcript parity,
and resource performance must be qualified with real utterances separately.

## Recognition

```python
from pathlib import Path
from hoast.stt2 import STT, STTConfig

with STT(STTConfig(threads=2, language=None, backend="cpu")) as recognizer:
    text = recognizer.transcribe(Path("command.wav"))
    raw = recognizer.last_raw_text  # Includes language/emotion/event/ITN metadata.
    stages = recognizer.last_timings  # Seconds, including frontend and decoding.
```

`transcribe_samples(samples, sample_rate)` accepts finite mono float32 arrays and
resamples to 16 kHz using SciPy's polyphase resampler. File decoding uses
libsndfile and averages channels. Language controls are `None`/`auto`, `zh`, `en`,
`yue`, `ja`, or `ko`; unsupported controls raise instead of selecting auto. The
reference default is non-ITN embedding 15; `use_itn=True` selects embedding 14.

`STTConfig.max_audio_seconds` defaults to **60.0 seconds** and must be finite
and positive. Source duration `frames / sample_rate` is accepted up to and
including the limit. Array inputs exceeding it raise before the finite-value
scan, resampling, frontend or inference. File inputs are checked using
`soundfile.info` frame/rate metadata before decoded audio is allocated, then
checked again after decoding. Inputs are never truncated. Set a larger explicit
limit for deliberate long-utterance diagnostics; attention memory/time remain
quadratic, and the duration guard is not a fixed RAM cap.

The engine owns one synchronous request and one inference stream. CPU compilation
requests latency mode, FP32 inference and exactly one or two inference threads.
`backend="GPU"` explicitly selects an experimental GPU candidate with the same
FP32 graph, one stream and latency hint. Unsupported graph/device compilation
raises without selecting CPU. GPU speed, numerical parity and shared-RAM cost
must be qualified before deployment; the default remains `backend="cpu"`.

After offline artifact validation, API construction applies the shared process
CPU budget before lazy plugin import/loading: one/two allowed CPU cores,
OpenBLAS one thread, and bounded OpenMP/MKL workers. GPU driver/host workers inherit
that CPU-affinity budget. The application's measured thread policy chooses one
or two; the API's configurable default is two. This is a **process-wide** policy,
not a separate core reservation for each recognizer. Close does not restore or
enlarge the process CPU mask. Calls and close serialize with a reentrant lock.
Close releases the inference request, compiled graph, native tokenizer and
frontend storage; text, timing and copied profile snapshots remain available.

Stage timings in `last_timings` include input validation, resampling, frontend,
synchronous inference/copy and decoding. File calls also record `file_decode`
and include file I/O/downmixing in `total`. Lock-queue time is excluded. Detailed
OpenVINO counters are optional:

```python
with STT(STTConfig(threads=1, backend="cpu", profiling=True)) as recognizer:
    text = recognizer.transcribe(Path("command.wav"))
    counters = recognizer.last_profile  # JSON-safe node records; durations in seconds.
```

`profiling=False` avoids enabling/collecting node counters. When enabled,
`last_timings["profiling"]` separates counter-collection overhead. Node names,
operation/kernel types, statuses, plugin wall/CPU seconds are retained without
holding native profiling objects. Fused nodes can represent several source
operators, and overlapping node times must not be summed as wall latency.
Instrumented runs should be measured separately from latency qualification.

For ORT, `profiling=True` instruments the entire session, but ORT flushes its
JSON trace only on `close()`. `last_profile` is empty while open and contains
all session node events after close; `profile_path` then identifies the durable
trace under `.cache/hoast/diagnostics`, outside model artifacts. Unique prefixes
avoid profile collisions. ORT microsecond wall durations are converted to
seconds; `cpu_seconds=0.0` means unavailable, not measured zero CPU work.
Native traces preserve the full session/request/thread metadata. No profile
files or native node instrumentation are requested with `profiling=False`.
Keep instrumented trials separate from headline measurements.

## Numerical contract and limitations

- Normalized samples are multiplied by 32768. Frames contain 400 samples with a
  160-sample hop; incomplete trailing frames are dropped (`snip_edges=True`).
- The frontend removes each frame's DC, applies preemphasis 0.97 with replicated
  first sample, uses a nonperiodic Hamming window, and computes 512-point FFT
  power. Eighty Kaldi mel triangles span 20–8000 Hz; Nyquist weights are zero.
  Log energies are floored at float32 epsilon.
- LFR uses seven frames every six frames, selecting `clip(6*j+k-3)` at the
  boundaries. Signed CMVN computes `(LFR + AddShift) * Rescale`.
- Dither defaults to **zero**, unlike the reference WavFrontend default of one
  int16-scale unit. Disable reference dither explicitly for deterministic parity.
  Equivalent float32 implementations need not be bit-identical across FFT and
  math libraries; qualification should compare tolerances and CTC content.
- Inputs with fewer than 400 resampled samples (25 ms) raise `ValueError`; window
  length is not silently reduced. Empty/nonfinite/incorrect-rank input is rejected.
- The four prefix embeddings are language, event 1, emotion 2, and ITN control.
  Input is scaled by sqrt(512); 560-wide sinusoidal positions start at one.
  Seventy SANM blocks execute as first 560-to-512, 49 remaining main blocks, and
  20 post-normalization blocks. Each uses four heads, FFN2048 and depthwise11.
- Per-frame CTC argmax includes metadata. Adjacent IDs collapse **before** blank
  zero is removed; actual SentencePiece decoding preserves its whitespace,
  unknown/byte-fallback and special-piece semantics. Canonical text strips model
  metadata and uses the application's input-text normalization; raw decode is
  retained unchanged, even if canonical numeric validation raises.
- There is no VAD, silence heuristic, chunking, fixed-duration padding or audio
  truncation. Metadata-only decoding returns empty text. Attention memory/time
  scale quadratically with input length; max_audio_seconds rejects over-limit
  input (60 seconds by default) rather than changing the audio.
- Offline initialization verifies all artifacts with streaming SHA256 and then
  validates graph input/output contracts. Hashing is startup work and uses a
  bounded buffer; it does not happen on transcription calls.

Source attribution and MIT/Apache-2.0 terms are in [LICENSES.md](LICENSES.md).
