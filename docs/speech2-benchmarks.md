# Native speech runtime qualification

## Environment and measurement contract

Development uses the existing project uv `.venv`, Python 3.14.7, on an
Intel i5-8500T (AVX2, no VNNI) and UHD630 integrated GPU. Runtime and artifact
preparation must not depend on alternate Python interpreters. Whisper/Kokoro
results below are historical comparison evidence. Their runtime, preparation
tools and alternate-interpreter frontend have been removed; the maintained
benchmark tool accepts only SenseVoice and Melo. Baseline commands and recorded
launchers describe the original experiments, not supported reproduction paths.

Hoast and its user-systemd restarter are stopped during development. Stopping
Docker alone is insufficient: `hoast.service` uses `Restart=always` and
force-recreates the container. Its configuration has not been changed or disabled.

`tools/benchmark_speech2.py` runs fresh processes serially, with one or two
compute cores via affinity, explicit native pool limits and one inference stream.
Each case includes its first call and three subsequent warm calls. Latencies
below are warm medians, include preprocessing and output production, and exclude
physical playback and model initialization. TTS durations can differ between
models; compare wall latency and real output-duration RTF separately.

RAM records include RSS/USS/PSS, process high-water RSS and 50-ms sampled
process-tree RSS. Tree sums can double-count shared pages; they are not unique
physical allocations. Integrated-GPU driver allocations may not all appear in
process RSS. Do not add independently measured model peaks to claim concurrent
resident application memory. Results retain input hashes, actual worker CPU masks,
service identity/state before and after, WAVs and complete diagnostic logs.

Thread selection is inclusive: select one thread if
`median_one <= 1.2 * median_two`; otherwise two. Require this across representative
cases rather than hiding a slow language behind an average.

## Historical Whisper/Kokoro baseline (clean stopped-service trials)

Result tags: `stable-whisper-t{1,2}` and `stable-kokoro-hybrid-t{1,2}` under
`.cache/speech-bench/results/speech2/`. Earlier `baseline-*` attempts were incomplete
or contaminated by systemd restarts and are excluded.

| Model / case                 | One thread (s) | Two threads (s) | One / two | Selection |
|------------------------------|----------------|-----------------|-----------|-----------|
| Whisper small, English 15 s  | 2.392          | 1.283           | 1.864     | Two       |
| Whisper small, Chinese 4.2 s | 0.924          | 0.539           | 1.714     | Two       |
| Kokoro hybrid, EN short      | 1.515          | 1.172           | 1.293     | Two       |
| Kokoro hybrid, EN long       | 2.981          | 2.602           | 1.146     | One       |
| Kokoro hybrid, ZH short      | 2.212          | 1.704           | 1.298     | Two       |
| Kokoro hybrid, ZH long       | 3.436          | 2.584           | 1.330     | Two       |
| Kokoro hybrid, mixed short   | 1.436          | 1.092           | 1.314     | Two       |
| Kokoro hybrid, mixed long    | 2.578          | 1.939           | 1.330     | Two       |

The bilingual Kokoro policy therefore remains two threads despite its long-English
case passing the one-thread threshold. Whisper two-thread sampled tree peak is
**1.243 GB**, Kokoro hybrid **2.573 GB** (decimal units). Maximum observed warm
worker RSS is approximately **1.217 GB** and **2.108 GB**, respectively. Kokoro's
Chinese frontend subprocess is included in tree RAM and end-to-end latency.
Its existing interpreter has a container-only symlink; the baseline-only launcher
`.cache/speech-bench/repro/speech2-kokoro-python.sh` resolves the unchanged installed
interpreter/packages on the host. This is **not** a new-runtime or export dependency.

## Low-precision candidates

Precision labels distinguish storage, arithmetic and activation quantization.
FP16 storage alone does not establish FP16 arithmetic, nor does compressed INT8
storage establish that compiled GPU kernels avoid a materialized float weight copy.

### SenseVoiceSmall

- [Sherpa-maintained UINT8 ONNX release](https://k2-fsa.github.io/sherpa/onnx/sense-voice/pretrained.html):
  2024-07-17 multilingual model, 239,233,841-byte ONNX, local SHA256
  `c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51`.
  Dynamic activation quantization plus UINT8 constant MatMul weights; 281
  `MatMulInteger` and 281 `DynamicQuantizeLinear` nodes. Other arithmetic remains
  float. The exporter explicitly uses UINT8 to avoid an observed signed-INT8
  correctness problem. No sherpa/FunASR runtime is needed to execute its graph.
- Waveform FBANK, LFR7/6 and CMVN remain native supporting algorithms. CMVN is
  ONNX metadata, not an operation inside the model. Import verifies all 25,055
  token strings/order and both signed CMVN vectors against official resources.
- `tools/prepare_stt2_community.py` rewrites only batch-one public input types,
  derives the length and appends argmax, preserving quantized neural equations.
  OpenVINO serialization after direct graph surgery failed; the retained successful
  approach rewrites the public contract at ONNX level before OpenVINO import.
- [Official-namespace GGUF](https://huggingface.co/FunAudioLLM/SenseVoiceSmall-GGUF)
  offers F16 and Q8 weights. Lower priority: the associated native runtime's CMVN
  omission and changing evaluation methodology need explanation before treating
  it as a parity-preserving replacement.
- [Community FP16 ONNX](https://huggingface.co/ruska1117/SenseVoiceSmall-onnx-fp16)
  exists, but its export/evaluation provenance is less complete than sherpa's.

SenseVoice model weights have upstream custom model-license terms; permissive
runtime/source licenses do not replace those terms.

### MeloTTS

- Official MyShell EN/ZH checkpoints do not include ready-made low-precision
  exports in their inspected file trees.
- [apinge/MeloTTS.cpp community exports](https://huggingface.co/apinge/MeloTTS.cpp)
  contain EN and Chinese/mixed TTS FP16-compressed XML/BIN (~84 MB each), static
  INT8 variants (~42 MB each), and real-conditioning INT8 BERTs (~96 / 154 MB).
  Unlike BERT-zeroing sherpa-oriented exports, these retain real EN/ZH BERT
  features in the 768-wide `ja_bert` input. The zero 1024-wide `bert` input is normal
  for these checkpoints and must retain its projection bias.
- FP16 exports contain `f16` constants converted to FP32 with decompression
  metadata. INT8 exports also quantize activations with FakeQuantize; they are not
  pure weight-only models. The community README and [Intel's OpenVINO article](https://blog.openvino.ai/blog-posts/optimizing-melotts-for-aipc-deployment-with-openvino-a-lightweight-tts-solution)
  acknowledge INT8 background noise/distortion. DeepFilterNet is a workaround,
  not evidence of unchanged quality; include its latency if used.
- The repository-native four-stage preparation retains BERT embeddings plus
  ten layers (third-from-last hidden state), explicit duration/acoustic noise,
  inverse flow and waveform decoder. Sparse phone/frame repeat-gather avoids the
  quadratic dense alignment matrix. All text/OOV resources are explicit and
  offline; no Melo API, alternate interpreter or NLTK downloads are runtime needs.

### GPU weight-only optimization

UHD630 lacks hardware integer tensor/dot acceleration, but packed INT8 weights
can still reduce memory traffic: load bytes, dequantize into registers, accumulate
in FP16/FP32 without a full expanded weight tensor. This is a separate experiment
from integer activation quantization. Stock backend node/kernel/storage evidence
and custom OpenCL experiments must establish whether conversion is truly fused.

Unified RAM does not remove synchronization. This driver reports coarse SVM,
not fine-grained/system SVM. Explicit ownership, mapping/queue synchronization
and crossover overhead remain part of end-to-end evaluation.

## Native SenseVoice CPU results

Matched EN **15.051 s** and ZH **4.204 s** WAVs, three measured warm repetitions,
whole transcription call. All three native pairs pass strict measurement/provenance
validation and select **two threads**; this is a performance policy, not a quality gate.

| Native candidate                  | EN, 1 thread | EN, 2 threads | EN one/two | ZH, 1 thread | ZH, 2 threads | ZH one/two |
| --------------------------------- | ------------ | ------------- | ---------- | ------------ | ------------- | ---------- |
| Official FP32 / OpenVINO           | 1.527 s      | 0.886 s       | 1.725      | 0.432 s      | 0.252 s       | 1.718      |
| Community dynamic UINT8 / OpenVINO | 2.110 s      | 1.249 s       | 1.690      | 0.640 s      | 0.389 s       | 1.646      |
| Community dynamic UINT8 / ORT      | 1.286 s      | 0.752 s       | 1.709      | 0.359 s      | 0.214 s       | 1.676      |

Against Whisper small's two-thread **1.283 s / 0.539 s**, native FP32 reduces
EN/ZH latency by **31.0% / 53.3%**. Community UINT8 through OpenVINO reduces it by
only **2.7% / 27.9%**, while ORT reduces it by **41.4% / 60.2%**. ORT is also
**15.0% / 14.9%** lower latency than native FP32 and **39.8% / 44.9%** lower than
the same community quantized equations through OpenVINO. Two-thread input-duration
RTFs, derived from the recorded WAV duration and warm median, are:

| Model                 | EN RTF | ZH RTF |
| --------------------- | ------ | ------ |
| Whisper small         | 0.085  | 0.128  |
| SenseVoice FP32 / OV   | 0.059  | 0.060  |
| SenseVoice UINT8 / OV  | 0.083  | 0.092  |
| SenseVoice UINT8 / ORT | 0.050  | 0.051  |

ORT's **0.757 GB** sampled tree peak is **39.1%** below Whisper, **70.9%** below
native FP32 and **83.9%** below the community OpenVINO trial. The OpenVINO UINT8
path is not the preferred CPU performance candidate: smaller stored weights do
not translate into lower measured runtime RAM or latency. These observations do
not establish a kernel-level explanation of its memory expansion.

**Quality is separate:** native official FP32 passed independent checkpoint,
selected intermediate, logit and CTC/text golden checks. Storage-FP16 SenseVoice
preserves frame IDs on the two noise/two matched speech cases but fails a selected
intermediate numerical gate and is **rejected**, not fully qualified. Seven
community original-ORT versus imported-OpenVINO collapsed transcripts match;
six have identical frame IDs and one English clip has four differing frame IDs
without changing text. That is import parity for a quantized model, not a
quantized-versus-original-FP32 quality approval. The subsequent nine-case
original-ORT/imported-ORT/direct-FP32 comparison passes exact import frame/text
parity in every case, but differs from FP32 on held-out English (**“code” versus
“gold”**) and one-second silence. ZH/JA/KO/Yue held-out transcripts match FP32.
Five-second silence and noise can hallucinate text in both models. Evidence is
`results/speech2/stt2-ort-heldout-v1/report.json`; matched features and IDs are
retained beside it. This is not broad WER/CER evidence. Community INT8 remains
an explicit speed/RAM trade-off, not the quality-safe default.

## Native Melo CPU results

Candidate: decoder-only FP16 **storage**, with BERT/encoder/flow retained in FP32
storage and **CPU f32 arithmetic hints for every stage**. Six representative
EN/ZH/mixed texts, three warm calls each. All cases exceed the inclusive 1.2
one/two ratio, selecting **two threads** with complete native provenance.

| Case        | Melo 1 thread | Melo 2 threads | One/two | Kokoro 2 threads | Melo latency change |
| ----------- | ------------- | -------------- | ------- | ---------------- | ------------------- |
| EN short    | 2.090 s       | 1.198 s        | 1.744   | 1.172 s          | 2.2% higher         |
| EN long     | 4.008 s       | 2.263 s        | 1.771   | 2.602 s          | 13.0% lower         |
| ZH short    | 2.495 s       | 1.437 s        | 1.736   | 1.704 s          | 15.7% lower         |
| ZH long     | 4.693 s       | 2.745 s        | 1.710   | 2.584 s          | 6.2% higher         |
| Mixed short | 1.781 s       | 1.030 s        | 1.729   | 1.092 s          | 5.7% lower          |
| Mixed long  | 3.453 s       | 2.070 s        | 1.668   | 1.939 s          | 6.8% higher         |

Inputs are exactly the same text strings as Kokoro, but Melo produces shorter
waveforms; a lower wall latency does not imply higher synthesis throughput or
identical speaking rate. RTF is call latency divided by actual output duration:

| Case        | Kokoro audio (s) | Melo audio (s) | Kokoro warm RTF | Melo warm RTF |
| ----------- | ---------------- | -------------- | --------------- | ------------- |
| EN short    | 3.050            | 2.627          | 0.384           | 0.456         |
| EN long     | 6.025            | 5.170          | 0.432           | 0.438         |
| ZH short    | 4.475            | 3.138          | 0.381           | 0.458         |
| ZH long     | 6.875            | 5.940          | 0.376           | 0.462         |
| Mixed short | 2.800            | 2.279          | 0.390           | 0.452         |
| Mixed long  | 5.200            | 4.338          | 0.373           | 0.477         |

Melo CPU RTF is higher on every case, despite lower wall latency on three of six.
Its sampled tree peak is **3.793 GB**, **47.4%** above Kokoro hybrid's **2.573 GB**.
The native path removes the Chinese cross-interpreter frontend bridge; that is
an architectural benefit, not evidence of a universal speed/RAM improvement.

Decoder-only storage-FP16 passed six independent real-text waveform comparisons
against the unadapted original Torch path, including mixed text: identical sample
counts, finite outputs and SNR **61.1–68.5 dB**. This is measured numerical parity
on those texts, not a broad perceptual-quality claim. BERT+decoder storage-FP16
is rejected by the full-waveform gate (about **28.4 dB**); full FP16 encoder also
fails its stage gate. No tolerance was relaxed to retain those candidates.
Reports: `melo-{en,zh}-fp16-decoder-real-parity.json` in the results directory.
Hybrid GPU-decoder `f16` arithmetic fails the six-text SNR ≥40 dB gate:
**27.5–34.4 dB**, despite exact sample geometry and finite outputs. Durable report:
`melo-hybrid-f16-real-parity-v2.json`. These are profiled first-shape calls,
including dynamic device work, **not** an uninstrumented warm latency comparison.
No GPU speed or zero-copy claim is made, and this path is not selected.

## Memory and bottlenecks

All RAM tables use **decimal GB = bytes / 1,000,000,000**; parenthetical GiB uses
**bytes / 1,073,741,824**. Tree peak covers the whole trial, including compilation
and first calls, at 50-ms sampling. Worker high-water RSS is Linux `ru_maxrss`
converted from KiB; warm RSS is the largest observed per-call worker RSS. None
is a concurrent application-residency measurement.

| Model / backend          | 1-thread tree GB (GiB) | 2-thread tree GB (GiB) | 2-thread worker high-water GB | 2-thread max warm worker GB |
| ------------------------ | --------------------- | --------------------- | ---------------------------- | -------------------------- |
| Whisper small            | 1.253 (1.167)         | 1.243 (1.157)         | 1.226                        | 1.217                      |
| Kokoro hybrid            | 2.534 (2.360)         | 2.573 (2.396)         | 2.108                        | 2.108                      |
| SenseVoice FP32 / OV      | 2.593 (2.415)         | 2.601 (2.422)         | 2.563                        | 2.564                      |
| SenseVoice UINT8 / OV     | 4.669 (4.349)         | 4.692 (4.369)         | 4.654                        | 4.655                      |
| SenseVoice UINT8 / ORT    | 0.757 (0.705)         | 0.757 (0.705)         | 0.757                        | 0.757                      |
| Melo decoder-FP16 CPU     | 3.971 (3.698)         | 3.793 (3.533)         | 3.756                        | 3.756                      |

Native coarse stage timers are present even though backend profiling is disabled.
SenseVoice uses flat `stage_profile` fields; inference is **99.1–99.6%** of warm
wall latency across the three backends/two cases. Frontend medians are approximately
**4.9–7.6 ms EN / 1.5–1.6 ms ZH**; validation, resampling and CTC/text decoding
are much smaller. Neural inference, rather than the native FBANK/CTC path, is the
major latency bottleneck.

Melo timings are under `stage_profile.segments`: sum each stage's seconds across
segments **per call**, then take the warm median. Token/phone/frame counts and
`stage_execution` metadata are not durations. Independent stage medians need
not sum to the end-to-end median. Two-thread stage medians:

| Case        | Frontend (ms) | BERT (ms) | Encoder (ms) | Expand (ms) | Flow (s) | Decoder (s) | Decoder share of wall |
| ----------- | ------------- | --------- | ------------ | ----------- | -------- | ----------- | --------------------- |
| EN short    | 0.397         | 21.872    | 11.582       | 0.850       | 0.096    | 1.069       | 89.3%                 |
| EN long     | 0.558         | 27.267    | 20.144       | 1.587       | 0.200    | 2.014       | 88.9%                 |
| ZH short    | 2.501         | 24.779    | 13.643       | 0.976       | 0.115    | 1.275       | 88.7%                 |
| ZH long     | 4.203         | 47.263    | 28.100       | 1.924       | 0.232    | 2.423       | 88.3%                 |
| Mixed short | 1.666         | 19.085    | 9.351        | 0.748       | 0.080    | 0.909       | 89.1%                 |
| Mixed long  | 3.291         | 36.475    | 25.220       | 1.477       | 0.166    | 1.818       | 88.1%                 |

The waveform decoder dominates (**88–89%**); inverse flow is next (**7.8–8.7%**),
with BERT around **1.2–1.9%**. The warm frontend is not the main CPU bottleneck.
Baseline JSONs do not contain stage timings; no baseline stage comparison is inferred.

## Reproducing the evidence summary

Immutable inputs are the `stable-whisper-t{1,2}`, `stable-kokoro-hybrid-t{1,2}`,
`native-sensevoice-{fp32,int8,ort-int8}-t{1,2}` and
`native-melo-fp16-decoder-cpu-t{1,2}` JSONs under
`.cache/speech-bench/results/speech2/`. The read-only summarizer is retained at
`.cache/speech-bench/repro/summarize-native-speech-v2.py`, with its scratch source
at `/tmp/ttc/1000/summarize_native_speech_v2.py`. From the project root:

```sh
.venv/bin/python .cache/speech-bench/repro/summarize-native-speech-v2.py --root "$PWD"
```

It calls `hoast.speech_policy.qualify_threads` with exact two-STT/six-TTS inventories
and **strict complete provenance for every native pair**. Baselines use the policy's
historical-evidence mode: clean stopped-service results remain comparable, but their
missing artifact/group provenance is explicitly retained, not silently invented.
It checks matched WAV identities/text, records source SHA256s, warm samples/medians,
ratios, selected threads, output-duration RTF and stage/memory semantics. It performs
**no inference, builds or tests**. Output is
`.cache/speech-bench/results/speech2/native-speech-summary-v2.json`; exclusive creation
refuses to overwrite it. To independently regenerate after the first run, supply
`--output /tmp/native-speech-summary-v2-check.json` with a fresh path.
Constructor timing is not disk-cold startup: artifact hashing warms filesystem caches,
and lazy language compilation remains in first calls rather than warm medians.

## Qualification status

Native STT/frontend/graph/runtime fixtures: **88 passed** in the consolidated
run. Native STT FP32 passed independent checkpoint, selected intermediate-block,
logit and CTC/text comparisons; its NumPy feature path also passed independent
`kaldi-native-fbank` checks on silence, short boundary, noise and matched EN/ZH.
Reports are `results/speech2/stt2-fp32-reference-parity.json` and
`stt2-fbank-parity.json` under the benchmark cache. Melo's distributed English
GRU matches twelve synthetic/real-resource predictions; nineteen official
EN/ZH/mixed texts pass frontend geometry and type invariants.

Packed-INT8 OpenCL: **26 device tests passed**. Eight matmul and twenty-seven
stride-one dilated-convolution synthetic shapes completed. Weights remain signed
bytes in global/local tiles and convert into registers, without an application
expanded-weight buffer. However, measured full wall time loses to the CPU FP32
reference on every tested shape: GPU/CPU **1.73–3.64×** for matmul and
**1.56–18.81×** for convolution. These kernels remain experiments, not runtime
replacements. The CPU reference uses the same two-core affinity and one-thread
OpenBLAS policy; CPU dequantization is excluded, and synthetic accuracy is not
real-model quality. Raw results: `results/speech2-gpu/int8-{matmul,conv}-first.json`.

The postscale variant applies each output-channel scale after the raw-byte dot,
reducing per-weight arithmetic but changing rounding/overflow behavior. **47
GPU/offline tests passed**, including both scale placements and copy/map paths.
Four matmul and twelve convolution synthetic shapes pass device arithmetic gates;
full-wall GPU/CPU FP32 ratios are **2.19–5.88×** and **1.45–13.49×**, respectively.
Raw results: `results/speech2-gpu/postscale-{matmul,conv}-v1.json`. No tested
full-path win justifies wiring these experimental kernels into either model.

Melo original-source FP32 neural/waveform parity and decoder-only storage-FP16
six-text waveform checks passed; bounded CPU timing/RAM evidence is complete.
Four native Melo-to-community-ORT round trips (EN, ZH and mixed) pass normalized
transcript checks with both engines in Python 3.14. Evidence:
`.cache/hoast/diagnostics/speech2-check/20261005T012331868591Z/roundtrip.json`.
This is a smoke test, not a human listening panel or general ASR accuracy score.
The community candidate changes held-out English content and remains opt-in;
GPU `f16` decoder arithmetic fails the waveform gate. Broader perceptual/ASR
coverage and application deployment are not established. No baseline-only or
prior six-thread Melo measurements establish success under this task's budget.

## Selected local defaults and final checks

The published local defaults are official SenseVoice **FP32 OpenVINO CPU** at
`models/stt2` and Melo **decoder-only FP16 storage / CPU f32 arithmetic** at
`models/tts2`, with **two threads**. Unchanged qualified graph bytes are
copied from the measured candidate roots; STT additionally bundles verbatim legal
texts/provenance. `models/speech2-policy.json` records benchmark result hashes,
ratios and published manifest hashes; it is not an automatic runtime policy loader.
FP32-STT/Melo native round trips also pass all four fixtures, with evidence under
`.cache/hoast/diagnostics/speech2-check/20261005T012754639976Z/`.

Final repository suite: **1,363 passed, 24 skipped**, with six existing Torch
quantization deprecation warnings. Opt-in GPU execution separately passes all
47 GPU/offline tests. Repository Pyright reports zero errors/warnings; Ruff lint,
touched-file formatting and `git diff --check` pass. Full-repository formatting
still reports 23 unrelated pre-existing files, intentionally left unchanged.
Pinned-input offline validation and 26 mocked bootstrap/fetch tests pass.
The complete online/bootstrap orchestration was not run as a single command;
individual native preparation/qualification/inference stages were run.

**Hoast service/container remain stopped.** No combined LLM/speech deployment,
live-satellite test or broad human-reference/perceptual qualification is claimed.
