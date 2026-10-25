# Native Python 3.14 speech

This is the direct native SenseVoiceSmall/MeloTTS path. The measured local
defaults are **SenseVoice FP32 CPU** and **Melo decoder-only FP16 storage with
CPU f32 arithmetic**, using **two threads**. Deployment has not been performed.
Use the existing project `.venv` and prepared local artifacts. Runtime and preparation require Python
3.14; neither uses an older-Python speech worker or subprocess model bridge.
For measured comparisons and their scope, see [speech2-benchmarks.md](speech2-benchmarks.md).

## Architecture

- **STT (`hoast.stt2`):** mono audio is resampled to 16 kHz, then repository-owned
  NumPy FBANK, LFR7/6 and signed CMVN produce FP32 `[1,T,560]` features. A resident
  SenseVoice graph returns frame IDs. Adjacent-repeat collapse, blank removal,
  native SentencePiece decoding and metadata stripping produce canonical text.
  The official checkpoint is loaded only during preparation. Runtime needs no
  FunASR or sherpa wrapper, Torch checkpoint loader, downloader or bridge.
- **TTS (`hoast.tts2`):** native EN/ZH normalization, dictionaries, POS tagging,
  NumPy English OOV GRU, Mandarin tone sandhi and local fast tokenizers produce
  phones, tones, languages and real BERT conditioning. Four resident OpenVINO
  stages run BERT → encoder/durations → inverse flow → waveform decoder. BERT
  keeps embeddings and ten of twelve layers, exactly `hidden_states[-3]`.
  The unused 1024-channel zero-BERT projection retains its trained bias.
- Duration noise and acoustic noise are explicit. Phone/frame repeat-gather
  replaces dense alignment; relative attention remains full quadratic attention,
  with a four-position relative embedding window. Batch is one, with no padding.
  Decoder output is mono FP32 at 44,100 Hz, with 512 samples per latent frame.
- EN/ZH graphs and tokenizers are separate. English dictionary/tagger/GRU/OOV
  resources are shared within one TTS instance **only when all manifest records
  for `cmudict.rep` and every `g2p/` file match**. The inference lock serializes
  their mutable cache. This is not unconditional sharing or a measured RAM claim.

## Offline assets and pins

Obtain missing snapshots/resources separately. The commands below do not fetch
assets. Missing, corrupt or incompatible inputs fail rather than selecting a
replacement. Source-tree caches are not redistributable solely because they are
locally available.

| Component                        | Repository / source              | Pinned revision                          |
| -------------------------------- | -------------------------------- | ---------------------------------------- |
| SenseVoiceSmall                   | `FunAudioLLM/SenseVoiceSmall`     | `3847d57b6bdf2dd8875cb1508d2af43d80a16bf7` |
| Melo neural/text source           | `myshell-ai/MeloTTS` (GitHub)     | `209145371cff8fc3bd60d7be902ea69cbdb7965a` |
| Melo English checkpoint           | `myshell-ai/MeloTTS-English`      | `bb4fb7346d566d277ba8c8c7dbfdf6786139b8ef` |
| Melo Chinese checkpoint           | `myshell-ai/MeloTTS-Chinese`      | `af5d207a364ea4208c6f589c89f57f88414bdd16` |
| English BERT                      | `bert-base-uncased`              | `86b5e0934494bd15c9632b12f734a8a67f723594` |
| Chinese/mixed BERT                | `bert-base-multilingual-uncased` | `7cbf9a625e29989f6b9c6c2fa68234c304f7e38f` |

Checkpoint SHA256:

- SenseVoice `model.pt`: `833ca2dcfdf8ec91bd4f31cfac36d6124e0c459074d5e909aec9cabe6204a3ea`.
- Melo EN `checkpoint.pth`: `acd278040eaf9536908e2b965273df5a731c44d8f0da66cc5fed7972772ed23c`.
- Melo ZH `checkpoint.pth`: `a74e9eadffff065c75eb6dfa040efa72cad23e72cfea70d39190bc174fb97093`.
- Sherpa-maintained SenseVoice 2024-07-17 community `model.int8.onnx`:
  `c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51`.

Melo's `models/speech2-inputs/inputs.melo-assets.json` inventories exact official
checkpoint/config/BERT/tokenizer bytes. Preparation checks pinned source revision
and helper hashes, strict-loads the original neural checkpoint, and verifies BERT
trunk loading. It executes only audited preparation-only neural helpers, not the
upstream Melo API/frontend/Numba/alignment runtime. Frontend preparation copies
static resources and converts a restricted primitive-only tagger pickle to JSON.
English G2P resources are static data under `models/speech2-inputs/g2p_en`, not
an upstream Python environment to execute/import.

Artifact manifests include file sizes and SHA256. Melo inventories nested assets
with relative POSIX names; runtime validates containment and hashes before graph
loading. STT verifies backend-specific artifacts, configuration and tokenizer
contracts. Publication uses unpublished staging and atomic rename, refusing
existing targets. Hashes establish integrity/provenance, not model quality.

## Prepare from the project root

These are real conversion commands, not lightweight checks. Run them serially
with `--threads 1` or `2`; choose a fresh output path if the illustrated one exists.

```sh
.venv/bin/python -m tools.fetch_speech2 --download
.venv/bin/python -m tools.prepare_stt2 --source models/speech2-inputs/sensevoice --license-dir models/speech2-inputs/sensevoice/licenses --output models/stt2-fp32 --threads 1 --precision FP32

for language in EN ZH; do
  .venv/bin/python -m tools.prepare_tts2 --source models/speech2-inputs/MeloTTS --assets-manifest models/speech2-inputs/inputs.melo-assets.json --output models/tts2-fp32 --language "$language" --threads 1 --precision FP32 --verify-original --g2p-resource-dir models/speech2-inputs/g2p_en --nltk-resource-dir models/speech2-inputs/nltk
done
```

For the **decoder-only FP16 candidate**, use the same Melo loop with
`--output models/tts2-fp16-decoder --precision FP16 --fp16-stages decoder`.
Root precision FP16 can mean mixed storage: default `--fp16-stages bert decoder`
leaves encoder/flow FP32. Explicit opt-ins can select other stages, but must pass
all gates. `--verify-only` instead writes standalone original-source parity JSON,
without conversion/publication; the output root is independent of model roots.

Experimental Melo `--precision INT8 --int8-stages bert` uses native NNCF symmetric
**weight-only** compression; BERT is the default opt-in and other stages stay
FP32. Embeddings/relative tables, duration/prior/coupling heads and sensitive
decoder boundaries, conditioning and deconvolution weights are protected.
It is not activation quantization or a guarantee of hardware integer execution.

For the separate community STT ONNX Runtime candidate:

```sh
.venv/bin/python -m tools.fetch_speech2 --download --with-community
.venv/bin/python -m tools.prepare_stt2_community --community models/speech2-inputs/community --official models/speech2-inputs/sensevoice --license-dir models/speech2-inputs/sensevoice/licenses --output models/stt2-community-ort --threads 1 --backend onnxruntime
```

This verifies community graph hash, CMVN and complete token ordering against
local official assets. It retains dynamic activation quantization/UINT8 MatMul
weights in transformed `model.onnx`; it does not invoke sherpa. Use
`--backend openvino` and a distinct output root for the IR comparison candidate.
Neither community import nor native STT preparation alone establishes accuracy;
independent frontend/checkpoint/CTC/text qualification is separate.

## Explicit execution policy

CPU budgets are process-wide affinity/native-pool policies, not independently
reserved pools. Pass **one or two threads** explicitly during comparisons.

- STT `cpu` / `GPU` select OpenVINO CPU / GPU with one request/stream and FP32
  arithmetic hint. GPU is an explicit candidate, never an automatic fallback.
- STT `ort` selects ONNX Runtime **CPUExecutionProvider only**, sequential execution,
  one inter-op thread and the chosen intra-op budget, without fallback.
- TTS Python `TTSConfig.backend` accepts `cpu`, `hybrid` (GPU decoder only), or
  `GPU` (all stages). CPU requests FP32 arithmetic. GPU uses the explicit
  `gpu_precision` hint (`f16` or `f32`); it requires independent qualification.
  Preparation's per-stage GPU-candidate hints are metadata, **not automatically
  enforced by the runtime's global GPU hint**. Preserve sensitive stages through
  deliberate device/hint selection when evaluating mixed artifacts.
- FP16/INT8 storage does not establish arithmetic precision, packed device
  residency, fused decompression, reduced memory, speed or perceptual quality.
  Preparation retains FP32 graph/I/O and qualifies on CPU with `f32`.

Application model roots are independent of the LLM `--device` setting:

```sh
.venv/bin/python -m hoast --config config.toml --stt-backend cpu --stt-model-dir models/stt2-fp32 --tts-model-dir models/tts2-fp32 --threads 2
# ORT candidate: use --stt-backend ort --stt-model-dir models/stt2-community-ort
```

Run only after preparing/qualifying those roots and the application's other
requirements. The ORT root is a separate candidate, not an implicit alternative
for a missing OpenVINO graph. Profiling is optional and separate from uninstrumented
comparisons; ORT finalizes its durable trace only at close.

## Numerical qualification and evidence

Melo compares conversion to FP32 Torch first, then serialized candidate stages
at lengths 4/5/9. Stage `atol=rtol` budgets are FP32 `3e-4`, FP16 `3e-3`, INT8
`2e-2`, selected by each stage's actual storage precision. Original-source parity
captures unadapted neural methods, full-BERT hidden states, injected identical
noise, spline tails/bounds and original dense-alignment/full-waveform inference.

Composition requires exact integer durations, native waveform geometry and
finite values, plus the root-policy elementwise waveform budget. SNR floors are
**60/40/30 dB for FP32/FP16/INT8**. Only SNR can be bypassed when reference RMS is
**strictly below `1e-4`** and maximum error is **at most `1/32768`** (one PCM16 LSB).
Allclose, geometry, finite-value and duration checks still apply. Boundary RMS
`1e-4` is not quiet; boundary one-LSB error qualifies. No stage budget is relaxed.

Before waveform rejection, NPZ arrays and JSON metrics/status/reasons/seeds are
saved under `.cache/hoast/diagnostics/tts2-qualification/RUN/`; the unique absolute
path is logged and successful manifests link it. These survive staging deletion.
Duration evidence is saved before exact-duration assertion. Preparation tracebacks
are in `.cache/hoast/diagnostics/prepare-{stt2,stt2-community,tts2}.log`.
Synthetic numerical gates do not replace real-text or perceptual evaluation.

Qualification snapshot for this development session: native STT FP32 is independently
qualified at `models/stt2-fp32`; Melo FP32 EN/ZH at `models/tts2-fp32`
passed six representative EN/ZH/mixed real texts. Decoder-only FP16 EN/ZH passes
six-text waveform parity at 61.1–68.5 dB SNR. Full-FP16 and BERT-plus-decoder
FP16 candidates fail numerical gates; GPU-decoder f16 arithmetic also fails
six-text SNR. Community UINT8 ORT passes exact source/import contract parity,
but changes one held-out English word versus FP32 (“gold” → “code”); it remains
an explicit faster/lower-RAM alternative rather than the quality-safe default.

Local default roots `models/stt2` and `models/tts2` contain unchanged
qualified graphs copied from the measured candidates (with an additional STT legal
bundle). Strict one/two comparisons select two on every representative case;
`models/speech2-policy.json` records source result hashes, medians, ratios and
published manifest hashes. This is an evidence record, **not an automatic runtime
policy loader**. Native four-case EN/ZH/mixed round-trip checks pass with both
FP32 and community-ORT STT. These are scoped checks, not broad WER/CER,
perceptual or deployment approval. Preserve rejected-run logs under
`.cache/speech-bench/logs/speech2/`.

## Playback, cancellation and bounds

Synthesis returns native 44.1-kHz audio. Playback resamples once to the shared
24-kHz bounded ordered queue, including raw debug audio. `wait_playback()` drains;
`close()` drains accepted audio and releases resident graphs. Cancellation skips
new submissions before synthesis/at queue boundaries; **accepted audio drains**.
It does not promise immediate interruption of synchronous model inference.

The shared playback worker requests a 20-ms device period and a 60-ms output
buffer, independently of the producer queue capacity. At close, it writes silence
covering the reported output latency plus one period before an explicit checked
stop. This pushes accepted speech through the ALSA/PipeWire adapter, whose stop
can otherwise discard the final buffered samples. Empty/cancelled-before-submit
queues never open a device. Device-write, flush and stop failures propagate to
the owner. This timing policy is qualified on the current host; sufficiently long
producer/scheduling stalls can still underrun, and other devices are not qualified.

### PCM scale versus playback loudness

- Melo's decoder returns normalized float32 PCM directly from its trained `tanh`
  output. Native synthesis, WAV output and ordinary playback add no normalization
  or gain. The application's initialization announcement has the separate,
  bounded loudness policy below.
  `resample_poly` preserves amplitude; it does not multiply by a sample-rate ratio.
  PCM16 WAV writing performs ordinary libsndfile quantization, not an extra manual
  division by 32,768. Raw microphone debug replay also preserves its captured level.
- Satellite PCM16 reaches both application STT and Silero as `float32(pcm)/32768`.
  Silero uses those normalized samples. SenseVoice FBANK intentionally multiplies
  normalized samples by 32,768 once, because Kaldi expects PCM16-scale values.
  Signed CMVN acts on log-power features, not waveform amplitude. File decoding
  already returns normalized floats and must not be divided by 32,768 again.
- Scale regression fixtures cover PCM extrema/one-LSB values, normalized VAD
  context, nonzero PCM16 file/sample feature equality, stereo downmix, resampling
  amplitude and TTS PCM16 writing. These do not measure live microphone gain,
  satellite AGC, analog amplification or acoustic loudness.

Host playback evidence is retained under
`.cache/hoast/diagnostics/playback-{buffer,tail,fixed}-20261005T02*/` with complete
logs, WAV sink-monitor captures and timing/alignment JSON. The existing short
Chinese Melo clip has peak 0.2286 (−12.82 dBFS), RMS 0.03118 (−30.12 dBFS).
Resampling changes RMS by only +0.0037 dB. The fixed shared queue played the
3.138-second clip in 3.171 seconds with no logged underruns, aligned sink gain
0.99966 and waveform SNR 62.64 dB. A silence-trimmed tail test demonstrates full
final-sample retention with the flush. These are digital-path checks, not proof
of acoustic volume. At inspection, the host's 40% sink setting reported −23.88 dB
hardware attenuation, which can make a correctly scaled clip hard to hear.

An intentional loudness policy is separate from a numerical scale correction.
For this clip, +25 dB means ×17.78 and puts about 8.2% of source samples above
full scale; it is not a safe unconditional gain. Device buffering/draining does
not apply gain or a limiter, or change system volume/routing.

The application levels **only** `系统初始化完毕` during startup: whole-clip RMS
target 0.07 (−23.1 dBFS), at most 4× gain (+12.0 dB), and source-rate peak at most
0.8, leaving interpolation headroom. It applies one scalar without clipping or
compression, and prepends 200 ms of silence before blocking playback/drain.
Startup statistics and the actual gain are logged; normal replies and warm-up
recognition clips remain unchanged. Silence is not amplified. This is an explicit
audibility policy, not a decoder scale correction or a perceptual loudness meter.

Exact-phrase measurements in
`.cache/hoast/diagnostics/audibility-20261005T0241Z/` compare three fixed seeds:
initialization RMS −32.8 to −32.4 dBFS versus “Light on/off.” −24.3 to −21.7 dBFS.
The startup policy therefore adds approximately 9.3–9.7 dB on those fixtures,
without changing ordinary speech. Capture feedback tones use 0.4 peak (+10.46 dB
relative to 0.12); debug replay bracket tones retain 0.12. Digital level checks
do not establish acoustic audibility or qualify echo behavior. System sink volume
was observed at 80% during this follow-up and is not changed by Hoast.

Default TTS limits reject rather than silently truncate: 512 BERT tokens and 2,048
phone positions per segment; 4,000 latent frames per segment; 12,000 frames and
64 segments per utterance. Speed is bounded to `[0.5,2]`. Application reply text is
bounded to 400 characters at speech boundaries. STT handles complete utterances
without internal VAD/truncation, rejects inputs shorter than a 25-ms feature frame,
and has quadratic attention. `STTConfig.max_audio_seconds` defaults to an inclusive
60-second source-duration limit. Array requests fail before resampling/frontend;
file requests preflight `sf.info` before decoding, then recheck decoded duration.
The duration limit is not a general bound on RAM or channel count.

## Licenses and integration boundaries

Retain Melo/MyShell MIT and Tacotron MIT attribution, PaddleSpeech tone-sandhi
and g2p_en Apache-2.0 notices, and version-specific CMU dictionary notices.
Prepared frontend assets include notices, license text and provenance; see
[frontend notices](../hoast/tts2/frontend/NOTICE.md) and
[asset contract](../hoast/tts2/frontend/ASSETS.md). Review BERT/tokenizer and all
checkpoint/model terms separately. SenseVoice's upstream custom model terms are
not replaced by permissive FunASR/sherpa runtime licenses; the community release's
license points back to FunASR. Both native STT preparers require local legal texts
and provenance (`--license-dir` overrides `SOURCE/licenses`) and copy them verbatim
into `licenses/`, inventoried under `manifest.licenses.files`. The custom model
terms, source licenses and official model card are distinct. Community `LICENSE`
is preserved as a URL reference, not a standalone grant. No separate explicit
license for tokenizer/CMVN/config was found. Collected terms do not resolve their
historical applicability; consult upstream before redistribution.

`README.md` and `prepare.sh` provision the native speech roots and pinned offline
inputs without speech bridges. Runtime defaults select `models/stt2` and `models/tts2`,
CPU speech independent of the LLM device, with a shared measured two-thread voice
budget. The Dockerfile uses Python 3.14 and locked dependencies; it does not
prepare speech2 model artifacts. Compose mounts the source tree, including its
permanent model roots and disposable cache, and exposes GPU/audio
devices. A future deployment must include refreshed locked dependencies and
qualified artifacts in the mounted `models/` tree. No service/container/configuration was
started or changed for deployment; **Hoast remains stopped**.

Offline model-only smoke check, with no playback or network:

```sh
.venv/bin/python -m tools.check_speech2 --threads 2 --chinese
# Opt-in quantized alternative:
.venv/bin/python -m tools.check_speech2 --threads 2 --chinese --stt-backend ort --stt-model-dir models/stt2-community-ort
```

Each invocation creates a unique diagnostics directory with FLOAT WAVs, exact
expected/recognized text, stage timings and complete logged failure evidence.
