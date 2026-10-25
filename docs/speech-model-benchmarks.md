# Historical local speech-model benchmarks

Benchmarked on 2026-10-04. This is an isolated CPU feasibility/performance experiment,
not a production Hoast integration or an accuracy leaderboard.

This report preserves historical research measurements, including models that
are not application backends. **The research runner `tools/speech_benchmark.py`
has been deleted; this is an archive, not a runnable setup or reproduction guide.**
Cache paths below identify the original experiment layout and do not guarantee
that its checkpoints, environments, scripts, or artifacts are retained.
Hoast supports only native SenseVoice/Melo through STT2/TTS2; see
[the native guide](speech2-native.md) for current preparation and runtime behavior
and [native benchmarks](speech2-benchmarks.md) for the supported benchmark tools.

## Host and methodology

- Intel Core i5-8500T, six physical cores, AVX2, approximately 15.5 GiB RAM,
  no swap and no CUDA GPU. Intel integrated graphics were not used.
- FP32 reference and same-checkpoint partial dynamic INT8 inference, batch size
  one (the INT8 follow-up has its own section below). Six PyTorch intra-op
  threads, one inter-op thread; OpenMP/MKL limited to six and OpenBLAS to one.
  CosyVoice also uses its upstream in-process CPU ONNX frontend, whose sessions
  each use one intra-op thread.
- One model per fresh process. Checkpoints predownloaded; Hugging Face offline
  mode enabled during measurement. No compilation, VAD, punctuation model,
  diarization model, or forced aligner was added. FP32 reference runs are
  unquantized; the follow-up quantizes supported Linear layers.
- Local model loading includes model-specific Python imports and checkpoint
  initialization, but excludes common harness imports, downloads, and environment
  installation. Filesystem page caches were not flushed: **load is not a
  cold-disk or boot-time measurement**.
- Chinese runs precede English runs. One first call per language is followed by
  three measured warm repetitions; the English first call uses an already
  initialized/warmed process. JSON also marks the true first call for the model.
- Python, NumPy, and PyTorch RNG seeds reset to 42 before each call, outside the
  timed region. Upstream TTS sampling is retained, with the same seed for repeats.
- STT input loading/resampling is outside timing; model feature extraction,
  inference, and text decoding are inside. TTS timing fully consumes synthesis
  generators and includes waveform decoding and, for CosyVoice, reference
  feature extraction on every call. WAV writing is outside timing. CosyVoice
  prompt caching and streaming time-to-first-audio were not benchmarked.
- Per-operation RSS sampled every 20 ms; whole-process kernel high-water RSS and
  a separate process-tree sampler are also retained. RAM figures include the
  Python/runtime footprint and resident shared pages, not just checkpoint tensors.
  Sampling can miss transient peaks, and lifetime peaks can occur during loading.
- $\mathrm{RTF}=\text{inference seconds}/\text{audio seconds}$; below 1 means
  faster than real-time. STT uses input duration; TTS uses generated duration.
  Warm results are medians of three calls, not statistical confidence intervals.
- Initial setup-overlapped runs are preserved under `results/preliminary` but
  excluded from the final report. Final benchmarks run without concurrent
  installers, static checkers, tests, or other model benchmarks. Existing host
  services remain a source of contention and timing variability.

## Inputs

Shared STT fixtures are official Qwen examples, downloaded once and normalized
using ffmpeg to mono 16 kHz PCM WAV:

- Chinese: 4.2039375 seconds; `甚至出现交易几乎停滞的情况。`
- English: 15.05125 seconds; conversational speech beginning “Oh yeah, yeah, he
  wasn't even that big when I started listening to him…”. The remote example is
  not the older “Mr. Quilter” sample described in some documentation. SHA-256
  hashes and original files are retained to identify the actual inputs.

Sources:

- <https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav>
- <https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav>

TTS texts:

- English: `The lights are on. What else can I help you with?`
- Chinese: `灯已经打开了。还有什么需要我帮忙的吗？`

Qwen uses Ryan for English and Vivian for Chinese. CosyVoice uses the official
`asset/zero_shot_prompt.wav` with transcript
`希望你以后能够做的比我还好呦。` and the required
`You are a helpful assistant.<|endofprompt|>` prefix. This is zero-shot synthesis,
including cross-language synthesis for English, rather than Qwen's preset-speaker
mode; the TTS tasks are therefore not feature-identical.

## Exact models and environments

| Model            | Official checkpoint                  | Model revision                           | Runtime                                       |
| ---------------- | ------------------------------------ | ---------------------------------------- | --------------------------------------------- |
| Qwen3-TTS 0.6B   | Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice | 85e237c12c027371202489a0ec509ded67b5e4b5 | Python 3.12, qwen-tts, Transformers 4.57.3    |
| CosyVoice 3 0.5B | FunAudioLLM/Fun-CosyVoice3-0.5B-2512 | 29e01c4e8d000f4bcd70751be16fa94bf3d85a18 | Python 3.10, upstream source, Torch 2.3.1+cpu |
| SenseVoiceSmall  | FunAudioLLM/SenseVoiceSmall          | 3847d57b6bdf2dd8875cb1508d2af43d80a16bf7 | Python 3.10, FunASR 1.4.14                    |
| Qwen3-ASR-0.6B   | Qwen/Qwen3-ASR-0.6B-hf               | 7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c | Python 3.12, Transformers 5.18.0              |
| Paraformer-Large | funasr/paraformer-zh                 | d7811ee3ac581fbcfdeb37c98c6ba674028433dc | Python 3.10, FunASR 1.4.14                    |

Except CosyVoice, environments use matched Torch/torchaudio 2.9.1+cpu.
The original Qwen/Qwen3-ASR-0.6B checkpoint and qwen-asr environment are also
installed, but final measurements use the official native `-hf` checkpoint.
The bundled qwen-asr CPU SDPA/eager attention has a reported windowing issue
beyond eight seconds; this matters for the shared 15-second English input:
<https://github.com/QwenLM/Qwen3-ASR/issues/213> and
<https://github.com/QwenLM/Qwen3-ASR/pull/103>.

Qwen TTS and bundled ASR pin incompatible Transformers versions, so they cannot
share an unmodified upstream environment. CosyVoice pins Transformers 4.51.3
and NumPy 1.26.4; GPU-only dependencies were omitted/replaced with CPU equivalents.
Its OpenAI Whisper dependency is used for mel extraction; its unused Triton
requirement is intentionally omitted, so its dependency check reports that one
exception. No upstream source patches were needed for setup.

CosyVoice's constructor also tries to initialize optional `wetext` assets from
ModelScope. That attempt failed authentication and it selected no external text
normalizer; `text_frontend=False` is explicit for every benchmark call, so this
does not block synthesis. Its reported load time includes that optional failed
initializer attempt and the RAM-triggered `hoast` stop, not purely checkpoint
I/O. Upstream warns that the short Chinese synthesis text is shorter than half
its prefixed prompt text; this realistic short-response workload is retained,
and the warning/round-trip transcript are preserved rather than hidden.

CosyVoice source commit: `074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc`;
Matcha-TTS submodule: `dd9105b34bf2be2230f4aa1e4769fb586a3c824e`.
Full CosyVoice snapshot includes optional artifacts not all loaded for inference.
Paraformer is the 220M-parameter offline large Chinese/mixed-English checkpoint,
not the similarly named streaming or hotword model.

## FP32 reference measurements

All five models completed eight calls: one first call and three warm repetitions
for each language. Times are seconds; ZH is Chinese and EN is English.

| Model            | Load (s) | First ZH / EN (s) | Warm ZH / EN (s) | Peak RSS (GiB) | Warm RTF ZH / EN |
| ---------------- | -------- | ----------------- | ---------------- | -------------- | ---------------- |
| SenseVoiceSmall  | 9.23     | 0.371 / 0.629     | 0.232 / 0.643    | 3.06           | 0.055 / 0.043    |
| Paraformer-Large | 8.75     | 0.236 / 0.506     | 0.188 / 0.521    | 3.00           | 0.045 / 0.035    |
| Qwen3-ASR-0.6B   | 5.68     | 1.392 / 6.410     | 1.264 / 6.683    | 4.88           | 0.301 / 0.444    |
| Qwen3-TTS 0.6B   | 7.46     | 23.813 / 19.755   | 22.488 / 19.428  | 5.53           | 5.020 / 5.059    |
| CosyVoice 3 0.5B | 19.98    | 21.687 / 20.095   | 21.738 / 20.168  | 7.89           | 6.247 / 6.634    |

Peak RSS is the maximum of kernel lifetime high-water RSS and sampled peaks,
including loading. Steady RSS immediately after loading was 2.97 GiB for
SenseVoice, 2.96 GiB for Paraformer, 3.42 GiB for Qwen ASR, 4.02 GiB for Qwen TTS,
and 6.22 GiB for CosyVoice. It can increase during inference.

Qwen TTS's 5.53 GiB high-water RSS occurs during loading, not at its steady
inference footprint (~4.66 GiB after generation). Checkpoint header inspection,
without loading tensors, shows that `model.safetensors` stores 905,788,672 BF16
tensor values (1.687 GiB on disk; 3.374 GiB equivalent in float32). The separate
speech tokenizer stores 170,557,441 float32 values (0.635 GiB). Stored tensor
entries are not necessarily a count of unique trainable parameters; the nominal
0.6B model label does not describe the entire runtime/pipeline footprint.
Float32 equivalents total ~4.01 GiB before runtime/activation/cache overhead;
loading can temporarily retain checkpoint pages and converted tensors together.
Inspection evidence: `logs/qwen-tts-weight-sizes.log` and
`repro/qwen_tts_weight_sizes.py`. These are FP32 reference numbers, not best
achievable CPU results; see the paired INT8 follow-up below.

Generated TTS duration is 4.48s Chinese / 3.84s English for Qwen, and 3.48s
Chinese / 3.04s English for CosyVoice. Both produce 24 kHz mono audio. Thus the
similar wall times do not mean similar throughput: Qwen's RTF is about 5,
CosyVoice's about 6.2–6.6. Both are substantially slower than real-time in these
CPU reference configurations.

Warm timing ranges (minimum–maximum of three calls):

| Model            | Chinese (s)   | English (s)   |
| ---------------- | ------------- | ------------- |
| SenseVoiceSmall  | 0.226–0.237   | 0.629–0.657   |
| Paraformer-Large | 0.178–0.199   | 0.498–0.571   |
| Qwen3-ASR-0.6B   | 1.253–1.285   | 6.492–7.036   |
| Qwen3-TTS 0.6B   | 22.312–23.011 | 19.376–20.375 |
| CosyVoice 3 0.5B | 21.566–51.063 | 19.891–20.871 |

### Output checks

- All three STT models recover the Chinese reference words. SenseVoice retains
  language/emotion/event tags; Paraformer has no separate punctuation model.
- English is broadly recognizable with all three, but Paraformer includes
  spurious beginning tokens and recognition differences. These two clips are
  insufficient for an accuracy ranking; do not equate its speed with superior
  recognition quality.
- All 16 generated WAVs have correct duration/sample rate, finite non-silent
  samples, and no clipped samples under the validation threshold. Each
  model/language's four WAVs have identical hashes with reseeding.
- Independent SenseVoice transcription recovers the intended words from both
  TTS models in both languages, with punctuation differences. This verifies
  intelligible content at a basic ASR level, not naturalness or speaker fidelity.
- Eight lightweight harness regression tests pass; Ruff lint/format and Pyright
  pass. Result validation confirms all 40 calls and RTF arithmetic. A language
  metadata collision in Qwen result records was corrected to separate ISO
  fixture `language` from `output_language`; measured times/RAM/text were unchanged.

### Service state and contention

The final serialized suite ran 19:26:23–19:34:13 UTC. During CosyVoice loading,
available RAM fell below the supervisor's 1.5 GiB guard; it issued the authorized
`docker stop hoast` at 19:26:41. The original container exited at 19:27:02.
The worker sampler observed as little as 0.60 GiB available while the stop was
in progress; no benchmark was OOM-killed.

Docker events show a **concurrent external recreation** of `hoast`: the original
container was destroyed and a different ID started at 19:27:12. The benchmark
supervisor has no destroy/create/rename commands. Consequently, `hoast` was not
kept stopped throughout the final suite. The Chinese CosyVoice warm outlier
(51.063s) overlaps this recreation and is consistent with startup contention,
but its exact cause was not isolated. The other Chinese warm calls took about
21.6–21.7s. Treat these as live-host measurements, not service-isolated results.

At completion, the existing `hoast` container is running (~3.1 GiB at the final
snapshot); the explicit post-validation `docker start` request was a no-op
because the replacement was already running. Other container IDs are unchanged,
and this task did not restart them. Before/after snapshots, stop output, and
Docker event evidence are retained in `logs/`. In `suite.json`, `hoast_stopped`
records that the supervisor issued its one stop action, not continuous service
state; use Docker events for actual state transitions.

### Practical conclusion

For this host, **SenseVoiceSmall and Paraformer-Large are inexpensive in CPU time**
(approximately 18–29 times real-time on these clips), although their Python
reference runtimes still peak around 3 GiB. Qwen ASR is also faster than real-time
but costs considerably more CPU and RAM. For a multilingual assistant,
SenseVoice is a promising next integration candidate; Paraformer's mixed-English
quality needs broader testing.

Neither tested TTS reference runtime is suitable for low-latency interactive
speech on this CPU without optimization or different hardware. Qwen is more
RAM-efficient and has better RTF here; CosyVoice's prompt features were not
cached. This FP32 reference alone does not establish performance of optimized
streaming, quantized, or ONNX variants.

## INT8 follow-up

The follow-up uses **same-checkpoint PyTorch dynamic qint8 Linear layers with
FBGEMM**, not a fully INT8 pipeline or a vendor-optimized ONNX/GGUF export.
This CPU has AVX2 but not VNNI. Each variant runs in a fresh process with the
same fixtures, voices, seed, six threads, language order, and first-plus-three
warm calls as the reference. No packages or production dependencies changed.

Targeted modules:

- SenseVoice/Paraformer: ordinary Linear layers in the FunASR network.
- Qwen ASR: ordinary Linear layers in the native Transformers network.
- Qwen TTS all-Linear trial: talker/code-predictor plus the separately owned
  speech codec network. A selective `qwen-mlp` ablation targets only the main
  talker's MLPs; attention, text projection, output heads, code predictor and
  speech codec remain FP32 in that ablation.
- CosyVoice: LLM and flow networks. The ONNX speaker/tokenizer frontend and
  convolution-heavy HiFT vocoder remain unquantized.

Embeddings, convolutions, normalization, attention products and caches remain
floating point; dynamically quantized Linear layers retain FP32 biases.
Coverage means original weights with a packed INT8 copy divided by original
parameters in the selected scopes, **not** percentage of total pipeline weights
or RAM saved. Tied output-head/embedding weights can have both packed INT8 and
retained FP32 copies. CosyVoice's tied text output head is included even though
its text logits are not used for speech generation.

INT8 conversion is in place after loading the original FP32 network. Loading
therefore still needs FP32 headroom. Conversion time and phase-specific RSS are
recorded separately. Garbage collection plus glibc `malloc_trim(0)` releases
unused arenas after conversion. **Matched FP32 controls receive the same trim**;
compare these controls rather than attributing allocator-release savings to
quantization. No allocator trim occurs between inference calls.

CosyVoice's first INT8 attempt was terminated during its constructor's download
of unused optional text-normalizer assets, before conversion/inference. The
complete diagnostic log is preserved as
`logs/cosyvoice3-int8-unused-normalizer-download.log`. Paired final CosyVoice
runs temporarily suppress optional `wetext`/`ttsfrd` imports during construction
in the benchmark harness, using the upstream no-normalizer fallback. Every
synthesis call already sets `text_frontend=False`; this avoids network setup,
not a change to the measured synthesis task. It also makes load time different
from the original FP32 reference's optional failed initializer attempt.

### Paired INT8 measurements and quality outcome

Three STT INT8 variants and all five FP32 controls completed eight calls each.
CosyVoice INT8 completed eight calls **but failed the independent content check**.
Both Qwen TTS INT8 scopes failed on their first Chinese call; English/warm calls
were not accepted/run after the failure. Thus 72 completed paired calls and two
failed Qwen diagnostic calls are retained. Do not equate completed inference with
correct synthesis.

Times below are median warm seconds, ZH / EN. **CosyVoice INT8 numbers are
failure diagnostics, not an acceptable performance result.** Speedup is wall
time for the same input text/audio; TTS generated duration is not held constant.

| Model                | FP32 ZH / EN    | INT8 ZH / EN     | Speedup ZH / EN | INT8 RTF ZH / EN |
| -------------------- | --------------- | ---------------- | --------------- | ---------------- |
| SenseVoiceSmall      | 0.248 / 0.687   | 0.188 / 0.508    | 1.32x / 1.35x   | 0.045 / 0.034    |
| Paraformer-Large     | 0.202 / 0.518   | 0.163 / 0.400    | 1.24x / 1.30x   | 0.039 / 0.027    |
| Qwen3-ASR 0.6B       | 1.253 / 6.958   | 0.725 / 3.457    | 1.73x / 2.01x   | 0.173 / 0.230    |
| Qwen3-TTS (MLP only) | 23.266 / 20.329 | FAILED / not run | N/A             | N/A              |
| CosyVoice 3 0.5B     | 22.326 / 20.078 | 23.082 / 36.449  | 0.97x / 0.55x   | 3.699 / 3.505    |

RAM in GiB; each cell is **matched FP32 / INT8**. Post-setup is immediately
after allocator trim/conversion. Inference peak is the maximum per-call sampled
RSS. Lifetime peak also includes FP32 loading, conversion and common runtime.
Qwen failed INT8 inference is omitted from the fair comparison; its raw failed
trial metrics remain in JSON.

| Model                | Post-setup RSS | Inference peak RSS | Lifetime peak RSS |
| -------------------- | -------------- | ------------------ | ----------------- |
| SenseVoiceSmall      | 1.32 / 0.69    | 1.49 / 0.88        | 3.07 / 3.06       |
| Paraformer-Large     | 1.33 / 0.74    | 1.42 / 0.84        | 2.96 / 2.96       |
| Qwen3-ASR 0.6B       | 3.39 / 1.87    | 3.73 / 2.24        | 4.88 / 4.88       |
| Qwen3-TTS (MLP only) | 3.90 / N/A     | 4.87 / N/A         | 5.53 / N/A        |
| CosyVoice 3 0.5B     | 5.84 / 4.11    | 6.31 / 4.91        | 8.04 / 8.05       |

Conversion includes packing, coverage verification and allocator release.
Target fraction uses original selected-scope parameters, not whole-pipeline RAM.
Qwen MLP-only excludes the separate codec from its scope denominator.

| Model                         | Linear layers | INT8 weight entries | Target fraction | Convert (s) | Convert peak GiB |
| ----------------------------- | ------------- | ------------------- | --------------- | ----------- | ---------------- |
| SenseVoiceSmall               | 281           | 233.1M              | 99.6%           | 3.66        | 2.97             |
| Paraformer-Large              | 284           | 214.1M              | 97.3%           | 3.32        | 2.96             |
| Qwen3-ASR 0.6B                | 308           | 778.0M              | 99.4%           | 12.26       | 4.53             |
| Qwen3-TTS MLP only (failed)   | 84            | 264.2M              | 29.2%           | 3.64        | 4.23             |
| CosyVoice 3 0.5B              | 330           | 826.8M              | 98.6%           | 16.37       | 6.81             |

| Model                       | First ZH / EN (s) | Warm ranges ZH / EN (s)       | Audio ZH / EN (s) |
| --------------------------- | ----------------- | ----------------------------- | ----------------- |
| SenseVoiceSmall             | 0.316 / 0.566     | 0.179–0.200 / 0.506–0.587     | 4.204 / 15.051    |
| Paraformer-Large            | 0.221 / 0.408     | 0.149–0.185 / 0.386–0.405     | 4.204 / 15.051    |
| Qwen3-ASR 0.6B              | 0.849 / 4.025     | 0.701–0.783 / 3.445–3.529     | 4.204 / 15.051    |
| Qwen3-TTS MLP only (failed) | 184.322 / not run | not run                       | 40.880 / not run  |
| CosyVoice 3 0.5B            | 27.508 / 36.520   | 22.973–23.153 / 36.396–36.826 | 6.240 / 10.400    |

**Practical result:** STT warm time improves roughly 1.2–2.0x and observed
inference-peak RSS falls about 40% versus matched trimmed FP32. Startup peaks
stay approximately unchanged because FP32 weights load before conversion.
Neither tested TTS dynamic-INT8 configuration passes the short-phrase content
sanity check; RAM reduction alone does not make them suitable for use.

### INT8 STT transcriptions

These are the two shared Qwen example WAVs described in **Inputs**, not a speech
corpus or WER/CER evaluation. SenseVoice metadata tags are omitted from the
following readable text; raw transcripts remain in the JSON artifacts.

| Model            | Chinese output                       |
| ---------------- | ------------------------------------ |
| SenseVoiceSmall  | 甚至出现交易几乎停滞的情况。         |
| Paraformer-Large | 甚至出现交易几乎停滞的情况           |
| Qwen3-ASR 0.6B   | 甚至出现交易几乎停滞的情况。         |

English:

**SenseVoiceSmall**

> Oh yeah, yeah, he wasn't even that big when I started listening to him, but
> and his solo music didn't do overly well, but he did very well when he started
> writing for other people.

**Qwen3-ASR 0.6B**

> Oh yeah, yeah. He wasn't even that big when I started listening to him, but in
> his solo music, didn't do overly well, but he did very well when he started
> writing for other people.

**Paraformer-Large**

> hmm oh yeah yeah he wasn't even that big when i started listening to him but
> in his solo music didn't do overwell but he did very well when he started
> writing for other people

Chinese words are unchanged versus FP32. SenseVoice's English text is unchanged
including its raw tags. Qwen drops the initial `Mhm.` and changes punctuation/
connective wording. Paraformer's beginning filler changes from `hh HH ah` to
`hmm oh`; its `overwell` recognition defect remains. These are small-sample
observations, not evidence of accuracy equivalence under quantization.

### Qwen TTS generation-boundary failure

The aggressive all-Linear Qwen TTS trial converted 359 Linear layers and
632.3M weight entries. Conversion took 10.19s; post-conversion RSS was 2.68 GiB,
but FP32-loading lifetime peak was still 5.53 GiB. Its first Chinese call took
**128.374s and produced 40.88s of audio**, compared with 4.48s in FP32.
The trial was stopped during its next repetition; it is incomplete and must not
be interpreted as a successful speed/quality result.

This checkpoint's nominal “12-Hz” codec actually decodes 1,920 samples/frame at
24 kHz (12.5 frames/s). Prefill emits no complete frame, so a 512-step generation
limit corresponds to **511 complete frames / 40.88s**. The output reaches that
boundary; waveform length alone cannot distinguish missing EOS from EOS sampled
on the final allowed step. The harness detects this boundary, preserves the WAV
and incomplete result, and raises rather than declaring successful synthesis.

The nested code predictor generates 15 codebook tokens per talker step. Dynamic
INT8 overhead on these small autoregressive operations and changes to sampled
logits/termination can both matter. Keeping the same RNG seed does not guarantee
the same output after changing model precision. A main-talker **MLP-only** INT8
ablation leaves the predictor, attention, logits heads and codec FP32 to avoid
these particularly aggressive changes. It is a different partial-INT8 scope,
not evidence that the original all-Linear variant works.

The MLP-only retry converted 84 layers / 264.2M weight entries, taking 3.64s
and reaching 3.16 GiB post-conversion RSS. **It also failed:** its first Chinese
call took 184.322s, returned 511 frames / 40.88s, and raised the explicit
boundary error. The intended phrase was not recovered by independent ASR in
either Qwen INT8 trial. No English INT8 sample or valid warm Qwen INT8 result
is claimed.

Preserved evidence: `results/int8-preliminary/qwen-tts-int8-all-linears.json`,
`logs/qwen-tts-int8-all-linears.log`, `results/qwen-tts-int8-mlp.json`,
`logs/qwen-tts-int8-mlp-final.log`, and both diagnostic Chinese WAVs.

### TTS content checks and saved samples

Independent **FP32 SenseVoice** checks the saved 24 kHz WAVs after resampling to
16 kHz. The tiny exact-content sanity check removes metadata tags, punctuation
and case before comparison to the input phrase. It is not a formal accuracy or
perceptual-quality score. All four FP32 model/language samples recover the
intended words. INT8 failures (metadata omitted):

- CosyVoice ZH, 6.24s: `灯已经打开了，还有说是要我帮当当。`
- CosyVoice EN, 10.40s: `The next I are yy, white, white, Yy, Yy, Yy, A, school I see, I see.`
- Qwen all-Linear ZH, 40.88s: `超好。`
- Qwen MLP-only ZH, 40.88s: `千1通把由同通通告。`

CosyVoice FP32 duration is 3.48s ZH / 3.04s EN. The longer INT8 output and wrong
content make its apparently improved RTF unsuitable as an optimization claim.
These failed scopes do not establish that every possible quantization scheme
will fail; calibrated or weight-only/full-graph backends were not tested.

Saved **eight** representative WAVs under:

`~/music-debug/speech-int8-2026-10-04/`

- `qwen-tts-fp32-{zh,en}.wav`
- `cosyvoice3-fp32-{zh,en}.wav`
- `cosyvoice3-FAILED-quality-int8-{zh,en}.wav`
- `qwen-tts-FAILED-all-linear-int8-zh.wav`
- `qwen-tts-FAILED-mlp-only-int8-zh.wav`

`README.md` identifies every file, duration, status and readable independent
transcript; `validation.json` preserves raw tags/hashes/RMS/clipping diagnostics.
**Use FP32 samples for normal listening; FAILED files are diagnostic only.**
The existing unrelated music/AEC samples were not overwritten.

Validation checked all 72 completed paired calls and RTF arithmetic, 24 WAVs
from completed synthesis runs for finite/non-silent audio, rate/duration and
clipping under the threshold, and both finite failed Qwen diagnostic WAVs.
Each completed model/precision/language group's four reseeded WAVs have the
same SHA-256, so representative content-check results apply to those identical
files. Copied sample hashes match their measured artifacts.

Twenty-one harness tests pass; Ruff lint/format and Pyright pass. Six expected
PyTorch eager-quantization deprecation warnings are retained; this API is
experimental tooling here, not a production deployment dependency change.

### Follow-up service state and contention

Measured workers ran serially from 22:41:43 to 23:07:30 UTC, including failed
attempts and paired reruns. No tests, installers or other model benchmarks ran
concurrently; validation/static checks followed measurement. The supervisor
issued four authorized stops of the current `hoast` IDs for CosyVoice load
headroom. It restores only an identical stopped ID, never a replacement.

`hoast` was externally recreated again rather than remaining stopped. Retained
Docker events corroborate stop/destroy/create/rename/start sequences during the
later CosyVoice runs; earlier events had fallen out of the daemon's retained
history by the final query. Per-run identity snapshots and complete supervisor
stop logs are also preserved. Do not treat these timings as service-isolated
measurements or assume any cold-call outlier has a single proven cause.

At the final snapshot `hoast` is running with a replacement ID starting at
23:00:52 UTC. Other service IDs are unchanged. The task did not modify production
configuration/dependencies or restart unrelated containers. Evidence:
`results/int8-suite-attempt1/`, `results/int8-service-actions.json`,
`logs/int8-suite-attempt1.log`, `logs/int8-suite-cosy-retry.log`,
`logs/int8-docker-events.txt`, and before/after container snapshots.

### Historical INT8 artifact inventory

The deleted runner used `<model>-int8` and `<model>-fp32-trim` artifact stems for
the paired trials. A tagged JSON was initialized as `status: incomplete` before
loading and marked `completed` only after all requested calls finished. Process
exit status, coverage, call count and content checks remain necessary when
interpreting historical records, particularly those predating the status field.
Full tracebacks were recorded in the corresponding logs.

Original artifact paths under `.cache/speech-bench/`:

- `results/<model>-int8.json`, `results/<model>-fp32-trim.json`: all paired calls.
- `results/int8-summary.json`, `results/int8-tables.md`: derived comparisons.
- `results/int8-stt-transcripts.json`: exact transcripts from every repetition.
- `results/int8-audio-validation.json`: waveform checks and independent ASR.
- `logs/*-int8-final.log`, `logs/*-fp32-trim-final.log`: complete per-model logs.
- `repro/run_int8_suite.py`, `repro/summarize_int8.py`,
  `repro/verify_int8_outputs.py`, `repro/label_int8_quality.py`: serialized
  supervisor, validation and explicit quality-labeling scripts. The historical
  workflow validated outputs, summarized results, then labeled quality, and
  refused to overwrite an existing dated sample directory.

## Community native Qwen3-TTS INT8 and neutral-voice follow-up

Tested the community [gabriele-mastrapasqua/qwen3-tts](https://github.com/gabriele-mastrapasqua/qwen3-tts)
runtime at clean commit `ef339be58a778b062e1c14382347964552eae007`.
It is **pure C**, not an official Qwen C++ runtime or downloadable official INT8
checkpoint. The original official 0.6B CustomVoice weights and nested codec were
reused unchanged, at the revision listed above.

### Native build and precision

- GCC 16.2.1; `make blas SIMD=portable` (`-mavx2 -mfma -O3 -ffast-math`).
  Source, binary, build logs and dependency hashes are retained in the cache.
- Isolated OpenBLAS 0.3.30, commit
  `993fad6aebbce34a97d3f8c34d6d79d35b64cc48`, `TARGET=HASWELL`,
  `USE_THREAD=0 USE_LOCKING=1 NOFORTRAN=1 NO_LAPACK=1`. No system or production
  dependencies were installed/changed.
- **Thread-safe BLAS is essential even when BLAS itself uses one thread.**
  Runtime workers call BLAS concurrently. An initial serial build without
  `USE_LOCKING=1` produced corrupt speech and `Bad memory unallocation`
  diagnostics. Those runs, their WAVs, library and complete failure logs are
  preserved in `*-unsafe-blas` artifact directories and **excluded**. A clean
  OpenBLAS rebuild with locking fixed the content failures; the native source
  and binary were unchanged. Do not reproduce the unsafe build.
- Native INT8 quantizes Talker and Code Predictor transformer projections and
  the CP output heads, in memory during loading. Embeddings/other heads retain
  BF16 weights, activations/attention are floating point, KV storage is BF16,
  and the speech decoder remains F32 on this host. Default prefill converts
  BF16 weights to F32 for OpenBLAS SGEMM.
- On this **AVX2-without-VNNI CPU**, the default single-request INT8 GEMV widens
  byte weights and uses FP32 FMA. It is not an end-to-end integer pipeline.
  The experimental `QWEN_AVX2_INT8_GEMV` integer-dot arm was not enabled.
- Original BF16 weights and conversion/cache buffers remain present by default;
  allocated INT8 copies can **increase**, rather than halve, RSS. The optional
  BF16-release/quantized-prefill flags were not tested.

### Resident full-WAV diagnostic results

One fresh plain server per precision, one worker, six runtime threads pinned to
CPUs 0–5; engine-owned decoder pool and serial BLAS. Chinese precedes English,
with four calls per language: the first for that language plus three retained
warm repetitions. Seed 42, temperature 0.5, top-k 50, top-p 1, repetition penalty
1.05, no style instructions/emotion/gain/time-stretch, model-default prompt
layout, EOS-assistance heuristic disabled and a 512-frame CLI generation limit.
Native CP heads use argmax, unlike the official Python sampled subtalker:
**native-versus-Python is not a quantization-only comparison**.

The server performs built-in synthesis pre-warm before listening. Startup-to-ready
includes that work and INT8 conversion; the first measured user request is not
first inference. English's first language-specific request is the fifth measured
request overall. No storage/page-cache-cold claim is made.

| Native mode                | Startup + pre-warm (s) | Warm wall ZH / EN (s) | Warm audio ZH / EN (s) | Warm TOTAL_RTF ZH / EN | Observed peak RSS (GiB) |
| -------------------------- | --------------------- | -------------------- | --------------------- | --------------------- | ---------------------- |
| BF16/unquantized control † | 8.357                 | 9.665 / 13.434       | 4.320 / 6.000         | 2.237 / 2.239         | 2.984                  |
| Partial weight INT8        | 9.880                 | 6.359 / 7.158        | 4.320 / 4.800         | 1.472 / 1.491         | 3.500                  |

Warm values are medians of three calls. Wall time is loopback `/v1/tts` full-WAV
receipt, including generation, decoder drain, WAV encoding and transport,
excluding model startup and subsequent disk writing. `TOTAL_RTF` divides each
request's wall time by its actual generated duration. Audio lengths differ,
especially for English: do not describe the latency ratio as an equal-duration
kernel speedup. Both modes are **slower than real-time** on these samples.
No streaming continuity, HTTP audible TTFA or concurrency capacity was qualified.

Peak RSS is the process's kernel `VmHWM` observed through the requests, including
loading/pre-warm and resident mmap pages, not model parameter bytes, PSS or
per-call allocations. Shutdown peaks were not observed. Default native INT8 uses
about **0.52 GiB more** than the native unquantized control in this experiment.

**Content checks:** all eight INT8 baseline calls recover the intended phrases
with FP32 SenseVoice after metadata/punctuation/case removal. BF16 Chinese also
passes all four. **† BF16 English is flagged for review:** SenseVoice inserts
`W, What` before the second sentence in all four calls, so its exact-content
count is 0/4; these timings remain diagnostic, not exact-content successes.
Independent FP32 Qwen3-ASR recovers the exact intended words from all four smoke
WAVs, including BF16 English. That disagreement does not establish an audible
stutter; listening is required. The flagged sample is retained explicitly.

All **27 corrected generated WAVs** (four smoke, sixteen baseline, five voice
candidates and two fresh CLI probes) pass mono 24-kHz geometry, finite/non-silent
signal, clipping threshold and saved-hash checks. All sixteen INT8 outputs among
these pass the SenseVoice phrase check, but they reuse just one text per language
and one seed. This is not a broad intelligibility or naturalness qualification.
Server frame-counter deltas match waveform length and stay below the limit;
normal EOS termination is inferred from successful below-cap completion and
source control flow because the server suppresses detailed generation logs.
CLI probes explicitly log EOS at frames 98 (BF16) and 49 (INT8), well below cap.

The first Chinese call and warm calls have different waveforms/durations despite
resetting the seed: BF16 4.40 → 4.32s; INT8 4.48 → 4.32s. Every repetition was
transcribed rather than assuming its content from a representative hash.

Separate fresh-process **English CLI diagnostics**, not resident warm numbers:
BF16 prefill 568ms, Talker 4,594ms, CP 11,695ms, decoder 4,899ms; INT8 prefill
544ms, Talker 1,744ms, CP 3,597ms, decoder 2,777ms. Decoder work overlaps generation,
so these component times must not be added as independent serial costs. Internal
first-chunk timing was 2,554ms / 1,938ms respectively; this is **not** HTTP audible
TTFA. Complete load/process-wall/RSS evidence remains in the CLI JSON/logs.

### Less dramatic speech: available controls

For the **0.6B CustomVoice model tested here**, spoken-style `instruct` is not
supported. The native CLI warns and ignores it, and the official Python wrapper
sets it to `None` for `0b6`, despite the small-model card's instruction-control
wording. Do not expect adding “speak neutrally” to work on this checkpoint.
Also specify temperature explicitly: an ineffective native CLI instruction can
still trigger a temperature increase before the ignore warning.

Useful experiments, not guaranteed neutrality:

- **Choose a gentler preset.** Official descriptions call Serena warm/gentle
  and Uncle_Fu mellow. They are good Chinese audition candidates. Aiden is an
  alternative English voice, but officially described as sunny, not flat;
  Ryan is dynamic/rhythmic and Vivian bright/slightly edgy.
- **Fix the seed and lower sampling temperature.** Test 0.5, or 0.3 against
  0.5. This reduces sampling randomness, not emotional intensity by definition.
  Native CLI defaults to 0.9 while HTTP defaults to 0.5; startup CLI sampling
  flags alone do not determine each HTTP request.
- **Omit expressive controls.** `--emotion neutral` is not a valid named native
  emotion. Inline `[neutral]` resets added emotion to baseline; it does not
  flatten the preset's learned prosody. Experimental emotion/steering assets
  were not downloaded or tested.
- **Rate is not an emotion suppressor.** Native full-WAV `--rate` is a
  post-generation, pitch-preserving time-stretch, not a prosody instruction;
  it does not apply to CLI streaming. All samples here keep rate and volume 1.
- **Actual instruction guidance:** Qwen **1.7B** supports prose such as “Speak
  in a neutral, matter-of-fact conversational tone, at a normal pace, with
  restrained emphasis.” It is soft guidance, not a guarantee; 1.7B was not
  downloaded/tested in this follow-up.
- **CosyVoice3:** for zero-shot synthesis, try a clean matter-of-fact reference
  recording with its correct transcript instead of expressive donor audio.
  Official `inference_instruct2` also accepts style instructions, but neutral
  prose was not tested here and is not a guaranteed preset. Its non-streaming
  speed control changes duration, not emotion.

Source evidence:
[official speaker descriptions](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice),
[official Qwen instruction gate](https://github.com/QwenLM/Qwen3-TTS/blob/main/qwen_tts/inference/qwen3_tts_model.py),
and [CosyVoice inference APIs](https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/cli/model.py).
Pinned native evidence: `main.c` instruction/temperature handling,
`qwen_tts_emotion.c` emotion table, `qwen_tts_compose.c` neutral reset, and
`qwen_tts_server.c` per-request defaults.

### Saved audition samples and evidence

Nine new WAVs plus `README.md` and hash/transcription evidence are saved under:

`~/music-debug/speech-native-qwen-2026-10-04/`

Four native BF16/INT8 bilingual baseline samples include the explicitly
`REVIEW-content-`-prefixed BF16 English sample. Five **INT8** voice candidates:

| Language | Preset    | Temperature | Audio (s) | SenseVoice phrase check |
| -------- | --------- | ----------- | --------- | ----------------------- |
| English  | Aiden     | 0.5         | 3.280     | Exact                   |
| Chinese  | Serena    | 0.5         | 3.840     | Exact                   |
| Chinese  | Uncle_Fu  | 0.5         | 6.720     | Exact                   |
| English  | Ryan      | 0.3         | 4.480     | Exact                   |
| Chinese  | Vivian    | 0.3         | 4.640     | Exact                   |

**All five recover the intended phrase, but perceived neutrality needs listening.**
SenseVoice emotion tags are retained as raw metadata, not used to score neutrality.
No existing music-debug or prior Python/INT8 sample was overwritten.

Historical source/evidence paths under `.cache/speech-bench/`:

- `repro/build_native_qwen.sh`: isolated build with OpenBLAS locking for
  concurrent callers.
- `repro/run_native_qwen.py`: serialized resident smoke/baseline/voice trials;
  the historical workflow refused existing trial paths.
- `repro/validate_native_qwen.py` and
  `repro/summarize_native_qwen.py`: waveform/content gates, publication and
  derived diagnostic timings, using the isolated FunASR interpreter.
- `repro/native_cli_probe.py`: separate BF16/INT8 English CLI evidence.
- `repro/native_secondary_asr.py`: secondary smoke transcription with the
  isolated Qwen3-ASR-HF interpreter.
- `results/native-qwen/`, `outputs/native-qwen/`, `logs/native-*.log`:
  exact commands, environments, binary/library hashes, requests, timings,
  process memory, service snapshots, quality flags and full diagnostics.
- `results/native-cpu-check-safe/`: native/fallback kernel checks and resolved
  dispatch, clean-source/binary identity and topology. Preflight passes with
  a CPU-governor warning; bandwidth roofs were not measured. This is a small
  diagnostic, not a production server SOAK.

Inference runs were sequential with no concurrent builds, installs, tests or
other models. Fixed precision/language order, three warm calls, one text/seed,
live host services and single-shot voice auditions limit generalization.
An authorized exact-ID Hoast stop was followed by external container recreation;
the replacement remained running, so measurements are **not service-isolated**.
The removed original was not restarted and replacements/unrelated services were
not restarted by this task. Production configuration/dependencies are unchanged.
Final integrity checks confirm all prior published samples and sixteen original
FP32 WAV hashes are preserved, nine new published hashes match, unrelated
container IDs are unchanged and no native workers survive. Native/fallback
self-tests and dispatch gates pass; Ruff lint/format and Pyright pass for the
six reproduction/validation scripts, and the build script passes `sh -n`.
No C source was modified.

## Isolated MeloTTS CPU setup (2026-10-05)

At the time of this experiment, MeloTTS EN/ZH was installed and verified
separately; **live Hoast used Kokoro**, not MeloTTS. Historical setup details,
exact revisions/dependency choices,
model/frontend caveats and measured results are in [MeloTTS setup](melotts-setup.md).

At six CPU threads, median warm synthesis for the short EN-US / Chinese / mixed
phrases was **1.288 / 1.251 / 0.830s**, with RTF **0.393 / 0.408 / 0.376**.
First EN/ZH synthesis was about 4.6s plus process import/model startup; resident
warm numbers must not be confused with fresh-CLI latency. Native output is
44.1 kHz. These are two warm repetitions per phrase on a live host, not streaming
or perceptual-quality qualification.

All 13 audition WAVs pass waveform/hash checks; FP32 SenseVoice content matches
11/13 and independent FP32 Qwen3-ASR matches 13/13. Warm EN-US has an explicitly
labeled ASR disagreement on “with” versus “wet”; no audible-defect claim is made.
Seven labeled samples (five English presets, Chinese and mixed) are saved in
`~/music-debug/speech-melo-2026-10-05/`. The setup needed no source patch, service
stop/restart or production dependency change. Source/assets and 17 prior
published sample hashes are verified; Ruff/Pyright and wrapper syntax pass.

## Historical experiment layout

The deleted research runner used separate per-model interpreters and serialized
workers. Its suite supervisor could stop `hoast` under the user's authorization
to protect available RAM; the runner itself did not stop services. These are
historical methodology details, not current operational instructions.

Original checkpoints, environments, complete failure logs, package freezes,
input hashes, metrics, and generated WAVs were stored under
**`.cache/speech-bench/`**:

- `models/manifest.json`: exact downloaded model/source revisions.
- `results/<model>.json`: per-call wall time, RSS, transcripts/audio metadata.
- `results/suite.json`: process wall time, tree RSS, available RAM/service action.
- `results/summary.json`: validated median/range summary.
- `results/audio-validation.json`: waveform checks and independent ASR sanity checks.
- `results/*-freeze.txt`: full installed versions, including CPU wheel suffixes.
- `inputs/`, `outputs/`: exact STT inputs and generated audio.
- `logs/<model>-final.log`, `logs/suite-final.log`: final complete diagnostics.
- `cosy-setup.md`, `cosy-requirements-cpu.txt`, `cosy-constraints-cpu.txt`:
  isolated CosyVoice installation details and compatibility choices.
- `repro/`: retained download/suite/summary/audio-validation scripts.

The experiment used Python 3.10/3.12 environments with CPU PyTorch wheels.
CosyVoice additionally needed CPU requirements/constraints and a Whisper
installation without the unused Triton dependency. These research environments
and unsupported model families are not application prerequisites; current
preparation instructions are in the native guide linked above.

## Scope and limitations

This measures short offline batch-one requests on a live home-server workload.
It does not establish multilingual accuracy, robustness to room noise, streaming
latency, perceptual voice quality, long-recording behavior, or multi-user
throughput. Finite, non-silent WAVs and independent ASR round trips are sanity
checks, not a listening test or proof of TTS fidelity. Optimized ONNX/GGUF/int8
full-graph backends and prompt caching may yield different performance; those
were not measured here. The INT8 follow-up uses PyTorch dynamic Linear layers,
not those optimized full-graph runtimes.
