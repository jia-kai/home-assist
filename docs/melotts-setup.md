# Isolated MeloTTS CPU audition

This historical audition was set up on 2026-10-05 on the i5-8500T home server,
before native STT2/TTS2 integration. It records an isolated upstream Melo
experiment, not the supported application runtime. For current preparation and
deployment use [the native guide](speech2-native.md) and
[the speech overview](speech.md). The alternate interpreter and commands below
describe the audition only; preserve model/frontend data needed by native
preparation independently of its environment.

## Use the installed environment

From the repository root:

```sh
# Each CLI invocation loads a fresh model. Choose fresh output names.
.cache/speech-bench/repro/melo.sh \
  'The lights are on. What else can I help you with?' \
  .cache/speech-bench/outputs/melo/manual-english.wav \
  --language EN --speaker EN-Default \
  > .cache/speech-bench/logs/melo-manual-english.log 2>&1

.cache/speech-bench/repro/melo.sh \
  '帮我打开 living room 的灯。' \
  .cache/speech-bench/outputs/melo/manual-mixed.wav \
  --language ZH \
  > .cache/speech-bench/logs/melo-manual-mixed.log 2>&1
```

The wrapper selects the isolated interpreter, offline model/frontend caches,
CPU, six OpenMP/MKL threads and one OpenBLAS thread. Only EN and ZH synthesis
assets are staged. Chinese uses Melo's `ZH_MIX_EN` frontend, including for pure
Chinese. The checkpoint's `ZH` speaker ID is **1**, not 0. English presets:
`EN-Default`, `EN-US`, `EN-BR` (British), `EN_INDIA`, `EN-AU`.

For an eventual resident integration, retain separate EN and ZH model instances
rather than spawning this CLI for every response. That integration, combined
resident memory and streaming/concurrency behavior have **not** been tested.

## Exact installation

Everything lives under `.cache/speech-bench/`:

- `source/MeloTTS`: unchanged MyShell source at
  `209145371cff8fc3bd60d7be902ea69cbdb7965a` (2024-12-24).
- `envs/melo`: Python 3.10.21; Torch/Torchaudio **2.5.1+cpu**;
  Transformers 4.27.4; NumPy 1.26.4; NLTK 3.8.1; cached-path 1.6.3;
  huggingface-hub 0.23.5; setuptools 80.10.2.
- `repro/melo-requirements.in`: scoped inference dependencies.
- `repro/melo-requirements.lock`: complete installed freeze, including local
  Melo source reference and all transitive versions. Install it with `--no-deps`
  to preserve the intentionally scoped environment.
- `models/melo-hf`, `models/melo-nltk`: isolated model/tokenizer and English G2P
  data caches. The English/multilingual BERT weights are required conditioning,
  not optional disabled features in these official configs.
- `results/melo/assets.json`: immutable revisions, sizes and SHA-256s for every
  staged HF asset. English checkpoint revision
  `bb4fb7346d566d277ba8c8c7dbfdf6786139b8ef`; Chinese
  `af5d207a364ea4208c6f589c89f57f88414bdd16`.

Upstream's eager multilingual frontend imports require small Japanese, Korean,
French and Spanish tokenizers plus MeCab/UniDic-lite even for EN/ZH. **Their TTS
checkpoints/BERT weights and the large full UniDic dictionary were not downloaded.**
UI/training packages `gradio`, `tensorboard`, and full `unidic` were intentionally
omitted. As a result the upstream distribution's broad dependency metadata is
not fully satisfied; its inference API/CLI are the tested scope, not its UI or
training commands. Avoid blindly installing upstream requirements or upgrading
Torch/Transformers in this environment.

The first dependency-resolution attempt failed because cached-path 1.6.3 needs
huggingface-hub below 0.24. Pinning 0.23.5 resolved it. Complete install/download
and warning/exception logs are retained as `logs/melo-*`. No upstream source
patch was needed. Old-library deprecation warnings and the trusted-checkpoint
pickle-loading warnings are retained; do not load untrusted `.pth` files.

To reconstruct this setup on this host, preserve the pinned checkout and lock:

```sh
export TMPDIR="$PWD/.cache/speech-bench/tmp/melo"
export HF_HOME="$PWD/.cache/speech-bench/models/melo-hf"
export NLTK_DATA="$PWD/.cache/speech-bench/models/melo-nltk"
mkdir -p "$TMPDIR"
# Only create a new venv if the existing one is absent or intentionally archived.
uv venv --python 3.10 .cache/speech-bench/envs/melo
uv pip install --python .cache/speech-bench/envs/melo/bin/python \
  --no-deps -r .cache/speech-bench/repro/melo-requirements.lock \
  --extra-index-url https://download.pytorch.org/whl/cpu
.cache/speech-bench/envs/melo/bin/python .cache/speech-bench/repro/download_melo.py
```

The downloader refuses a revision change once the assets manifest exists.
Retain that manifest and checkpoints for exact reproduction; the live `main`
refs are populated only for unmodified tokenizer imports. Measurements run
with Hugging Face/Transformers offline flags.

## Measured CPU behavior

Fresh processes, one model at a time; FP32, six intra-op threads and one inter-op
thread. Speed 1.0; upstream defaults `sdp_ratio=0.2`, `noise_scale=0.6`,
`noise_scale_w=0.8`; seed 1234 reset before each call. No emotion/style instruction,
postprocessing gain or time stretching. Output is **44.1-kHz mono PCM WAV**.
Synthesis wall time includes text frontend, BERT, full waveform generation and
file writing, but excludes model import/loading and subsequent validation.

| Input / voice                   | First text call (s) | Warm synthesis (s) | Warm audio (s) | Warm RTF | Model-process peak RSS (GiB) |
| ------------------------------- | ------------------- | ------------------ | -------------- | -------- | --------------------------- |
| English / EN-US                 | 4.597               | 1.288              | 3.278          | 0.393    | 2.119                       |
| Chinese / ZH                    | 4.545               | 1.251              | 3.069          | 0.408    | 2.786                       |
| Chinese + `living room` / ZH     | 1.853               | 0.830              | 2.209          | 0.376    | 2.786                       |

Warm values are the medians of **two** subsequent calls, not a broad benchmark.
The first mixed call follows three Chinese calls and already has its BERT loaded.
Fresh import + TTS load took **12.953s EN / 6.975s ZH**, excluding lazy BERT
initialization in the first synthesis. EN was the first-ever environment import
and can include Numba JIT/cache preparation. These are not disk-cold guarantees.

Warm synthesis is faster than playback for these short phrases. This does not
establish audible time-to-first-audio, streaming continuity or concurrent capacity.
Hoast remained running; results include live-server contention. There were no
concurrent installations, checkers or other benchmark models during synthesis.
Available RAM remained above 6 GiB; no Hoast stop/restart was required. The
supervisor bounds workers to 600 seconds, 5 GiB RSS and 2 GiB available RAM.

The first EN/ZH call consumes extra random draws during lazy BERT initialization,
so its waveform differs despite the same seed. Warm repetitions have identical
hashes. All calls, not just representatives, are checked independently.

## Audition and verification

Generated English text: “The lights are on. What else can I help you with?”
Chinese: “灯已经打开了。还有什么需要我帮忙的吗？”
Mixed: “帮我打开 living room 的灯。”

Melo exposes speaker/rate/noise controls, **not a neutrality/emotion instruction
switch**. Voice quality and perceived neutrality require listening; ASR emotion
tags are not used as voice-quality scores.

Evidence and reproduction:

- `results/melo/{en6,zh6}.json`, `outputs/melo/*.wav`, `logs/melo-{en6,zh6}.log`:
  13 calls, geometry, actual durations, wall time/RTF, RSS, hashes and diagnostics.
- `repro/run_melo.py --language EN --threads 6 --label <fresh-en-label>` and the
  corresponding ZH command: bounded offline auditions; refuse existing outputs.
- `repro/validate_melo.py --asr sensevoice`, using `envs/funasr/bin/python`, and
  `--asr qwen`, using `envs/qwen-asr-hf/bin/python`: independent FP32 transcripts
  after exact rational 44,100-to-16,000 Hz resampling. Run sequentially.
- `results/melo/validation-{sensevoice,qwen}.json`: raw transcripts and exact
  normalized-content flags. ASR disagreements are review flags, not proof of
  audible defects or proof of perfection.

Seven labeled audition WAVs, a listening guide and hash/transcription evidence:

**`~/music-debug/speech-melo-2026-10-05/`**

This includes all five English presets, Chinese, and mixed Chinese/English.
Start with `en-EN-Default.wav`, `zh-ZH.wav`, and `mixed-ZH.wav`. The warm EN-US
sample is labeled `REVIEW-ASR-en-EN-US.wav`: SenseVoice substitutes “wet” for
“with”, while Qwen3-ASR recovers the intended sentence exactly. SenseVoice
matches **11/13** audition WAVs; Qwen3-ASR matches **13/13**. Both match all
Chinese/mixed calls and the other English presets. Listening is needed to
resolve the EN-US disagreement and choose the most pleasant voice.

All 13 audition WAVs pass finite/nonempty mono geometry, native sample rate,
duration, non-silence, clipping and stored-hash checks. A separate fresh mixed
CLI smoke also passes waveform checks (`outputs/melo-cli/mixed.wav`,
`logs/melo-cli-smoke.log`); it is not one of the measured/ASR-audited 13 calls.
Ruff lint/format, Pyright for all four reproduction scripts, and `sh -n` for
the CLI wrapper pass. Publication verifies clean pinned source, identical
installed Python source and checkpoint/tokenizer hashes, seven copied sample
hashes, **17 preserved prior published sample hashes**, and the same running
Hoast ID/start time without a restart. See `results/melo/run_manifest.json` and
`repro/publish_melo.py` (publication refuses to overwrite the dated directory).
