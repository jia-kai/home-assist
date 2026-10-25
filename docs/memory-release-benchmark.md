# Speech intermediate-memory release qualification

## Decision

**Retain intermediate storage for both STT2 and TTS2 on the measured CPU setup.**
Neither engine passes the complete per-case policy, and neither shows a useful
idle-memory reduction. This is an enablement recommendation, not a serving-default
change or a claim about unmeasured devices.

| Engine | Median screen | p95 failures                          | Exact outputs | Recommendation |
| ------ | ------------- | ------------------------------------- | ------------- | -------------- |
| STT2   | All six pass  | ZH short +19.98%; ZH long +11.68%     | All six pass  | Keep retained  |
| TTS2   | All six pass  | EN long +103.9 ms; ZH short +111.9 ms | All six pass  | Keep retained  |

## Evidence and method

All measurements below come exclusively from the frozen run:

```text
.local/diagnostics/memory-release-20261007/frozen-benchmark/results.json
SHA256: 0b020b958b687d353065537cef9b7a204b9da688ed37b76393db42e5e2cbbecb
```

The result has `status=complete`, `qualification_valid=true`, and an empty
`source_changes` record. All eight workers completed. Independent post-run checks
matched all **17 recorded source hashes and 63 recorded artifact sizes/hashes**,
verified fixture files and saved waveform bytes, and reproduced per-case
median/p95 statistics. Audit evidence and patchable analysis scripts are preserved
alongside the run in `frozen-report-analysis.json`, `analyze-frozen-benchmark.py`,
and `render-frozen-report-tables.py`.

**Harness provenance limitation:** the supervisor rechecks sources, but records
artifact hashes only before measurement. Its validity flag alone is insufficient
to prove unchanged artifacts; the independent post-run artifact comparison above
supplies that check for this run. A previous run in the parent diagnostics directory
is excluded because source drift invalidated qualification, despite eight completed
workers. None of its measurements are used here.

- Intel Core i5-8500T, Linux 6.18.55, recreated project `.venv`, Python 3.14.7;
  OpenVINO 2026.3.1, NumPy 2.4.6, Torch 2.9.1+cpu.
- Native CPU STT2/TTS2, two inference threads, exact process/native-thread affinity
  to CPUs **4,5**, profiling disabled. Permanent artifact roots are `models/stt2`
  and `models/tts2`. Hoast's container was stopped at both recorded endpoints.
- Six fixed cases: English, Chinese and mixed-language, each short and long.
  Retained-mode seeded TTS generated shared FLOAT WAV fixtures once; STT workers
  consumed the same verified bytes. Text, WAV hashes and audio metadata are saved.
- Two rounds, reversing retained/release order in round two; fresh serial workers
  for each engine/mode/round. Each worker ran **two warmups plus eight measured
  calls per case**. This gives **16 measurements per case per mode**, 96 measured
  calls per engine/mode, and 480 worker calls including warmups across both engines.
- Wall latency covers the complete synchronous transcription/synthesis call,
  including preprocessing, owned output production, cleanup and subsequent
  reallocation. Initialization, file hashing, evidence serialization, playback and
  the 0.2-second idle intervals are excluded. Warmups are excluded from statistics.

An engine qualifies only if **every case** adds at most **10% AND 100 ms** to
its median, passes the **same two limits for p95**, and has exact output agreement.
p95 uses linear interpolation over the 16 observations. The tail screen is a
conservative small-sample check, not a statistical guarantee. Negative additions
are observed reductions; they do not establish that release intrinsically speeds
up inference. Gates use unrounded values; displayed deltas are rounded separately.

## Latency

### Median

| Engine | Case        | Retained ms | Release ms | Added ms | Added % | Gate |
| ------ | ----------- | ----------- | ---------- | -------- | ------- | ---- |
| STT    | en-short    | 164.0       | 158.9      | -5.1     | -3.10%  | Pass |
| STT    | en-long     | 289.9       | 275.8      | -14.1    | -4.87%  | Pass |
| STT    | zh-short    | 194.2       | 185.1      | -9.1     | -4.67%  | Pass |
| STT    | zh-long     | 336.2       | 320.7      | -15.5    | -4.61%  | Pass |
| STT    | mixed-short | 142.5       | 137.7      | -4.8     | -3.37%  | Pass |
| STT    | mixed-long  | 234.4       | 222.1      | -12.2    | -5.22%  | Pass |
| TTS    | en-short    | 1097.2      | 1104.2     | +7.1     | +0.64%  | Pass |
| TTS    | en-long     | 2057.7      | 2059.8     | +2.2     | +0.11%  | Pass |
| TTS    | zh-short    | 1324.1      | 1326.3     | +2.2     | +0.17%  | Pass |
| TTS    | zh-long     | 2442.9      | 2471.1     | +28.2    | +1.15%  | Pass |
| TTS    | mixed-short | 844.6       | 848.6      | +4.1     | +0.48%  | Pass |
| TTS    | mixed-long  | 1696.2      | 1692.5     | -3.7     | -0.22%  | Pass |

### p95

| Engine | Case        | Retained ms | Release ms | Added ms | Added % | Gate |
| ------ | ----------- | ----------- | ---------- | -------- | ------- | ---- |
| STT    | en-short    | 181.6       | 160.6      | -21.0    | -11.59% | Pass |
| STT    | en-long     | 305.2       | 290.1      | -15.1    | -4.94%  | Pass |
| STT    | zh-short    | 205.4       | 246.5      | +41.0    | +19.98% | Fail |
| STT    | zh-long     | 350.9       | 391.9      | +41.0    | +11.68% | Fail |
| STT    | mixed-short | 155.1       | 154.7      | -0.4     | -0.28%  | Pass |
| STT    | mixed-long  | 259.5       | 232.0      | -27.6    | -10.62% | Pass |
| TTS    | en-short    | 1108.7      | 1173.1     | +64.4    | +5.81%  | Pass |
| TTS    | en-long     | 2089.8      | 2193.7     | +103.9   | +4.97%  | Fail |
| TTS    | zh-short    | 1355.1      | 1466.9     | +111.9   | +8.26%  | Fail |
| TTS    | zh-long     | 2499.9      | 2583.5     | +83.6    | +3.34%  | Pass |
| TTS    | mixed-short | 931.0       | 885.1      | -45.9    | -4.93%  | Pass |
| TTS    | mixed-long  | 1730.3      | 1782.9     | +52.6    | +3.04%  | Pass |

Pooled engine statistics are descriptive only: mixing case lengths can hide a
failing case, so these passing pooled screens do not override the failures above.

| Engine | Ret. med ms | Rel. med ms | Median added ms / % | Ret. p95 ms | Rel. p95 ms | p95 added ms / % |
| ------ | ----------- | ----------- | ------------------- | ----------- | ----------- | ---------------- |
| STT    | 213.9       | 221.1       | +7.2 / +3.39%       | 338.8       | 325.3       | -13.5 / -3.98%   |
| TTS    | 1510.2      | 1597.1      | +87.0 / +5.76%      | 2455.1      | 2533.8      | +78.7 / +3.21%   |

## Idle and peak memory

Values are **MiB** ($2^{20}$ bytes). Idle values are medians of 16 snapshots per
engine/mode, taken after each complete six-case measured repetition and its idle
interval. They include resident weights and loaded resources; they are not
per-case memory measurements. RSS is resident memory, PSS proportionally accounts
for shared pages, and USS counts private pages. Saved means retained minus release;
a negative saving means release-mode workers used more memory.

| Engine | Metric | Retained MiB | Release MiB | Saved MiB | Saved % |
| ------ | ------ | ------------ | ----------- | --------- | ------- |
| STT    | RSS    | 2441.2       | 2440.7      | +0.5      | +0.02%  |
| STT    | PSS    | 2407.2       | 2423.6      | -16.4     | -0.68%  |
| STT    | USS    | 2387.0       | 2418.7      | -31.7     | -1.33%  |
| TTS    | RSS    | 3577.0       | 3579.7      | -2.7      | -0.08%  |
| TTS    | PSS    | 3557.6       | 3560.3      | -2.7      | -0.08%  |
| TTS    | USS    | 3552.0       | 3554.7      | -2.7      | -0.08%  |

Process lifetime peak RSS (`ru_maxrss`) includes initialization and warmups,
not just measured calls. These are individual-engine process peaks, not sampled
application/LLM peaks or combined resident-memory estimates.

| Engine | Mode     | Round 0 MiB | Round 1 MiB | Max MiB |
| ------ | -------- | ----------- | ----------- | ------- |
| STT    | retained | 2440.0      | 2442.2      | 2442.2  |
| STT    | release  | 2441.9      | 2444.0      | 2444.0  |
| TTS    | retained | 3581.2      | 3582.0      | 3582.0  |
| TTS    | release  | 3580.2      | 3599.4      | 3599.4  |

STT's 0.5 MiB RSS difference is negligible and is not corroborated by PSS/USS.
TTS shows a small increase across all idle metrics. Maximum observed peak RSS
also increases in release mode for both engines. Without allocator trimming or
forced garbage collection, freeing OpenVINO intermediate storage need not return
pages to the OS. These results provide no demonstrated memory benefit for enabling
per-call release on this setup.

## Output agreement and scope

For every case, all **40 calls** across both modes and rounds (including warmups)
returned the same STT canonical transcript and raw transcript, or the same exact
TTS float32 waveform bytes, sample count, 44,100-Hz rate and dtype. The six saved
waveforms were independently rehashed. Output agreement is **6/6 for each engine**.
STT transcripts contain recognition errors in both modes; this is a consistency
check on generated fixtures, not an accuracy or perceptual-quality evaluation.

Release calls retain compiled models, weights and inference requests. They run
under engine serialization only after borrowed outputs are consumed or copied.
A short cleanup call alone does not establish acceptable overhead: next-call
reallocation belongs in the complete-call latency comparison. Results cover warm
filesystem caches and CPU execution only, not GPU/ORT, cold-start latency,
playback/networking, or concurrent whole-application memory.

OpenVINO GenAI is a separate API. The installed `LLMPipeline` exposes chat lifecycle
methods but **no generic `release_memory()` method**; speech compiled-model cleanup
cannot be applied to it by analogy. Single-turn application history and
`finish_chat()` do not guarantee physical KV-cache or allocator-memory reclamation.
No LLM/KV memory reduction is measured or claimed by this benchmark.

## Reproduce without overwriting evidence

From the project root, use the recreated locked `.venv`, unchanged permanent
models and runtime sources, and a fresh timestamped directory outside `.cache`.
Stop Hoast and its automatic restarter before running; do not edit runtime/harness
sources, artifacts or the environment during measurement. The harness verifies
the container is stopped but does not stop it for you.

```sh
output="$PWD/.local/diagnostics/memory-release-$(date -u +%Y%m%dT%H%M%SZ)"
test ! -e "$output"
mkdir -p "$output"

export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE
export TOKENIZERS_PARALLELISM=false TORCHINDUCTOR_COMPILE_THREADS=1 MAX_JOBS=1
export PYTHONHASHSEED=7 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

taskset -c 4,5 .venv/bin/python -m tools.benchmark_memory_release --stt-root "$PWD/models/stt2" --tts-root "$PWD/models/tts2" --output "$output" --seed 7 --warmups 2 --repeats 8 --rounds 2 --idle-seconds 0.2 --timeout 1800 > "$output/supervisor.log" 2>&1
```

Preserve the complete logs, fixtures, raw worker records and results. Require
`status=complete`, `qualification_valid=true`, no source changes, and an independent
post-run artifact-hash comparison before considering enablement. A changed device,
model, plugin or latency policy requires a fresh qualification; do not infer a
passing result from a pooled median or from an invalidated run.
