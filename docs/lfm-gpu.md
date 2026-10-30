# Experimental standalone LFM OpenCL runner

- `hoast.lfm_gpu` is a batch-one, unpadded LFM2.5-350M OpenCL implementation. `GPUGeneration` adds generation, optional raw streaming callbacks and tool-call validation, but does not dispatch tools or store conversations.
- The application selects a live OpenCL adapter for LFM; it requires exactly one OpenCL GPU and does not fall back to CPU. This integration requires end-to-end correctness, routing and cancellation qualification; do not treat the experimental measurements below as proof of deployment readiness. OpenVINO model preparation and the separate library backend are described in [llm.md](llm.md#lfm25).

## Setup

- Run `uv sync --locked` from the project root. Use an in-order OpenCL GPU queue with `cl_khr_fp16` **and FP16 denormals**, adequate allocation limits and available memory. Construction compiles the bundled kernels; allocation can still fail.
- Supply a checkpoint containing `config.json`, `openvino_model.xml`, `openvino_model.bin`, and, for generation, the matching fast tokenizer and chat template. See [LFM preparation](llm.md#lfm25).
- Only the exact XML/BIN pairs in `ARTIFACT_DIGESTS` in `hoast/lfm_gpu/runner.py` are accepted; mixed pairs and other revisions fail. Local fingerprints identify inspected files, not independently authenticated provenance. Do not bypass the allowlist.

## Runner and generation

- Construct `Runner(queue, checkpoint_path, WorkspaceSpec(capacity, chunk_size))`; capacity includes prompt and output and is at most 128000. Inspect `WorkspaceSpec.buffer_sizes` before allocating: scores scale with chunk size and capacity, KV state with capacity, and model weights are additional.
- Call `prefill()` with nonempty token IDs on empty state, then `decode()` for each continuation; each returns independent FP32 final-position logits. IDs are Python integers in `[0, 65536)`. Reset before a new sequence; after execution or reset failure, successfully reset before reuse. Serialize other uses of the caller-owned queue too.
- Optional prepared prefill, KN, tiled attention and vec4 matvec tuning trade memory and device-specific performance. Persistent KN requires prepared KN and a budget of at least 574,619,648 bytes, retaining original quantized weights. It is not a guaranteed speedup. Attention materializes FP32 scores; this is not FlashAttention or all-FP16 inference.
- `prefix_cache=True` allows `cache_prefix(ids)` and, after reset, `prefill_cached(full_prompt_ids)`. The **entire token prefix must match exactly**; mismatch raises. `prefix_slots=2` allocates two independent snapshots, selected by slot. Reset preserves the snapshots; invalidation or device/copy/reset failure discards them. Each snapshot allocates KV and convolution state for the full workspace capacity, not just the prefix length.
- `GPUGeneration` needs exclusive runner and queue ownership for the whole request, the original tokenizer/template, and a `ToolRegistry`. It uses greedy generation with a repetition penalty and rejects invalid logits, malformed calls, and output reaching its token limit. It does not sample; raw streaming callbacks precede validation and require protocol-safe handling. In strict prefix mode, initialize the caller-chosen stable prefix or two prefixes explicitly; prompts that do not match either fail. Template, tools, prompt or tokenization changes can invalidate the assumption. A runner or rollback failure faults the adapter; construct a new runner and adapter.
- The live adapter builds two strict prefixes at load, one for observed music on and one for observed music off. Each covers the complete rendered system message and tool declarations; requests select a snapshot by exact token-prefix match. Unavailable, unconfigured or player-required music is represented as observed off, not as an observation that playback is actually off. Missing or invalid observed state and unknown statuses fail. Building both snapshots adds startup prefill cost. At the default 4096-token capacity, the second full-capacity snapshot adds approximately 96 MiB of device memory compared with one snapshot; this scales with capacity.
- GPU arithmetic is **not bit-identical** to OpenVINO. Rounding, reductions, chunk boundaries and device differences can alter logits and greedy tokens. Short-input parity tests are not a model-wide or action-safety guarantee.

## Tests and evaluation

```sh
uv run pytest -q tests/test_lfm_gpu.py tests/test_lfm_gpu_generation.py \
  tests/test_lfm_gpu_weights.py tests/test_lfm_gpu_reference.py \
  tests/test_lfm_gpu_kernels.py tests/test_lfm_gpu_ops.py
```

Kernel tests use the first available GPU and skip if unavailable. Real-device tests are opt-in; select a unique case-insensitive GPU name substring and a supported checkpoint:

```sh
export LFM_GPU_CHECKPOINT=/absolute/path/to/checkpoint
export LFM_GPU_DEVICE='unique GPU name substring'
LFM_GPU_RUNNER_TEST=1 LFM_GPU_PREPARED_TEST=1 LFM_GPU_PREFIX_TEST=1 \
  uv run pytest -q tests/test_lfm_gpu.py
LFM_GPU_OPS_TEST=1 LFM_GPU_OPS_DEVICE="$LFM_GPU_DEVICE" \
  uv run pytest -q tests/test_lfm_gpu_ops.py
```

Check skip counts: a skipped test is not device validation. Compare the same IR, tokens, state and precision; inspect logits, top tokens and margins. `tools/lfm_gpu_reference.py` is a numerical investigation tool, not a performance benchmark.

For non-dispatching routing evaluation, obtain the exact corpus pinned by [the manifest](lfm-routing-benchmark-manifest.json), set `CORPUS`, reserve the GPU, and use a fresh output path:

```sh
uv run python -m tools.evaluate_lfm_gpu --corpus "$CORPUS" \
  --manifest docs/lfm-routing-benchmark-manifest.json --split dev \
  --backend custom --checkpoint "$LFM_GPU_CHECKPOINT" --device "$LFM_GPU_DEVICE" \
  --gpu-coordinated --chunk-size 930 --prepared-kn --persistent-kn \
  --persistent-kn-budget-bytes 574619648 --vec4-matvec --tiled-attention \
  --prefix-cache --output dev-python.jsonl
```

For the frozen test selection, replace `--split dev` with `--split test --test-selection docs/lfm-routing-test-selection.json`. The evaluator verifies corpus and schema hashes and does not execute tools; see `--help` for provenance requirements. `tools/benchmark_lfm_phases.py` measures teacher-forced phases, while `tools/benchmark_lfm_gpu.py` compares three captured weather requests; consult their `--help`, isolate GPU use, and record device, artifacts, workload and cache state.

## Key experimental results

- Before dual-prefix caching, a non-dispatching UHD 630 live-prompt smoke test measured 9.58 s for a warm weather tool call; its short fixed prefix did not reproduce the 923-token benchmark below. This historical measurement does not measure the two-snapshot live adapter. In the same smoke test, “hello” incorrectly produced a light-on call. The live backend can dispatch unrequested actions; tool schema validation does not establish user intent.
- On UHD 630 with the deployed checkpoint, a serial non-dispatching synthetic 86-case test gave Python OpenCL GPU **68/86**, OpenVINO CPU **68/86**, and OpenVINO GPU **67/86** ordered exact matches; each had six invalid outputs. Median case walls were 4.252 s, 9.048 s and 3.213 s respectively, with different backend and cache configurations. Python and CPU raw responses matched; GPU differed twice, including an incorrect pause instead of resume. Each had one counted false activation; four of eight no-tool cases contained parsed call attempts, three invalid. All six ASR-noisy cases failed; there were no multi-action targets. The selection had been used previously, and GPU exclusivity was not guaranteed. This synthetic sample does **not** establish production accuracy or action safety.
- A fixed-prefix weather experiment used `WorkspaceSpec(1152, 128, prepared_prefill=True, prepared_kn=True, vec4_matvec=True, tiled_attention=True, prefix_cache=True)` and strict prefix caching, reserving 128 output tokens. The exact 923-token prefix took 5.488 s to build. Across three requests and three measured rounds, median full-request walls were 0.769 s custom and 3.426 s OpenVINO GenAI GPU; all 12 aligned warmup and measured outputs and validated calls matched. This is a tiny, fixed-order, non-dispatching comparison with different cache conditions and engines, not a service benchmark. Its 923-token prefix is not the live adapter's state-specific prefix. A prefix miss cannot use strict mode; longer prompts or changed declarations require a revised configuration and verified prefix.
- In a separate teacher-forced 930-token phase comparison, median prefill was 5.571 s custom versus 3.066 s raw OpenVINO GPU; decode medians were 40.72 and 35.72 ms. Output shapes and precision differ, raw IR is not GenAI, and only three sequences were measured. The custom path did not meet the raw GPU prefill time on that fixture.
- A C++ scheduler experiment was removed: measured benefits were modest and did not justify a second implementation. Its historical parity and routing results do not validate the current service; its implementation and tests are not retained. Exploratory fusion did not demonstrate a whole-runner speedup.

Local raw benchmark records under `.cache/hoast/benchmarks/` and `/tmp/ttc/1000/` may not persist. Measurements are hardware- and configuration-specific; avoid treating them as guarantees.
