# Kokoro GPU convolution and zero-copy experiments

This report preserves the analysis and kernel-development journey. The supporting
benchmark/profiling drivers and rejected kernels have been retired. The maintained
runtime is `hoast/tts_gpu.py` with `hoast/kernels/conv_window.cl`, integrated into
the public TTS API. See [Local speech](speech.md) for current preparation and the
compact `tools.check_speech` command. Names of retired tools below identify the
historical measurements, not supported commands.

## Scope and measurement contract

Target hardware is the i5-8500T and Intel UHD 630 (24 EUs, maximum 1100 MHz),
OpenVINO 2026.3.1 and Intel OpenCL driver 24.35.30872.36. Experiments run serially
with two-core affinity, a 4 GiB process-tree limit, no swap and an available-memory
watchdog. Production speech settings are independent of these experiment drivers.

`tools.kokoro_gpu_probe` times phonemization, CPU frontend, decoder, transfers,
allocation/synchronization and the final waveform copy. Model/kernel preparation
is reported separately. Four utterances produce 111, 181, 241 and 467 ASR frames,
or 2.775, 4.525, 6.025 and 11.675 seconds of audio. Each input gets a first call
and two repeats. Different inputs encounter different lengths in the same loaded
engine. “First” is not a warmed repeat; filesystem and driver caches can already
exist. CPU time sums this process's threads, rather than measuring machine-wide
utilization. Full GPU profiles may contain cumulative counters.

## Validated result

The validated study path is **CPU frontend and surrounding decoder operations,
with zero-copy INT8-weight GPU convolution**. It uses the cooperative-window kernel,
SIMD16, and the largest of 32/16/8 temporal tiles that compiles without spilling
for each actual FP32-I/O layer geometry. Tile selection happens during preparation,
not when a new text length arrives.

These measurements used the study's split native English exports. The maintained
implementation applies the same decoder-only offload to complete published English
ONNX and Chinese v1.1 IR models, avoiding a dependency on split-export scaffolding.
Post-cleanup verification covers kernel numerics and public English/Chinese round
trips; it does not relabel the study timings as a new performance measurement.

Matched Python 3.12/OpenVINO 2026.3.1 runs, using identical input hashes:

| Audio duration | CPU first | Hybrid first | CPU repeated median | Hybrid repeated median | CPU CPU-time | Hybrid CPU-time |
| -------------- | --------- | ------------ | ------------------- | ---------------------- | ------------ | --------------- |
| 2.775 s        | 1.551 s   | 1.228 s      | 1.324 s             | 1.041 s                | 2.538 s      | 1.454 s         |
| 4.525 s        | 2.327 s   | 1.689 s      | 2.284 s             | 1.686 s                | 4.394 s      | 2.329 s         |
| 6.025 s        | 2.960 s   | 2.322 s      | 2.812 s             | 2.272 s                | 5.424 s      | 3.112 s         |
| 11.675 s       | 5.691 s   | 4.526 s      | 5.518 s             | 4.361 s                | 10.601 s     | 5.583 s         |

CPU-time columns are medians of summed process-thread CPU seconds, not wall time.
The weather case improves repeated latency by **26%** and CPU time by **47%**.
Across this set, repeated latency improves about 19–26%, and CPU time about 43–47%.
First calls include new-length handling, with no 10–45-second GPU specialization
stall. These timings end at the host waveform; they exclude WAV writing, STT,
speaker playback, and concurrent LFM inference.

All four resulting transcripts match the CPU reference under full-context Whisper
recognition. One-core execution takes 2.169 s for the weather case versus 1.686 s
with two, so two cores are retained. FP16-stored weights take 1.720 s in the same
shared-memory implementation; the difference is small, and INT8 remains the
selected experimental storage format. Setup time depends on binary-cache state;
both model loading and geometry-specific kernel compilation remain separate costs.

Guarded peak process-tree RSS was approximately 1.49 GiB for the INT8 experiment,
1.55 GiB for the FP16-weight control and 1.31 GiB for the CPU reference. Zero-copy
sharing eliminates activation transfer copies; it does **not** establish a lower
overall resident-memory footprint than CPU-only inference.

INT8 means signed, symmetric, per-output-channel weight storage. Activations use
FP16 arithmetic, with FP32 accumulation of bounded FP16 partial sums. The shared
CPU graph keeps its activation buffers in FP32 and converts values in GPU registers.
No complete expanded FP16 weight array is created on the GPU by the custom kernels.

## Ideal-convolution feasibility

The GPU custom-layer XML ablation retains the convolution graph edges and output
shapes but fills outputs with zeros. Its audio is **invalid**. This estimates
remaining overhead; it is not a valid implementation or an exact timing predictor,
because fusion/layouts and activation values differ.

In the earlier feasibility measurements, the weather case uses 181 frontend frames
and CPU synthesis takes about 2.20 s warm.
The stock CPU-frontend/GPU-decoder path takes about 1.93 s warm, but its first
encounter with that length takes 9.82 s. Other new lengths take up to 45.63 s.
The XML zero-convolution ablation takes about 0.55 s warm but still about 23 s
on each new length. XML custom operations support dynamic shape inference, but
the GPU plugin emits shape-specific JIT definitions and recompiles implementations.
An efficient kernel alone does not fix this integration path.

## Zero-copy CPU graph with GPU convolution

`hoast.tts_gpu.CPUConvOffload` runs as a CPU reference operation and dispatches
its convolution to a persistent raw OpenCL context. The surrounding graph stays
on CPU, including the verified fused Snake activation. It replaces 67 decoder
convolutions; two tiny pitch/noise filters stay native. All geometry-specific
programs are prepared before timing, with lengths supplied as runtime arguments.

CPU activation pages are imported using `CL_MEM_USE_HOST_PTR`. Imports use
page-aligned bases and cache-line-sized extents; kernel offsets identify only the
actual tensor region. Kernels must not access neighboring allocator bytes covered
by the imported pages. The executor is synchronous and requires live contiguous
CPU tensors and serialized graph execution.

If expanded input/output page ranges overlap, a single OpenCL buffer is used with
two logical offsets. Logical in-place convolution is rejected. Cleanup is
exception-safe and drains pending work before releasing mappings and bindings.

This is verified beyond pointer equality: Intel's diagnostic callback explicitly
reports that the imports share physical memory with the CPU and that output
map/unmap requires no data copy. GPU completion and output mapping occur before
CPU consumers resume. Kernel input/output bindings are cleared before imported
buffers are released, so cached kernels cannot retain pages after CPU arena reuse.
Immutable layer attributes prevent graph optimization from merging operations
with different weights.

Every successful import requires the driver's physical-sharing confirmation;
every output map requires its copy-free confirmation and the original CPU address.
Evidence is in `shared-test/results.json` and each completed variant's
`driver-diagnostics.json`. The messages explicitly say “buffer will share the same
physical memory with CPU” and “will not require any data copy.” Operation identity
includes immutable weight/scale, geometry, precision and kernel fingerprints,
independent of whether model nodes happen to have equal friendly names.

The corresponding zero-arithmetic ablation handles new lengths in roughly
0.63, 0.80, 1.11 and 1.98 s, without the GPU-plugin specialization stalls.
This demonstrates a useful integration path, not valid synthesized speech.

## Roofline methodology

`tools.gpu_roofline` measures a 256 MiB resident-buffer copy, counting both reads
and writes, and independent FP16 vector FMA chains. Device event timings exclude
buffer allocation and compilation. The selected microbenchmark measured
**25.21 GB/s** and **623.62 GFLOP/s**. These are empirical reference ceilings,
not simultaneous sustainable rates for every convolution. The nominal maximum
FP16 rate is 844.8 GFLOP/s; EU utilization is not inferred from these roofline rates.

`tools.kokoro_conv_inventory` computes conventional direct-convolution FLOPs and
an optimistic per-layer traffic estimate: one activation read, one output write,
one weight read, and scales. It assumes ideal reuse, excludes extra cache traffic,
reorders and dequantization work, and is summed across layers rather than claiming
that it measures resident memory.

For the weather case, the 69 ordinary convolutions total **230.20 GFLOPs**.
The compute-only empirical reference time is about **0.369 s**. Minimum W8A16 traffic
is 429.08 MB versus 474.42 MB for W16A16: weight compression saves only about
9.6% of that traffic. Most expensive layers are compute-bound under ideal reuse;
shrinking weights alone is not enough. An approximate perfect-kernel projection
for the shared CPU graph is 0.80 + 0.37 ≈ 1.17 s, with some double-counted dummy
output writes. This is a heuristic projection, not an achieved result or formal
bound: altered values, fusion, CPU/GPU contention and scheduling can change costs.

The shared executor actually has FP32 activation I/O. Its 67 offloaded convolutions
have approximately **813 MB** of minimum W8/FP32-I/O traffic in this case; excluding
the two tiny CPU filters barely changes the FLOP count. The mixed-precision
accumulation, conversion and shuffle work also make the pure-FP16 FMA reference
an optimistic ceiling rather than a promised convolution rate.

## What limited utilization, and what improved it

The initial shared-memory-tiled kernel achieved roughly 28–46 GFLOP/s, followed
by register-tiled and channel-vectorized variants at roughly 40–100 GFLOP/s for
INT8 weights. Actual `ocloc` disassembly showed SIMD8 FP16 MAD instructions and
many scalar uniform weight-conversion/address instructions. Increasing work per
item alone did not make efficient use of the FP16 hardware.

The cooperative-window design assigns output channels to SIMD16 lanes. It loads
each temporal input window once per input channel, reuses overlapping taps through
register shuffles, packs weights for adjacent-lane access, and prefetches taps.
A 32-position tile reached about **310 GFLOP/s** on the hot 128×128, 11-tap layer,
around half of the empirical pure-FP16 arithmetic ceiling. Operator checks retain
the FP32 reference comparison; maximum relative RMS versus outputs computed with
the original weights was
about 1.4% for INT8, with arithmetic-only error much smaller.

The register-capacity limit is real: a 64-position tile spilled **3296 bytes per
hardware thread** and fell to about **34 GFLOP/s**. The standard OpenCL private-memory
query still reported zero; the Intel-specific spill query detected the problem.
Some 32-position variants also spill, depending on geometry, weight type and I/O
type. For example, the raw FP16-I/O INT8 hot-layer variant has 448 bytes of spills.
The integrated backend queries its actual FP32-I/O kernels and chooses a smaller
tile when needed. Kernel resource figures from one I/O variant are not assumed
to describe another.

Intel MDAPI counter collection succeeded using the driver-matched Gen9 stack:

- EU thread occupancy was about **98–99%**: lack of resident threads was not the
  main problem in the sampled hot layer.
- The earlier channel kernel spent about **94%** of time EU-active despite low
  useful FLOP throughput, consistent with its inefficient instruction mix.
- The improved window kernel showed about **76–79% EU-active** and **21–24% stalled**
  time. Remaining instruction and cache-access costs matter; the arithmetic peak
  alone cannot predict its speed.
- Shader-to-L3 traffic was much larger than external-memory traffic. The measured
  window32 external-memory interface rate was approximately **2.1–2.3 GB/s**, well
  below the **25.2 GB/s** streaming reference. This does not support a DRAM-bandwidth
  bottleneck. GTI is the GPU's external-memory interface, not a direct measurement
  of all system DRAM traffic.

The driver reports unavailable context-switch filtering for these counter queries.
The collector therefore records that limitation and verifies expected compute-thread
counts, zero graphics-thread activity, and absence of error/inconsistent/context-
mismatch reports. These are isolated query-interval observations, not guaranteed
per-process counters. Profiling adds overhead; headline timings use uninstrumented
runs. Raw data is under `mdapi-conv_channel`, `mdapi-conv_window` and `mdapi-window32`.

A post-load CPU `perf` profile found about **85%** of sampled cycles in remaining
CPU JIT/OpenVINO computation, math libraries and the fused activation, versus about
4.5% in the OpenCL driver. Further convolution speedups increasingly encounter the
CPU portion of the pipeline. The profile is retained under `host-profile/`.

## Research references

- [OpenVINO Gen9-style osv16 convolution](https://github.com/openvinotoolkit/openvino/blob/2026.3.1/src/plugins/intel_gpu/src/kernel_selector/cl_kernels/convolution_gpu_bfyx_os_iyx_osv16.cl): cooperative register windows, subgroup shuffles and tap prefetch.
- [Its kernel selector](https://github.com/openvinotoolkit/openvino/blob/2026.3.1/src/plugins/intel_gpu/src/kernel_selector/kernels/convolution/convolution_kernel_bfyx_os_iyx_osv16.cpp): tile/prefetch selection and unrolling tradeoffs.
- [oneDNN Gen9 GEMM](https://github.com/oneapi-src/oneDNN/blob/v2.7/src/gpu/ocl/gemm/gen9_gemm_compute.cl): packed operands, subgroup communication and independent accumulators.
- [Nugteren's OpenCL register-blocking tutorial](https://cnugteren.github.io/tutorial/pages/page8.html): register reuse and compiler-hoisting pressure. Its Kepler hardware limits are not applied to Gen9.
- [OpenCL Intercept Layer MDAPI guide](https://github.com/intel/opencl-intercept-layer/blob/main/docs/mdapi.md) and [ISA guide](https://github.com/intel/opencl-intercept-layer/blob/main/docs/kernel_isa_gpu.md).
- [Driver-matched dependency manifest](https://github.com/intel/compute-runtime/blob/24.35.30872.36/manifests/manifest.yml): IGC 1.0.17537.24, Metrics Discovery 1.13.176 and Metrics Library 1.0.173.

## Reproduction and evidence

The study's prepared models were `.cache/hoast/tts/kokoro-hybrid/{frontend,decoder}.xml` and
`.cache/hoast/tts/kokoro-current/kokoro-fp16.xml`, with matching binaries and voice
tables. The study used an isolated Python 3.12 export environment, with PyOpenCL
2026.1.4, pytools 2026.1.1, platformdirs 4.11.8 and siphash24 1.9.

Historical invocation (the benchmark driver is retired):

```sh
uv run python -m tools.guarded_run --threads 2 --memory-gib 4 --timeout 600 --log-file .cache/hoast/diagnostics/shared-probe.log -- env PYTHONPATH=. .cache/hoast/experiments/kokoro-export-env/.venv/bin/python -m tools.kokoro_gpu_probe --mode shared --output .cache/hoast/diagnostics/hybrid-conv/reproduction
```

The controls used `--mode cpu`, `stock`, `ideal`, `shared-ideal`, or `shared-fp16`.
The `ideal` variants did not publish WAV files. GPU microbenchmarks, exact geometries,
raw timings, numerical errors and native driver diagnostics are retained under
`.cache/hoast/diagnostics/hybrid-conv/`. Full failures are retained before retries.
The retired `tools.summarize_gpu_probe` derived the compact JSON summaries without
rerunning models. The retired `tools.verify_gpu_speech` compared retained audio
against full-context CPU-reference transcripts; that did not replace listening
evaluation. Current checks use `tools.check_speech`, documented in [Local speech](speech.md).

Validation completed with **673 offline tests**, Ruff and Pyright. The C++ operation
passes clang-format and clangd checks; the active OpenCL kernel compiles with the
installed Intel `ocloc` for Coffee Lake and passed real-device numerical tests.
Generic Clang does not declare the Intel-specific shuffle intrinsic in its default
OpenCL headers, so the Intel compiler is used for that check.

The user enabled performance counters for this investigation. To restore the
original settings after profiling:

```sh
sudo sysctl -w dev.i915.perf_stream_paranoid=1 kernel.perf_event_paranoid=2
```
