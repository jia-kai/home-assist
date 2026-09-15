# Kokoro native OpenVINO export: FP16 and INT8 weights

This report preserves the export investigation and its historical commands and
measurements. Its experiment scripts and alternative serving backends have been
retired. See [Local speech](speech.md) for the maintained hybrid runtime and
compact correctness check, and [the hybrid study](kokoro-hybrid-conv.md) for the
subsequent kernel and zero-copy analysis.

## Round-trip correctness

The short and long test utterances passed TTS-to-STT word comparison for the
upstream PyTorch reference, native OpenVINO CPU, GPU FP16, and GPU FP16 using
INT8-stored weights. Comparison normalizes punctuation and number formatting:
“twenty degrees” and “20 degrees” are considered equivalent.

The FP16 runs preserved every predicted token duration in these cases. INT8
weight compression changed one duration in the long case, shortening it by
25 ms, while preserving the recognized words. Generated arrays were nonempty
and finite. These checks establish transcript agreement, not identical waveforms
or subjective listening quality.

## Source checkpoint and toolchain

The source is the official [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
checkpoint at revision `f3ff3571791e39611d31c381e3a41a3af07b4987`:

- `kokoro-v1_0.pth`
- `config.json`
- `voices/af_heart.pt`

The `.pth` SHA256 is
`496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4`.
Source hashes, export hashes, package versions, RNG state, reference inputs and
reference audio are retained with the artifacts.

Latest stable releases were checked against PyPI on September 13, 2026. The
final export and benchmarks use an isolated Python 3.12 environment because the
upstream Kokoro package requires Python below 3.13.

| Library                    | Version       |
| -------------------------- | ------------- |
| PyTorch CPU                | 2.14.0+cpu    |
| OpenVINO                   | 2026.3.1      |
| OpenVINO GenAI / tokenizers | 2026.3.1.0    |
| Transformers               | 5.17.0        |
| NNCF                       | 3.3.0         |
| ONNX                       | 1.22.0        |
| ONNX Runtime               | 1.30.0        |
| CTranslate2                | 4.8.2         |
| faster-whisper             | 1.2.1         |
| Kokoro                     | 0.9.4         |
| kokoro-onnx                | 0.6.1         |
| Hugging Face Hub           | 1.31.0        |
| NumPy                      | 2.4.6         |
| SciPy                      | 1.18.1        |

NumPy 2.4.6 is the newest compatible version under NNCF 3.3's `<2.5` constraint;
the standalone latest NumPy release is 2.5.3. Direct `ov.convert_model` conversion
permits Transformers 5.17; the current Optimum Intel release requires
Transformers below 5.6 and is not an export dependency here.

## Why the published ONNX FP16 path failed

The UHD 630 supports real FP16 convolution. An isolated convolution selected
`convolution_gpu_bfyx_os_iyx_osv32__f16` and took about 5.38 ms versus 9.46 ms
for FP32, including request/output overhead.

The failure was isolated to a **dynamic FP16 matrix multiplication** in the
nine-to-one harmonic mixer:

- Data shape `[1, dynamic_length, 9]`, weight shape `[9, 1]`, `transpose_b=False`
  fails with the same `9` versus `1` matrix-dimension error as the full model.
- FP32 and fixed-length FP16 versions work.
- Keeping weights as `[1, 9]` with `transpose_b=True` works in dynamic FP16.
- Equivalent reduction and 1×1 convolution representations also work.

The published ONNX path selects `fully_connected_gpu_bfyx_ref__f16` for this
layer. The direct PyTorch export selects the working tiled FP16 fully-connected
path. It also avoids the published ONNX path's interpolation compatibility
rewrite. Thus this is an export-representation/kernel-path problem, not evidence
that FP16 convolutions are unavailable.

Direct exports made with both PyTorch 2.9.1 and 2.14.0 worked. The result therefore
does not establish that a new PyTorch bug fix alone resolved the issue. The final
artifacts use 2.14.0 and the current library versions above.

## Native export and execution

The exporter wraps upstream `KModel.forward_with_tokens` with tensor inputs and
calls `ov.convert_model`. Inputs are `input_ids` (int64, `[1, tokens]`), `ref_s`
(float32, `[1, 256]`) and `speed` (float32, `[1]`). Outputs are mono audio and
per-token durations. Two different token lengths are validated, preventing an
example-length-only trace from passing unnoticed.

`ov.save_model(..., compress_to_fp16=True)` produces FP16-weight IR. Weight
storage and execution precision are separate: GPU tests also explicitly request
`INFERENCE_PRECISION_HINT=f16`. Actual runtime profiles show FP16 convolution
kernels; selected precision-sensitive operations remain FP32.

NNCF `compress_weights(..., mode=INT8_ASYM)` produces the weight-only variant.
Activations are not quantized to INT8. Storage is mixed: approximately 70.4 MB of
uint8 constants and 43.9 MB of retained float32 constants, plus small metadata
and scale tensors. This differs from the activation-quantized ONNX export tested
in [speech.md](speech.md).

### Measured synthesis latency

Median of three warm inference calls, using identical saved token/style/speed
inputs. Phonemization, model loading and STT are excluded from these times.
The short and long reference audio durations are 4.525 and 13.275 seconds.
CPU uses two threads and one stream; GPU uses one stream and explicit GPU.0
execution, without automatic CPU fallback.

| Stored weights   | Execution | Short TTS | Long TTS |
| ---------------- | --------- | --------- | -------- |
| FP16             | CPU FP32  | 2.345 s   | 6.926 s  |
| FP32             | GPU FP16  | 2.774 s   | 8.264 s  |
| FP16             | GPU FP16  | 2.734 s   | 8.165 s  |
| INT8 weight-only | GPU FP16  | 2.796 s   | 8.406 s  |

GPU first-use and new-length specialization remain expensive: first inference
took approximately 11–57 seconds across these configurations and lengths.
Warm figures exclude those costs. FP16 GPU execution works, but CPU remains
faster on this machine for the tested utterances.

### What limits the iGPU?

A fixed-length steady-state profile separates the recurrent execution penalty
from convolution throughput. Four warmups precede three measured calls. OpenVINO
reports cumulative average counters, so the warm-window estimate is
`(7 * average_after - 4 * average_before) / 3`, with no kernel changes across the
window. Summed counters agree with measured time to within about 2% on GPU and
0.2% on CPU. Raw counters spanning warmup or different lengths must not be read
as a single warm call's breakdown.

For the 4.525-second short utterance:

| Attributed work                    | CPU FP32, 2 threads | GPU FP16 |
| ---------------------------------- | ------------------- | -------- |
| Convolution                        | 1.415 s             | 1.357 s  |
| LSTMSequence / TensorIterator      | 0.031 s             | 0.927 s  |
| Other work and request overhead    | 0.905 s             | 0.503 s  |
| Measured mean total                | 2.351 s             | 2.787 s  |

Backend fusion differs, so these are operator-group attributions. Convolutions
are the largest aggregate GPU cost, about half the request. They are slightly
faster than the CPU convolution group. The decisive regression is recurrent
execution: the GPU TensorIterator path takes about 0.9 seconds more than the
CPU LSTMSequence path. The largest individual profiled GPU nodes are these
recurrent iterators.

The GPU plugin's loop implementation executes the body network per timestep,
manages loop-carried dependencies and waits for events. Its profiling label
`CPU` describes this host-side loop implementation; it does not establish that
all recurrent tensor arithmetic ran on CPU. This is expensive for batch-one,
sequential recurrent work compared with the CPU's fused LSTMSequence path.

The GPU does have a higher arithmetic ceiling. OpenVINO reports 24 execution
units and 844.8 GOP/s for FP16, versus 422.4 GOP/s for FP32. The system's GPU
maximum clock is 1100 MHz. Two CPU cores have an optimistic AVX2 FP32 ceiling of
roughly 134–224 GFLOP/s at 2.1–3.5 GHz (two 256-bit FMAs per core per cycle).
Peak arithmetic therefore favors GPU FP16 by roughly four to six times.

That ceiling excludes kernel dispatch, dependent recurrent steps, memory traffic,
layout changes, address calculations and subgroup shuffles. The selected GPU
convolution kernel contains input caching, subgroup shuffles and dilation/stride
handling around its half-precision multiply-adds. Batch-one 1D/dilated convolution
and recurrent workloads do not behave like one large dense GEMM. Hardware EU
utilization and bandwidth counters were not measured, so the precise hardware
limit behind the convolution group's modest gain remains unproven. The measured
whole-model deficit is primarily the recurrent/TensorIterator path.

Implementation references for the tested OpenVINO revision:
[GPU loop](https://github.com/openvinotoolkit/openvino/blob/759c5a6ab8c/src/plugins/intel_gpu/src/graph/impls/common/loop.cpp)
and [FP16 convolution](https://github.com/openvinotoolkit/openvino/blob/759c5a6ab8c/src/plugins/intel_gpu/src/kernel_selector/cl_kernels/convolution_gpu_bfyx_f16.cl).
Warm-window counters are in `steady-CPU.json`, `steady-GPU.json` and
`steady-profile.log` under the diagnostic directory; reproduction script:
`/tmp/opencode/profile_kokoro_steady.py`.

## Does INT8-to-FP16 conversion reduce execution memory?

It substantially reduces the model file, but provided little GPU-allocation
saving and increased observed process RSS in this test.

| Stored weights   | Weight file | Compiled GPU allocations | Warm GPU allocations | Warm process RSS |
| ---------------- | ----------- | ------------------------ | -------------------- | ---------------- |
| FP32             | 309.5 MiB   | 231.4 MiB                | 954.8 MiB            | 1129.9 MiB       |
| FP16             | 181.2 MiB   | 231.3 MiB                | 954.6 MiB            | 1173.5 MiB       |
| INT8 weight-only | 108.7 MiB   | 225.3 MiB                | 947.7 MiB            | 1262.9 MiB       |

Each configuration runs in a fresh process. GPU values are the sum of reported
OpenVINO `GPU_MEMORY_STATISTICS` allocation categories. Warm values follow both
test lengths, before loading STT. Process RSS and GPU allocation counters describe
different, potentially overlapping memory views on an integrated GPU; do not add
them together as total physical memory.

Relative to FP16 weights, INT8 saves **40% of file size**, about **2.6% of compiled
GPU allocations**, and only **0.7% of warm GPU allocations**. The slightly shorter
INT8 long waveform can also affect warm activation-buffer allocation. Resident
process memory did not improve in the recorded samples.

Both variants still use ordinary FP16 convolution kernel families. OpenVINO's
runtime graph hides internal constant-weight inputs, so it does not establish
that every convolution performs fused on-the-fly INT8-to-FP16 decompression.
The allocation measurements do not support treating this as a broadly effective
resident-memory optimization for Kokoro. INT8 storage is useful for smaller
artifacts; FP16 storage is the better GPU tradeoff in these measurements.

## Artifacts and reproduction

Final artifacts are in `.cache/hoast/tts/kokoro-current/`:

- `kokoro-fp32.xml` / `.bin`
- `kokoro-fp16.xml` / `.bin`
- `kokoro-int8-weights.xml` / `.bin`
- `manifest.json`, `int8-weight-manifest.json`, `export-sha256.json`
- Reference `case-*.npz`, `torch-*.wav`, and `export-rng-state.pt`

The isolated environment specification and lockfile are under
`/tmp/opencode/kokoro-export-env/`. From the repository root, the retained scripts
can be rerun with that environment's Python:

```sh
PYTHONPATH=. /tmp/opencode/kokoro-export-env/.venv/bin/python /tmp/opencode/export_kokoro_torch.py --output .cache/hoast/tts/kokoro-current
PYTHONPATH=. /tmp/opencode/kokoro-export-env/.venv/bin/python /tmp/opencode/compress_kokoro_weights.py --source .cache/hoast/tts/kokoro-current/kokoro-fp32.xml
PYTHONPATH=. /tmp/opencode/kokoro-export-env/.venv/bin/python /tmp/opencode/test_kokoro_torch_ir.py --model .cache/hoast/tts/kokoro-current/kokoro-fp16.xml --device GPU --hint f16
PYTHONPATH=. /tmp/opencode/kokoro-export-env/.venv/bin/python /tmp/opencode/test_kokoro_torch_ir.py --model .cache/hoast/tts/kokoro-current/kokoro-int8-weights.xml --device GPU --hint f16
```

The application CPU CLI accepts native IR input names and preserves fractional
speech speed:

```sh
uv run python -m hoast.tts "The weather is sunny." --model .cache/hoast/tts/kokoro-current/kokoro-fp16.xml --speed 1.25 --output /tmp/speech.wav
uv run python -m hoast.stt /tmp/speech.wav
```

Detailed logs, profiles, minimal convolution/mixer reproductions, memory records
and generated audio are under `.cache/hoast/diagnostics/kokoro-fp16/`.

Native IR CLI smoke tests passed with both the project frontend and
kokoro-onnx 0.6.1 in the isolated environment, including fractional speed.
All 652 offline tests passed. Ruff and Pyright passed on changed application
code; the export and investigation scripts also passed Pyright against their
isolated environment.
