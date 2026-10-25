/* Experimental portable OpenCL 1.2 weight-only speech kernels.
 * Workgroup: 8 rows/time positions x 32 output channels, 64 workitems.
 * Reduction tile: 32. Local storage: 1024 bytes FP32 + 1024 bytes INT8.
 * Weights stay signed char in global AND local memory. Only four register
 * values are dequantized at a time; no float/half weight tensor is allocated.
 * Activations, scales, fma accumulators and outputs are FP32. No integer dot,
 * subgroup, FP16, or fast-relaxed-math extension is required.
 * SPEECH_POSTSCALE=0 retains per-weight scaling before the dot (reference).
 * SPEECH_POSTSCALE=1 accumulates converted raw bytes, then scales once per
 * output. This reassociates FP32 arithmetic and can overflow raw accumulators
 * for extreme activations even when the scaled reference would remain finite.
 */
#define ROWS 8
#define COLS 32
#define REDUCE 32
#ifndef SPEECH_POSTSCALE
#define SPEECH_POSTSCALE 0
#endif
#if SPEECH_POSTSCALE
#define MATMUL_NAME speech_matmul_postscale
#define CONV_NAME speech_conv1d_postscale
#else
#define MATMUL_NAME speech_matmul
#define CONV_NAME speech_conv1d
#endif

/* Compute tiled A[M,K] @ dequantized W[K,N] in FP32.
 * All dimensions and tails are runtime values; M need not be a tile multiple.
 * Args:
 *   input:
 *     Contiguous FP32 activations (rows,reduction).
 *
 *   weights:
 *     Packed signed INT8 weights (reduction,padded).
 *
 *   scales:
 *     FP32 per-output scales (padded,), with zero padded channels.
 *
 *   output:
 *     Writable FP32 result (rows,channels).
 *
 *   rows:
 *     Positive token/row count M.
 *
 *   reduction:
 *     Positive shared dimension K.
 *
 *   channels:
 *     Positive logical output channel count N.
 *
 *   padded:
 *     Channels rounded upward to a multiple of 32.
 *
 */
__attribute__((reqd_work_group_size(8, 8, 1)))
__kernel void MATMUL_NAME(const __global float *input,
                            const __global char *weights,
                            const __global float *scales,
                            __global float *output,
                            int rows, int reduction, int channels, int padded) {
    const int cx = get_local_id(0);
    const int ry = get_local_id(1);
    const int tid = ry * 8 + cx;
    const int row = get_group_id(1) * ROWS + ry;
    const int col = get_group_id(0) * COLS + cx * 4;
    __local float a[ROWS * REDUCE];
    __local char b[REDUCE * COLS];
    const float4 scale = vload4(0, scales + col);
    float4 sum = (float4)(0.0f);
    for (int base = 0; base < reduction; base += REDUCE) {
        for (int i = tid; i < ROWS * REDUCE; i += 64) {
            const int r = get_group_id(1) * ROWS + i / REDUCE;
            const int k = base + i % REDUCE;
            a[i] = r < rows && k < reduction ? input[r * reduction + k] : 0.0f;
        }
        for (int i = tid; i < REDUCE * COLS; i += 64) {
            const int k = base + i / COLS;
            const int c = get_group_id(0) * COLS + i % COLS;
            b[i] = k < reduction ? weights[k * padded + c] : (char)0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < REDUCE; ++k) {
            float4 w = convert_float4(vload4(0, b + k * COLS + cx * 4));
#if !SPEECH_POSTSCALE
            w *= scale;
#endif
            sum = fma((float4)(a[ry * REDUCE + k]), w, sum);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
#if SPEECH_POSTSCALE
    sum *= scale;
#endif
    if (row < rows) {
        if (col + 0 < channels) output[row * channels + col + 0] = sum.s0;
        if (col + 1 < channels) output[row * channels + col + 1] = sum.s1;
        if (col + 2 < channels) output[row * channels + col + 2] = sum.s2;
        if (col + 3 < channels) output[row * channels + col + 3] = sum.s3;
    }
}

/* Stride-one, batch-one convolution with explicit symmetric zero padding.
 * X[CI,T], packed W[CI*KW,ceil(CO/32)*32], Y[CO,T]. A tile is gathered
 * directly from X, including dilated taps; no im2col tensor is materialized.
 * padding must equal dilation*(KW-1)/2 (odd KW) for this same-length contract.
 * Args:
 *   input:
 *     Contiguous FP32 activations (input_channels,width).
 *
 *   weights:
 *     Signed INT8 weights (input_channels*taps,padded), with taps fastest.
 *
 *   scales:
 *     FP32 per-output scales (padded,), with zero padded channels.
 *
 *   output:
 *     Writable channel-major FP32 result (channels,width).
 *
 *   width:
 *     Positive temporal length in samples, e.g. frames*cumulative_upsampling.
 *
 *   input_channels:
 *     Positive input channel count.
 *
 *   channels:
 *     Positive logical output channel count.
 *
 *   taps:
 *     Positive odd filter length.
 *
 *   dilation:
 *     Positive tap spacing in input samples.
 *
 *   padding:
 *     Symmetric zero padding in input samples.
 *
 *   padded:
 *     Output channels rounded upward to a multiple of 32.
 *
 */
__attribute__((reqd_work_group_size(8, 8, 1)))
__kernel void CONV_NAME(const __global float *input,
                            const __global char *weights,
                            const __global float *scales,
                            __global float *output,
                            int width, int input_channels, int channels,
                            int taps, int dilation, int padding, int padded) {
    const int cx = get_local_id(0);
    const int ry = get_local_id(1);
    const int tid = ry * 8 + cx;
    const int t = get_group_id(1) * ROWS + ry;
    const int col = get_group_id(0) * COLS + cx * 4;
    const int reduction = input_channels * taps;
    __local float a[ROWS * REDUCE];
    __local char b[REDUCE * COLS];
    const float4 scale = vload4(0, scales + col);
    float4 sum = (float4)(0.0f);
    for (int base = 0; base < reduction; base += REDUCE) {
        for (int i = tid; i < ROWS * REDUCE; i += 64) {
            const int time = get_group_id(1) * ROWS + i / REDUCE;
            const int r = base + i % REDUCE;
            const int position = time + (r % taps) * dilation - padding;
            a[i] = time < width && r < reduction && position >= 0 && position < width
                ? input[(r / taps) * width + position] : 0.0f;
        }
        for (int i = tid; i < REDUCE * COLS; i += 64) {
            const int r = base + i / COLS;
            const int c = get_group_id(0) * COLS + i % COLS;
            b[i] = r < reduction ? weights[r * padded + c] : (char)0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < REDUCE; ++k) {
            float4 w = convert_float4(vload4(0, b + k * COLS + cx * 4));
#if !SPEECH_POSTSCALE
            w *= scale;
#endif
            sum = fma((float4)(a[ry * REDUCE + k]), w, sum);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
#if SPEECH_POSTSCALE
    sum *= scale;
#endif
    if (t < width) {
        if (col + 0 < channels) output[(col + 0) * width + t] = sum.s0;
        if (col + 1 < channels) output[(col + 1) * width + t] = sum.s1;
        if (col + 2 < channels) output[(col + 2) * width + t] = sum.s2;
        if (col + 3 < channels) output[(col + 3) * width + t] = sum.s3;
    }
}
