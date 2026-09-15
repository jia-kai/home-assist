/* Gen9 cooperative-window convolution, inspired by OpenVINO's osv16 kernel.
 * Lanes own output channels. Each input window is loaded once per subgroup and
 * reused through register shuffles. FP16 tap sums accumulate channels in FP32.
 * Packed weights: [CI,KW,CO padded to16]. Lengths remain runtime arguments.
 */
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#define SG 16
#define PREFETCH 2
#define COP ((CO + 15) / 16 * 16)
#define WINDOW ((TILE - 1) * STRIDE + (KW - 1) * DILATION + 1)
#define BLOCKS ((WINDOW + SG - 1) / SG)

/** Compute an output tile without length-specific compilation.
 * Args:
 *   input:
 *     Contiguous FP32 activations (1,CI,width), converted to FP16 in registers.
 *
 *   weights:
 *     Packed signed INT8 weights (CI,KW,COP).
 *
 *   scales:
 *     FP16 channel scales padded to COP.
 *
 *   output:
 *     FP32 output (1,CO,out_width), rounded to FP16 after accumulation.
 *
 *   width:
 *     Input temporal length in elements.
 *
 *   out_width:
 *     Output temporal length in elements.
 *
 *   input_offset:
 *     Input FP32 element offset within imported pages.
 *
 *   output_offset:
 *     Output FP32 element offset within imported pages.
 *
 */
__attribute__((intel_reqd_sub_group_size(SG)))
__attribute__((reqd_work_group_size(SG, 1, 1))) __kernel void
hoast_conv(const __global float *input, const __global char *weights,
           const __global half *scales, __global float *output, const int width,
           const int out_width, const int input_offset,
           const int output_offset) {
  const int lane = get_local_id(0);
  const int o = get_group_id(1) * SG + lane;
  const int t0 = get_group_id(0) * TILE;
  const int left = t0 * STRIDE - PAD;
  const half scale = o < CO ? scales[o] : (half)0;
  const bool interior = left >= 0 && left + WINDOW <= width;
  float total[TILE];
#pragma unroll
  for (int t = 0; t < TILE; ++t)
    total[t] = 0;
  for (int c = 0; c < CI; ++c) {
    half window[BLOCKS], partial[TILE], prefetched[PREFETCH];
    int input_base = c * width + left;
    input_base += input_offset;
#pragma unroll
    for (int b = 0; b < BLOCKS; ++b) {
      const int position = b * SG + lane;
      if (interior) {
        window[b] = position < WINDOW
                        ? convert_half(input[input_base + position])
                        : (half)0;
      } else {
        window[b] =
            position < WINDOW && left + position >= 0 && left + position < width
                ? convert_half(input[input_base + position])
                : (half)0;
      }
    }
#pragma unroll
    for (int t = 0; t < TILE; ++t)
      partial[t] = 0;
#pragma unroll
    for (int p = 0; p < PREFETCH; ++p)
      prefetched[p] =
          p < KW && o < CO
              ? convert_half(weights[(c * KW + p) * COP + o]) * scale
              : (half)0;
#pragma unroll
    for (int k = 0; k < KW; ++k) {
      const half w = prefetched[k % PREFETCH];
      if (k + PREFETCH < KW)
        prefetched[k % PREFETCH] =
            o < CO ? convert_half(weights[(c * KW + k + PREFETCH) * COP + o]) *
                         scale
                   : (half)0;
#pragma unroll
      for (int t = 0; t < TILE; ++t) {
        const int position = t * STRIDE + k * DILATION;
        const half x =
            intel_sub_group_shuffle(window[position / SG], position % SG);
        partial[t] = fma(x, w, partial[t]);
      }
    }
#pragma unroll
    for (int t = 0; t < TILE; ++t)
      total[t] += convert_float(partial[t]);
  }
  if (o < CO) {
    int base = o * out_width;
    base += output_offset;
#pragma unroll
    for (int t = 0; t < TILE; ++t)
      if (t0 + t < out_width)
        output[base + t0 + t] = convert_half_rte(total[t]);
  }
}
