/* Experimental batch-one w1/w3 + SwiGLU, not wired into the runner.
 * x[K], q1/q3[N,K] U8, z1/z3[N] U8, s1/s3[N] FP32 (exact
 * promotions of finite FP16 scales), out[N] FP32; all buffers disjoint.
 * Dimensions K,N must be positive. Dispatch (LFM_MLP_LOCAL,ceil(N/R)),
 * local (LFM_MLP_LOCAL,1), where R=LFM_MLP_LOCAL/16. K/N tails are
 * guarded. FP16 denormal support is required for subnormal weights; do not
 * compile with relaxed math or denormal flushing. Each dequantized weight is
 * rounded to FP16 RTE before FP32 FMA, matching the existing IR path. Output
 * is (w1*x)/(1+exp(-w1*x))*(w3*x), without intermediate workspace or state.
 * Volatile preserves the half rounding boundary on the NEO compiler.
 */
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#ifndef LFM_MLP_LOCAL
#define LFM_MLP_LOCAL 128
#endif
#if LFM_MLP_LOCAL != 64 && LFM_MLP_LOCAL != 128 && LFM_MLP_LOCAL != 256
#error Unsupported LFM_MLP_LOCAL
#endif
__attribute__((reqd_work_group_size(LFM_MLP_LOCAL, 1, 1)))
__kernel void lfm_fused_mlp_ar(
    const __global float *x, const __global uchar *q1,
    const __global uchar *z1, const __global float *s1,
    const __global uchar *q3, const __global uchar *z3,
    const __global float *s3, __global float *out,
    const int k_size, const int n_size) {
    const int lid = (int)get_local_id(0);
    const int lane = lid & 15;
    const int n = (int)get_group_id(1) * (LFM_MLP_LOCAL / 16) + lid / 16;
    __local float a[LFM_MLP_LOCAL], b[LFM_MLP_LOCAL];
    float sum1 = 0.0f, sum3 = 0.0f;
    if (n < n_size) {
        const size_t base = (size_t)n * (size_t)k_size;
        const float zero1 = (float)z1[n], zero3 = (float)z3[n];
        const float scale1 = s1[n], scale3 = s3[n];
        for (int k = lane; k < k_size; k += 16) {
            const float v = x[k];
            volatile half h1 = convert_half_rte(((float)q1[base + k] - zero1) * scale1);
            volatile half h3 = convert_half_rte(((float)q3[base + k] - zero3) * scale3);
            sum1 = fma(v, convert_float(h1), sum1);
            sum3 = fma(v, convert_float(h3), sum3);
        }
    }
    a[lid] = sum1;
    b[lid] = sum3;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 8; stride > 0; stride >>= 1) {
        if (lane < stride) {
            a[lid] += a[lid + stride];
            b[lid] += b[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0 && n < n_size) {
        const float gate = a[lid];
        out[n] = (gate / (1.0f + exp(-gate))) * b[lid];
    }
}
#endif
