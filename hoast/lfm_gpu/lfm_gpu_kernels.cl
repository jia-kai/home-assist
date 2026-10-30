/* Batch-one LFM weight-only matrix/vector primitive, OpenCL C 1.2.
 *
 * Contract: y[n] = sum_k x[k] * ((float)q[n,k] - (float)zero[n])
 *                                      * scale[n].
 * q is contiguous row-major unsigned bytes [N,K], zero unsigned bytes [N],
 * scale float32 [N] (convert checkpoint's FP16 scale to FP32 once at load),
 * x and y float32 [K] and [N]. The model's asymmetric per-output quantization
 * has no grouping. Buffers must not overlap. Dimensions are positive int32;
 * K must be <= INT_MAX-15, N <= INT_MAX-7, and N*K and addresses must fit
 * size_t. No padding or extensions are required.
 *
 * Eight output rows share a 128-workitem group; 16 lanes per row cooperatively
 * read consecutive weights and reduce partial FP32 sums. Dispatch global size
 * (128, ceil(N/8)), local size (128, 1). K and N may have arbitrary tails.
 * Scaling is factored outside the sum, changing floating-point association
 * versus per-element dequantization; the checkpoint graph also rounds the
 * dequantized weights to FP16, unlike this kernel. Extreme inputs can overflow
 * intermediate values. Neither the complete model nor its output accuracy is guaranteed by
 * this isolated primitive. No relaxed-math build options are needed.
 */
__attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void lfm_matvec_u8(
    const __global float *x, const __global uchar *q,
    const __global uchar *zero, const __global float *scale,
    __global float *y, const int k_size, const int n_size) {
    const int lid = get_local_id(0);
    const int lane = lid & 15;
    const int row = (int)get_group_id(1) * 8 + lid / 16;
    __local float partial[128];
    float acc = 0.0f;
    if (row < n_size) {
        const size_t base = (size_t)row * (size_t)k_size;
        const float z = (float)zero[row];
        for (int k = lane; k < k_size; k += 16) {
            acc = fma(x[k], (float)q[base + (size_t)k] - z, acc);
        }
    }
    partial[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 8; stride > 0; stride >>= 1) {
        if (lane < stride) partial[lid] += partial[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0 && row < n_size) y[row] = partial[lid] * scale[row];
}

/* Prefill X[M,K] @ dequantized Q[N,K]^T -> Y[M,N].
 * X/Y are contiguous FP32; Q, zero and scale use the matvec contract above.
 * All buffers must not overlap. M, N and K are positive int32, with M and N
 * <= INT_MAX-7, K <= INT_MAX-15 and all products/addresses fitting size_t.
 * Dispatch local (128,1,1), global (128,ceil(N/8),ceil(M/8)). Eight token
 * rows share each weight load; 16 lanes reduce each of eight output channels.
 * M/N/K tails require no padding. The kernel uses 4096 bytes local memory.
 * FP32 FMA, reduction and post-scaling differ in rounding from the checkpoint's
 * FP16 dequantized weight multiply; extreme inputs may overflow intermediates.
 * On UHD 630 (NEO 24.35), synthetic [M,N,K]=[32,1024,1024] measured about
 * 1.17 ms device time, median of five warmed runs, excluding copies. Results
 * depend on clocks, driver, and shape; this is not an end-to-end benchmark.
 */
__attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void lfm_matmul_u8(
    const __global float *x, const __global uchar *q,
    const __global uchar *zero, const __global float *scale,
    __global float *y, const int m_size, const int k_size, const int n_size) {
    const int lid = (int)get_local_id(0);
    const int lane = lid & 15;
    const int n = (int)get_group_id(1) * 8 + lid / 16;
    const int m = (int)get_group_id(2) * 8;
    __local float8 partial[128];
    float8 sum = (float8)(0.0f);
    if (n < n_size) {
        const size_t base = (size_t)n * k_size;
        const float z = (float)zero[n];
        for (int k = lane; k < k_size; k += 16) {
            const float w = (float)q[base + k] - z;
            float8 v = (float8)(0.0f);
            if (m + 0 < m_size) v.s0 = x[(size_t)(m + 0) * k_size + k];
            if (m + 1 < m_size) v.s1 = x[(size_t)(m + 1) * k_size + k];
            if (m + 2 < m_size) v.s2 = x[(size_t)(m + 2) * k_size + k];
            if (m + 3 < m_size) v.s3 = x[(size_t)(m + 3) * k_size + k];
            if (m + 4 < m_size) v.s4 = x[(size_t)(m + 4) * k_size + k];
            if (m + 5 < m_size) v.s5 = x[(size_t)(m + 5) * k_size + k];
            if (m + 6 < m_size) v.s6 = x[(size_t)(m + 6) * k_size + k];
            if (m + 7 < m_size) v.s7 = x[(size_t)(m + 7) * k_size + k];
            sum = fma(v, (float8)(w), sum);
        }
    }
    partial[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 8; stride > 0; stride >>= 1) {
        if (lane < stride) partial[lid] += partial[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0 && n < n_size) {
        sum = partial[lid] * scale[n];
        if (m + 0 < m_size) y[(size_t)(m + 0) * n_size + n] = sum.s0;
        if (m + 1 < m_size) y[(size_t)(m + 1) * n_size + n] = sum.s1;
        if (m + 2 < m_size) y[(size_t)(m + 2) * n_size + n] = sum.s2;
        if (m + 3 < m_size) y[(size_t)(m + 3) * n_size + n] = sum.s3;
        if (m + 4 < m_size) y[(size_t)(m + 4) * n_size + n] = sum.s4;
        if (m + 5 < m_size) y[(size_t)(m + 5) * n_size + n] = sum.s5;
        if (m + 6 < m_size) y[(size_t)(m + 6) * n_size + n] = sum.s6;
        if (m + 7 < m_size) y[(size_t)(m + 7) * n_size + n] = sum.s7;
    }
}

/* Optional IR-dequantization variants, present only with cl_khr_fp16.
 * Their argument lists, layouts and dispatches are identical to lfm_matvec_u8
 * and lfm_matmul_u8 respectively. Every FP32 scale must be the exact promotion
 * of a finite FP16 checkpoint scale. For each weight, the integer difference
 * is multiplied by that scale, rounded to FP16 round-to-nearest-even, then
 * promoted to FP32 before FMA accumulation. The integer difference and product
 * are exactly representable in FP32 for finite FP16 scales. Do not use
 * relaxed-math/denorm-flushing flags if IR-style rounding is required; FP16
 * subnormals additionally require device half denormal support. These kernels
 * reproduce weight dequantization, not necessarily the IR matmul accumulation
 * precision or full-model output. Infinities from FP16 overflow follow IEEE
 * arithmetic, including NaN for zero activations multiplied by infinity.
 * The volatile half temporary enforces a materialized rounding boundary: the
 * UHD 630 NEO compiler eliminated a plain half-to-float conversion pair.
 */
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void lfm_matvec_u8_f16_dequant(
    const __global float *x, const __global uchar *q,
    const __global uchar *zero, const __global float *scale,
    __global float *y, const int k_size, const int n_size) {
    const int lid = (int)get_local_id(0);
    const int lane = lid & 15;
    const int n = (int)get_group_id(1) * 8 + lid / 16;
    __local float partial[128];
    float sum = 0.0f;
    if (n < n_size) {
        const size_t base = (size_t)n * (size_t)k_size;
        const float z = (float)zero[n];
        const float s = scale[n];
        for (int k = lane; k < k_size; k += 16) {
            volatile half rounded = convert_half_rte(((float)q[base + k] - z) * s);
            const float w = convert_float(rounded);
            sum = fma(x[k], w, sum);
        }
    }
    partial[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 8; stride > 0; stride >>= 1) {
        if (lane < stride) partial[lid] += partial[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0 && n < n_size) y[n] = partial[lid];
}

/* Opt-in vec4 batch-one projection. Arguments, buffer layout, dimension and
 * non-aliasing requirements match lfm_matvec_u8_f16_dequant. Dispatch local
 * (128,1), global (128,ceil(N/8)). Each lane handles four adjacent values;
 * row and K tails need no padding. vload4 supports unaligned row starts.
 * Per-weight FP16 rounding and FP32 FMA match the IR-dequant contract above,
 * but the changed reduction order can change output rounding. This kernel is
 * not selected by the runner by default. */
__attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void lfm_matvec_u8_f16_dequant_vec4(
    const __global float *x, const __global uchar *q,
    const __global uchar *zero, const __global float *scale,
    __global float *y, const int k_size, const int n_size) {
    const int lid = (int)get_local_id(0);
    const int lane = lid & 15;
    const int n = (int)get_group_id(1) * 8 + lid / 16;
    __local float partial[128];
    float sum = 0.0f;
    if (n < n_size) {
        const size_t base = (size_t)n * (size_t)k_size;
        const float z = (float)zero[n];
        const float s = scale[n];
        for (size_t k = (size_t)lane * 4; k < (size_t)k_size; k += 64) {
            if (k + 4 <= (size_t)k_size) {
                const uchar4 a = vload4(0, q + base + k);
                const float4 v = vload4(0, x + k);
                volatile half h0 = convert_half_rte(((float)a.s0 - z) * s);
                volatile half h1 = convert_half_rte(((float)a.s1 - z) * s);
                volatile half h2 = convert_half_rte(((float)a.s2 - z) * s);
                volatile half h3 = convert_half_rte(((float)a.s3 - z) * s);
                sum = fma(v.s0, convert_float(h0), sum);
                sum = fma(v.s1, convert_float(h1), sum);
                sum = fma(v.s2, convert_float(h2), sum);
                sum = fma(v.s3, convert_float(h3), sum);
            } else {
                for (size_t j = k; j < (size_t)k_size; ++j) {
                    volatile half h = convert_half_rte(((float)q[base + j] - z) * s);
                    sum = fma(x[j], convert_float(h), sum);
                }
            }
        }
    }
    partial[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 8; stride > 0; stride >>= 1) {
        if (lane < stride) partial[lid] += partial[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0 && n < n_size) y[n] = partial[lid];
}

__attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void lfm_matmul_u8_f16_dequant(
    const __global float *x, const __global uchar *q,
    const __global uchar *zero, const __global float *scale,
    __global float *y, const int m_size, const int k_size, const int n_size) {
    const int lid = (int)get_local_id(0);
    const int lane = lid & 15;
    const int n = (int)get_group_id(1) * 8 + lid / 16;
    const int m = (int)get_group_id(2) * 8;
    __local float8 partial[128];
    float8 sum = (float8)(0.0f);
    if (n < n_size) {
        const size_t base = (size_t)n * (size_t)k_size;
        const float z = (float)zero[n];
        const float s = scale[n];
        for (int k = lane; k < k_size; k += 16) {
            volatile half rounded = convert_half_rte(((float)q[base + k] - z) * s);
            const float w = convert_float(rounded);
            float8 v = (float8)(0.0f);
            if (m + 0 < m_size) v.s0 = x[(size_t)(m + 0) * k_size + k];
            if (m + 1 < m_size) v.s1 = x[(size_t)(m + 1) * k_size + k];
            if (m + 2 < m_size) v.s2 = x[(size_t)(m + 2) * k_size + k];
            if (m + 3 < m_size) v.s3 = x[(size_t)(m + 3) * k_size + k];
            if (m + 4 < m_size) v.s4 = x[(size_t)(m + 4) * k_size + k];
            if (m + 5 < m_size) v.s5 = x[(size_t)(m + 5) * k_size + k];
            if (m + 6 < m_size) v.s6 = x[(size_t)(m + 6) * k_size + k];
            if (m + 7 < m_size) v.s7 = x[(size_t)(m + 7) * k_size + k];
            sum = fma(v, (float8)(w), sum);
        }
    }
    partial[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 8; stride > 0; stride >>= 1) {
        if (lane < stride) partial[lid] += partial[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0 && n < n_size) {
        sum = partial[lid];
        if (m + 0 < m_size) y[(size_t)(m + 0) * n_size + n] = sum.s0;
        if (m + 1 < m_size) y[(size_t)(m + 1) * n_size + n] = sum.s1;
        if (m + 2 < m_size) y[(size_t)(m + 2) * n_size + n] = sum.s2;
        if (m + 3 < m_size) y[(size_t)(m + 3) * n_size + n] = sum.s3;
        if (m + 4 < m_size) y[(size_t)(m + 4) * n_size + n] = sum.s4;
        if (m + 5 < m_size) y[(size_t)(m + 5) * n_size + n] = sum.s5;
        if (m + 6 < m_size) y[(size_t)(m + 6) * n_size + n] = sum.s6;
        if (m + 7 < m_size) y[(size_t)(m + 7) * n_size + n] = sum.s7;
    }
}

/* Long-prompt two-stage alternative. lfm_prepare_u8_f16 writes exactly
 * 2*N*K bytes of row-major half weights to a caller-owned, suitably aligned
 * workspace, using the same FP16 rounding as the IR-dequantization kernels.
 * Dispatch local (64), global (round_up(N*K,64)); 2*N*K and the rounded
 * global size must fit size_t. The caller must order the next kernel after this one
 * and must not reuse workspace until all readers finish. Positive K/N and
 * promoted FP16 scales use the same contract as the kernels above.
 *
 * lfm_matmul_prepared_f16 consumes that workspace, with FP32 x[M,K], output
 * y[M,N], argument order (x, prepared, y, M, K, N). Dispatch local (64,1,1),
 * global (64,ceil(N/8),ceil(M/8)). The M/N limits above still apply; K is
 * <= INT_MAX-7. Buffers must not overlap. This trades workspace and an extra
 * launch for weight sharing across the entire prompt; no allocation occurs in
 * either kernel. The caller decides whether the measured tradeoff is useful.
 */
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void lfm_prepare_u8_f16(
    const __global uchar *q, const __global uchar *zero,
    const __global float *scale, __global half *prepared,
    const int k_size, const int n_size) {
    const size_t i = get_global_id(0);
    if (i < (size_t)k_size * (size_t)n_size) {
        const size_t n = i / (size_t)k_size;
        const float value = ((float)q[i] - (float)zero[n]) * scale[n];
        vstore_half_rte(value, i, prepared);
    }
}

__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void lfm_matmul_prepared_f16(
    const __global float *x, const __global half *prepared,
    __global float *y, const int m_size, const int k_size, const int n_size) {
    const int lid = (int)get_local_id(0);
    const int lane = lid & 7;
    const int n = (int)get_group_id(1) * 8 + lid / 8;
    const int m = (int)get_group_id(2) * 8;
    __local float8 partial[64];
    float8 sum = (float8)(0.0f);
    if (n < n_size) {
        const size_t base = (size_t)n * (size_t)k_size;
        for (int k = lane; k < k_size; k += 8) {
            const float w = vload_half(0, prepared + base + k);
            float8 v = (float8)(0.0f);
            if (m + 0 < m_size) v.s0 = x[(size_t)(m + 0) * k_size + k];
            if (m + 1 < m_size) v.s1 = x[(size_t)(m + 1) * k_size + k];
            if (m + 2 < m_size) v.s2 = x[(size_t)(m + 2) * k_size + k];
            if (m + 3 < m_size) v.s3 = x[(size_t)(m + 3) * k_size + k];
            if (m + 4 < m_size) v.s4 = x[(size_t)(m + 4) * k_size + k];
            if (m + 5 < m_size) v.s5 = x[(size_t)(m + 5) * k_size + k];
            if (m + 6 < m_size) v.s6 = x[(size_t)(m + 6) * k_size + k];
            if (m + 7 < m_size) v.s7 = x[(size_t)(m + 7) * k_size + k];
            sum = fma(v, (float8)(w), sum);
        }
    }
    partial[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 4; stride > 0; stride >>= 1) {
        if (lane < stride) partial[lid] += partial[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0 && n < n_size) {
        sum = partial[lid];
        if (m + 0 < m_size) y[(size_t)(m + 0) * n_size + n] = sum.s0;
        if (m + 1 < m_size) y[(size_t)(m + 1) * n_size + n] = sum.s1;
        if (m + 2 < m_size) y[(size_t)(m + 2) * n_size + n] = sum.s2;
        if (m + 3 < m_size) y[(size_t)(m + 3) * n_size + n] = sum.s3;
        if (m + 4 < m_size) y[(size_t)(m + 4) * n_size + n] = sum.s4;
        if (m + 5 < m_size) y[(size_t)(m + 5) * n_size + n] = sum.s5;
        if (m + 6 < m_size) y[(size_t)(m + 6) * n_size + n] = sum.s6;
        if (m + 7 < m_size) y[(size_t)(m + 7) * n_size + n] = sum.s7;
    }
}
/* Transposed prepared-weight alternative for prefill. The prepare kernel takes
 * (q, zero, scale, prepared, K, N), with the same input contract as
 * lfm_prepare_u8_f16, but writes a distinct K-major half layout [K,N]:
 * prepared[k*N+n] = round_to_half_rte((q[n*K+k]-zero[n])*scale[n]).
 * Its workspace is exactly 2*K*N bytes and is not interchangeable with the
 * N-major workspace used by lfm_matmul_prepared_f16. Dispatch local (16,16),
 * global (round_up(K,16),round_up(N,16)).
 *
 * lfm_matmul_prepared_kn_f16 takes (x, prepared, y, M, K, N), with contiguous
 * FP32 x[M,K] and y[M,N]. Dispatch local (8,8), global
 * (8*ceil(N/32),8*ceil(M/32)); each group computes a 32x32 output tile.
 * Arbitrary M/N/K tails require no padding. M and N must be positive and at
 * most INT_MAX-31; K must be positive and at most INT_MAX-15. Products,
 * rounded global sizes and byte addresses must fit size_t. Buffers must not
 * overlap; order preparation before the matmul and do not reuse the workspace
 * until all readers finish. The pack needs a 256-work-item group.
 *
 * The finite FP32 scales must be exact promotions of FP16 checkpoint scales.
 * vstore_half_rte materializes IR-style FP16 dequantization before FP32 FMA.
 * Do not enable relaxed math or denormal flushing; half denormal support is
 * required for subnormal equivalence. Half overflow produces infinities and
 * normal IEEE arithmetic applies. Matmul accumulation order differs from the
 * IR and other kernels, so complete-model equivalence is not guaranteed.
 */
__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void lfm_prepare_u8_f16_kn(
    const __global uchar *q, const __global uchar *zero,
    const __global float *scale, __global half *prepared,
    const int k_size, const int n_size) {
    const int k = (int)get_global_id(0);
    const int n = (int)get_global_id(1);
    __local half tile[16][17];
    const float value = k < k_size && n < n_size
        ? ((float)q[(size_t)n * k_size + k] - (float)zero[n]) * scale[n]
        : 0.0f;
    vstore_half_rte(value, 0, &tile[get_local_id(1)][get_local_id(0)]);
    barrier(CLK_LOCAL_MEM_FENCE);
    const int ko = (int)get_group_id(0) * 16 + (int)get_local_id(1);
    const int no = (int)get_group_id(1) * 16 + (int)get_local_id(0);
    if (ko < k_size && no < n_size)
        prepared[(size_t)ko * n_size + no] = tile[get_local_id(0)][get_local_id(1)];
}

__attribute__((reqd_work_group_size(8, 8, 1)))
__kernel void lfm_matmul_prepared_kn_f16(
    const __global float *x, const __global half *prepared, __global float *y,
    const int m_size, const int k_size, const int n_size) {
    const int cx = (int)get_local_id(0);
    const int ry = (int)get_local_id(1);
    const int tid = ry * 8 + cx;
    const int mb = (int)get_group_id(1) * 32;
    const int nb = (int)get_group_id(0) * 32;
    __local float a[4][32];
    __local half b[4][32];
    float4 c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
    for (int kb = 0; kb < k_size; kb += 4) {
        for (int i = tid; i < 4 * 32; i += 64) {
            const int ak = i % 4, am = i / 4;
            const int m = mb + am, ka = kb + ak;
            a[ak][am] = m < m_size && ka < k_size
                ? x[(size_t)m * k_size + ka] : 0.0f;
            const int kk = i / 32, col = i % 32;
            const int k = kb + kk, n = nb + col;
            b[kk][col] = n < n_size && k < k_size
                ? prepared[(size_t)k * n_size + n] : (half)0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int kk = 0; kk < 4; kk++) {
            const float4 w = vload_half4(0, &b[kk][cx * 4]);
            c0 = fma((float4)(a[kk][ry * 4]), w, c0);
            c1 = fma((float4)(a[kk][ry * 4 + 1]), w, c1);
            c2 = fma((float4)(a[kk][ry * 4 + 2]), w, c2);
            c3 = fma((float4)(a[kk][ry * 4 + 3]), w, c3);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int j = 0; j < 4; j++) {
        const int n = nb + cx * 4 + j, m = mb + ry * 4;
        if (n < n_size) {
            if (m < m_size) y[(size_t)m * n_size + n] = c0[j];
            if (m + 1 < m_size) y[(size_t)(m + 1) * n_size + n] = c1[j];
            if (m + 2 < m_size) y[(size_t)(m + 2) * n_size + n] = c2[j];
            if (m + 3 < m_size) y[(size_t)(m + 3) * n_size + n] = c3[j];
        }
    }
}
#endif
