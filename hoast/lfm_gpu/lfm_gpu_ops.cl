/* Batch-one LFM non-matmul primitives, portable OpenCL C 1.2.
 *
 * All buffers are contiguous FP32 and non-overlapping unless explicitly noted.
 * Kernels use FP32 arithmetic, without fast-relaxed-math. Numerical association
 * differs from framework kernels; optimized paths are not bitwise equivalent.
 * Host must validate positive dimensions, finite epsilon > 0 and theta > 0,
 * even head_dim, q_heads divisible by kv_heads, nonnegative start, and integer
 * products/address ranges; allocations must cover the shapes specified below.
 * Dispatch exactly the documented global dimensions and local sizes.
 * All cache/state buffers are owned by one sequence and one layer. Reset them
 * between sequences. Commands sharing state must execute in order; use events
 * to order out-of-order queues. No kernel supports overlapping invocations on
 * the same mutable cache. For padded batches use unpadded batch-one tokens.
 *
 * Capacity failures: cache_append and attention do no writes when start < 0,
 * count <= 0 or count > capacity - start. Host MUST check these before launch;
 * silent no-write is only a memory-safety backstop, not an error notification.
 */

/* x/y [rows,width], weight [width]; global (rows). Output may equal x. */
__kernel void lfm_rmsnorm(const __global float *x,
                          const __global float *weight, __global float *y,
                          const int rows, const int width, const float epsilon) {
    int r = (int)get_global_id(0);
    if (r >= rows || width <= 0) return;
    size_t base = (size_t)r * (size_t)width;
    float sum = 0.0f;
    for (int i = 0; i < width; ++i) sum += x[base + i] * x[base + i];
    float inv = 1.0f / sqrt(sum / (float)width + epsilon);
    for (int i = 0; i < width; ++i) y[base + i] = x[base + i] * inv * weight[i];
}

/* Same inputs as lfm_rmsnorm; launch global (64, rows), local (64, 1).
 * One workgroup per row; output may equal x, but weight may not alias output.
 * Reduction is a 64-lane FP32 tree, so differs numerically from serial sum.
 */
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void lfm_rmsnorm_parallel(const __global float *x,
                                   const __global float *weight, __global float *y,
                                   const int rows, const int width, const float epsilon) {
    int lane = (int)get_local_id(0), row = (int)get_group_id(1);
    if (row >= rows || width <= 0) return;
    size_t base = (size_t)row * width;
    __local float sums[64];
    float acc = 0.0f;
    for (int i = lane; i < width; i += 64) {
        float value = x[base+i];
        acc += value * value;
    }
    sums[lane] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 32; stride > 0; stride >>= 1) {
        if (lane < stride) sums[lane] += sums[lane+stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float inv = 1.0f / sqrt(sums[0] / (float)width + epsilon);
    for (int i = lane; i < width; i += 64) y[base+i] = x[base+i] * inv * weight[i];
}

/* q [count,q_heads,dim], k [count,kv_heads,dim]; outputs same shapes.
 * start is absolute position of first token. Rotate-half, theta-base RoPE.
 * Input and output buffers must not overlap. Global (count, max(head counts), dim/2).
 */
__kernel void lfm_rope_qk(const __global float *q, const __global float *k,
                           __global float *qo, __global float *ko,
                           const int count, const int q_heads, const int kv_heads,
                           const int dim, const int start, const float theta) {
    int t = (int)get_global_id(0), h = (int)get_global_id(1);
    int i = (int)get_global_id(2), half_dim = dim / 2;
    if (t >= count || i >= half_dim) return;
    float angle = (float)(start + t) * pow(theta, -(2.0f * (float)i) / (float)dim);
    float c = cos(angle), s = sin(angle);
    if (h < q_heads) {
        size_t b = ((size_t)t * q_heads + h) * dim;
        float a = q[b+i], z = q[b+i+half_dim];
        qo[b+i] = a*c-z*s; qo[b+i+half_dim] = z*c+a*s;
    }
    if (h < kv_heads) {
        size_t b = ((size_t)t * kv_heads + h) * dim;
        float a = k[b+i], z = k[b+i+half_dim];
        ko[b+i] = a*c-z*s; ko[b+i+half_dim] = z*c+a*s;
    }
}

/* Append k,v [count,kv_heads,dim] to fixed caches [capacity,kv_heads,dim].
 * Input is already RoPE-rotated for K. Global (count,kv_heads,dim).
 * Append must finish before attention; untouched cache slots remain unchanged.
 */
__kernel void lfm_kv_append(const __global float *k, const __global float *v,
                             __global float *kc, __global float *vc,
                             const int count, const int kv_heads, const int dim,
                             const int start, const int capacity) {
    if (start < 0 || count <= 0 || start > capacity || count > capacity-start) return;
    int t = (int)get_global_id(0), h = (int)get_global_id(1), d = (int)get_global_id(2);
    if (t >= count || h >= kv_heads || d >= dim) return;
    size_t src = ((size_t)t * kv_heads + h) * dim + d;
    size_t dst = ((size_t)(start+t) * kv_heads + h) * dim + d;
    kc[dst] = k[src]; vc[dst] = v[src];
}

/* q/out [count,q_heads,dim], caches [capacity,kv_heads,dim]. Cache includes
 * this chunk and all prior tokens. Causal position is start+t, no padding.
 * Stable two-pass softmax; global (count,q_heads,dim). Each work item computes
 * one output coordinate independently. Uses scale=1/sqrt(dim), no dropout.
 */
__kernel void lfm_causal_gqa(const __global float *q, const __global float *kc,
                              const __global float *vc, __global float *out,
                              const int count, const int q_heads, const int kv_heads,
                              const int dim, const int start, const int capacity) {
    if (start < 0 || count <= 0 || start > capacity || count > capacity-start ||
        kv_heads <= 0 || q_heads <= 0 || q_heads % kv_heads != 0 || dim <= 0) return;
    int t = (int)get_global_id(0), h = (int)get_global_id(1), d = (int)get_global_id(2);
    if (t >= count || h >= q_heads || d >= dim) return;
    int kh = h / (q_heads / kv_heads), last = start+t;
    size_t qb = ((size_t)t * q_heads + h) * dim;
    float scale = 1.0f / sqrt((float)dim), maximum = -INFINITY;
    for (int p = 0; p <= last; ++p) {
        size_t b = ((size_t)p * kv_heads + kh) * dim;
        float dot = 0.0f;
        for (int j = 0; j < dim; ++j) dot += q[qb+j] * kc[b+j];
        maximum = fmax(maximum, dot * scale);
    }
    float denom = 0.0f, numer = 0.0f;
    for (int p = 0; p <= last; ++p) {
        size_t b = ((size_t)p * kv_heads + kh) * dim;
        float dot = 0.0f;
        for (int j = 0; j < dim; ++j) dot += q[qb+j] * kc[b+j];
        float a = exp(dot * scale - maximum);
        denom += a; numer += a * vc[b+d];
    }
    out[qb+d] = numer / denom;
}

/* Optimized GQA is two ordered launches using a caller-owned FP32 workspace
 * [max_query, q_heads, capacity]. Only rows [0,count) and valid causal columns
 * [0,start+t] are read/written; stale workspace contents need no clearing.
 * Allocate workspace once per concurrent request/layer invocation and retain it
 * across calls; q_heads and capacity must match that allocation. Host validates
 * count <= max_query, shape/products, capacity, and positive compatible heads.
 * No alias with inputs, cache or output. Stage 1 global (start+count,count*q_heads)
 * local NULL. Stage 2 global (64,count*q_heads), local (64,1). Enqueue after KV
 * append and enforce stage1 -> stage2 event dependency on out-of-order queues;
 * stage2 overwrites valid workspace scores with unnormalized softmax weights.
 * Do not reuse workspace until stage2 completes. FP32 tree reductions and exp
 * introduce small rounding differences from lfm_causal_gqa and framework paths.
 */
__kernel void lfm_gqa_scores(const __global float *q, const __global float *kc,
                              __global float *workspace, const int count,
                              const int q_heads, const int kv_heads, const int dim,
                              const int start, const int capacity) {
    if (start < 0 || count <= 0 || start > capacity || count > capacity-start ||
        kv_heads <= 0 || q_heads <= 0 || q_heads % kv_heads != 0 || dim <= 0) return;
    int p = (int)get_global_id(0), row = (int)get_global_id(1);
    int t = row / q_heads, h = row % q_heads;
    if (t >= count || p > start+t) return;
    int kh = h / (q_heads / kv_heads);
    size_t qb = (size_t)row * dim, kb = ((size_t)p * kv_heads + kh) * dim;
    float dot = 0.0f;
    for (int j = 0; j < dim; ++j) dot += q[qb+j] * kc[kb+j];
    workspace[(size_t)row * capacity + p] = dot / sqrt((float)dim);
}

/* Fixed-shape prefill score specialization: q [count,16,64], kc
 * [capacity,8,64], workspace [count,16,capacity]. Arguments after the three
 * buffers are count, start, capacity. Global size is
 * (64*ceil((start+count)/8), ceil(count/4), 8), local size (64,1,1).
 * Each group covers four queries, both heads sharing one KV head, and eight
 * positions; local K is transposed for access by score lanes. Only causal
 * columns p <= start+t are written. The existing lfm_gqa_apply consumes this
 * workspace unchanged. Use lfm_gqa_scores for other head shapes/dimensions or
 * as a fallback. The host must validate allocation sizes and dispatch products.
 * KV append must complete before this kernel and this kernel before apply;
 * enforce dependencies with events on out-of-order queues.
 */
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void lfm_gqa_scores_tiled_16_8_64(
    const __global float *q, const __global float *kc, __global float *workspace,
    const int count, const int start, const int capacity) {
    if (start < 0 || count <= 0 || start > capacity || count > capacity-start)
        return;
    int lane = (int)get_local_id(0);
    int p0 = (int)get_group_id(0) * 8;
    int t0 = (int)get_group_id(1) * 4;
    int kh = (int)get_group_id(2);
    if (kh >= 8 || t0 >= count || p0 > start + t0 + min(3, count-1-t0))
        return;
    __local float qtile[8 * 64];
    __local float ktile[64 * 8];
    for (int i = lane; i < 8*64; i += 64) {
        int r = i / 64, d = i % 64, t = t0 + r / 2;
        qtile[i] = t < count ? q[((size_t)t*16 + kh*2 + r%2)*64 + d] : 0.0f;
        int p = p0 + r;
        ktile[d*8 + r] = p < start+count ? kc[((size_t)p*8 + kh)*64 + d] : 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int r = lane / 8, c = lane % 8, t = t0 + r / 2, p = p0 + c;
    if (t < count && p <= start+t) {
        float dot = 0.0f;
        for (int d = 0; d < 64; ++d)
            dot += qtile[r*64+d] * ktile[d*8+c];
        size_t row = (size_t)t*16 + kh*2 + r%2;
        workspace[row*(size_t)capacity+p] = dot * 0.125f;
    }
}

__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void lfm_gqa_apply(__global float *workspace, const __global float *vc,
                             __global float *out, const int count,
                             const int q_heads, const int kv_heads, const int dim,
                             const int start, const int capacity) {
    if (start < 0 || count <= 0 || start > capacity || count > capacity-start ||
        kv_heads <= 0 || q_heads <= 0 || q_heads % kv_heads != 0 || dim <= 0) return;
    int lane = (int)get_local_id(0), row = (int)get_group_id(1);
    int t = row / q_heads, h = row % q_heads;
    if (t >= count) return;
    int last = start+t, kh = h / (q_heads / kv_heads);
    size_t wb = (size_t)row * capacity;
    __local float reduce[64];
    float maximum = -INFINITY;
    for (int p = lane; p <= last; p += 64) maximum = fmax(maximum, workspace[wb+p]);
    reduce[lane] = maximum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 32; stride > 0; stride >>= 1) {
        if (lane < stride) reduce[lane] = fmax(reduce[lane], reduce[lane+stride]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    maximum = reduce[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    float sum = 0.0f;
    for (int p = lane; p <= last; p += 64) {
        float a = exp(workspace[wb+p] - maximum);
        workspace[wb+p] = a;
        sum += a;
    }
    reduce[lane] = sum;
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    for (int stride = 32; stride > 0; stride >>= 1) {
        if (lane < stride) reduce[lane] += reduce[lane+stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float inv = 1.0f / reduce[0];
    for (int d = lane; d < dim; d += 64) {
        float value = 0.0f;
        for (int p = 0; p <= last; ++p)
            value += workspace[wb+p] * vc[((size_t)p * kv_heads + kh) * dim + d];
        out[(size_t)row * dim+d] = value * inv;
    }
}

/* bx,c,out [count,channels], weight [channels,3], state [channels,3].
 * State is oldest-to-newest B*X history; initialize to zero. Each work item
 * owns one channel and walks the chunk in sequence. Cross-correlation order:
 * w[0]*oldest + w[1]*previous + w[2]*current. Output is C * convolution.
 * Global (channels). State is updated to the last three B*X values, including
 * zero padding on first short chunk. This primitive excludes linear projections.
 */
__kernel void lfm_causal_conv3(const __global float *bx, const __global float *c,
                               const __global float *weight, __global float *state,
                               __global float *out, const int count, const int channels) {
    int ch = (int)get_global_id(0);
    if (ch >= channels || count <= 0) return;
    size_t s = (size_t)ch * 3;
    float a = state[s], b = state[s+1], z = state[s+2];
    float w0 = weight[s], w1 = weight[s+1], w2 = weight[s+2];
    for (int t = 0; t < count; ++t) {
        size_t i = (size_t)t * channels + ch;
        a = b; b = z; z = bx[i];
        out[i] = c[i] * (a*w0 + b*w1 + z*w2);
    }
    state[s] = a; state[s+1] = b; state[s+2] = z;
}
