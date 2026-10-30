/* Candidate batch-one LFM fusions; OpenCL 1.2, FP32, no fast math.
 * All dimensions and byte products must be validated by the host. Buffers do
 * not overlap except residual is updated in place. State belongs to one layer
 * and sequence; initialize it to zero at reset and order launches using events.
 * Capacity failures in rotary append are no-write backstops, not error reports.
 */

/* AR: input [count,3,width], weights/state [width,3], output [count,width].
 * Global (width). State is updated in place; serial token order supports chunks.
 */
__kernel void lfm_fused_conv_ar(const __global float *input,
    const __global float *weight, __global float *state, __global float *out,
    int count, int width) {
    int ch = get_global_id(0);
    if (ch >= width || count <= 0) return;
    size_t s = (size_t)ch * 3;
    float a = state[s], b = state[s+1], z = state[s+2];
    float w0 = weight[s], w1 = weight[s+1], w2 = weight[s+2];
    for (int t = 0; t < count; ++t) {
        size_t i = (size_t)t * 3 * width + ch;
        a = b; b = z; z = input[i] * input[i+2*width];
        out[(size_t)t*width+ch] = input[i+width] * (a*w0 + b*w1 + z*w2);
    }
    state[s] = a; state[s+1] = b; state[s+2] = z;
}

/* Prefill: global (width,count). Separate state_in and state_out are required:
 * this permits all tokens to read the initial state while the last token writes
 * the final history. Swap the state buffers between ordered chunks.
 */
__kernel void lfm_fused_conv_prefill(const __global float *input,
    const __global float *weight, const __global float *state_in,
    __global float *state_out, __global float *out, int count, int width) {
    int ch = get_global_id(0), t = get_global_id(1);
    if (ch >= width || t >= count || count <= 0) return;
    size_t s = (size_t)ch*3, i = (size_t)t*3*width+ch;
    float z = input[i]*input[i+2*width];
    float b = t >= 1 ? input[i-3*width]*input[i-3*width+2*width] : state_in[s+2];
    float a = t >= 2 ? input[i-6*width]*input[i-6*width+2*width] :
              (t == 1 ? state_in[s+2] : state_in[s+1]);
    out[(size_t)t*width+ch] = input[i+width] *
        (a*weight[s] + b*weight[s+1] + z*weight[s+2]);
    if (t == count-1) { state_out[s] = a; state_out[s+1] = b; state_out[s+2] = z; }
}

/* residual += update, then parallel RMSNorm, matching the runner's 64-lane
 * reduction. Global (64,rows), local (64,1), output separate from residual.
 */
__attribute__((reqd_work_group_size(64,1,1)))
__kernel void lfm_fused_add_rms(__global float *residual,
    const __global float *update, const __global float *weight,
    __global float *out, int rows, int width, float epsilon) {
    int lane = get_local_id(0), row = get_group_id(1);
    if (row >= rows || width <= 0) return;
    size_t base = (size_t)row*width;
    __local float sums[64];
    float acc = 0.0f;
    for (int i = lane; i < width; i += 64) {
        float x = residual[base+i] + update[base+i];
        residual[base+i] = x;
        acc += x*x;
    }
    sums[lane] = acc;
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    for (int stride = 32; stride > 0; stride >>= 1) {
        if (lane < stride) sums[lane] += sums[lane+stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float inv = 1.0f / sqrt(sums[0] / (float)width + epsilon);
    for (int i = lane; i < width; i += 64)
        out[base+i] = residual[base+i] * inv * weight[i];
}

/* Rotate q/k and append k/v directly to caches. Global
 * (count,max(q_heads,kv_heads),dim/2); cache [capacity,kv_heads,dim].
 * Output q is separate from all inputs and caches. Invalid ranges write nothing.
 */
__kernel void lfm_fused_rope_append(const __global float *q,
    const __global float *k, const __global float *v, __global float *qo,
    __global float *kc, __global float *vc, int count, int q_heads,
    int kv_heads, int dim, int start, int capacity, float theta) {
    if (start < 0 || count <= 0 || start > capacity || count > capacity-start) return;
    int t = get_global_id(0), h = get_global_id(1), i = get_global_id(2);
    int half_dim = dim/2;
    if (t >= count || i >= half_dim) return;
    float angle = (float)(start+t) * pow(theta, -(2.0f*(float)i)/(float)dim);
    float c = cos(angle), s = sin(angle);
    if (h < q_heads) {
        size_t b = ((size_t)t*q_heads+h)*dim;
        float a = q[b+i], z = q[b+i+half_dim];
        qo[b+i] = a*c-z*s; qo[b+i+half_dim] = z*c+a*s;
    }
    if (h < kv_heads) {
        size_t b = ((size_t)t*kv_heads+h)*dim;
        size_t dst = ((size_t)(start+t)*kv_heads+h)*dim;
        float a = k[b+i], z = k[b+i+half_dim];
        kc[dst+i] = a*c-z*s; kc[dst+i+half_dim] = z*c+a*s;
        vc[dst+i] = v[b+i]; vc[dst+i+half_dim] = v[b+i+half_dim];
    }
}
