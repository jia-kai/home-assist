/* Auxiliary batch-one LFM kernels. Host validates all sizes and token IDs. */
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void lfm_embedding(const __global int *ids, const __global uchar *q,
                            const __global uchar *zero, const __global float *scale,
                            __global float *out, int count, int width) {
    int i = get_global_id(0), t = i / width, d = i % width;
    if (t >= count) return;
    int token = ids[t];
    float v = ((float)q[(size_t)token * width + d] - (float)zero[token]) * scale[token];
    volatile half rounded = convert_half_rte(v);
    out[i] = convert_float(rounded);
}
__kernel void lfm_conv_split(const __global float *in, __global float *bx,
                              __global float *c, int count, int width) {
    int i = get_global_id(0), t = i / width, d = i % width;
    if (t >= count) return;
    size_t base = (size_t)t * 3 * width + d;
    bx[i] = in[base] * in[base + 2 * width];
    c[i] = in[base + width];
}
__kernel void lfm_add(__global float *a, const __global float *b, int n) {
    int i = get_global_id(0); if (i < n) a[i] += b[i];
}
__kernel void lfm_swiglu(__global float *a, const __global float *b, int n) {
    int i = get_global_id(0); if (i < n) { float x = a[i]; a[i] = (x / (1.0f + exp(-x))) * b[i]; }
}
__kernel void lfm_last(const __global float *x, __global float *out, int rows, int width) {
    int i = get_global_id(0); if (i < width) out[i] = x[(size_t)(rows-1)*width+i];
}
