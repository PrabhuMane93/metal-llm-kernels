#include <metal_stdlib>
using namespace metal;

// Naive single-head attention — Rung A. Three separate dispatches, each
// the simplest possible version of its step: one thread per output
// element, no threadgroup memory, no tiling, no online softmax. S is
// fully materialized to device memory between stages — deliberately,
// since this kernel exists specifically to be the O(n^2)-memory baseline
// that Rung B (tiled, online-softmax) gets checked against and is built
// to avoid. See "Tiled attention, decoded" Principle 2 for the diagram
// this mirrors. Non-causal — causal masking is Rung C, added on top of
// the tiled kernel, not here.

kernel void qk_scaled_matmul(device const float* Q      [[buffer(0)]],
                              device const float* K      [[buffer(1)]],
                              device float*       S      [[buffer(2)]],
                              constant uint&      seqLen [[buffer(3)]],
                              constant uint&      dHead  [[buffer(4)]],
                              constant float&     scale  [[buffer(5)]],
                              uint2               gid    [[thread_position_in_grid]])
{
    uint i = gid.y; // query row
    uint j = gid.x; // key row
    if (i >= seqLen || j >= seqLen) return;

    float sum = 0.0;
    for (uint d = 0; d < dHead; d++) {
        sum += Q[i * dHead + d] * K[j * dHead + d];
    }
    S[i * seqLen + j] = sum * scale;
}

// One thread per row — no cooperative reduction. Reads its whole row
// three times (max, exp+sum, normalize), all from device memory. This
// is exactly the "seq_len^2 in device memory, softmax needs it all at
// once" story the tiled version's online softmax exists to avoid.
kernel void row_softmax(device float*  S      [[buffer(0)]],
                         constant uint& seqLen [[buffer(1)]],
                         uint           row    [[thread_position_in_grid]])
{
    if (row >= seqLen) return;
    device float* rowPtr = S + row * seqLen;

    float m = -INFINITY;
    for (uint j = 0; j < seqLen; j++) {
        m = max(m, rowPtr[j]);
    }

    float sum = 0.0;
    for (uint j = 0; j < seqLen; j++) {
        float e = exp(rowPtr[j] - m);
        rowPtr[j] = e;
        sum += e;
    }

    for (uint j = 0; j < seqLen; j++) {
        rowPtr[j] /= sum;
    }
}

kernel void av_matmul(device const float* A      [[buffer(0)]],
                       device const float* V      [[buffer(1)]],
                       device float*       O      [[buffer(2)]],
                       constant uint&      seqLen [[buffer(3)]],
                       constant uint&      dHead  [[buffer(4)]],
                       uint2               gid    [[thread_position_in_grid]])
{
    uint i = gid.y; // output row / query token
    uint k = gid.x; // output col / head dim
    if (i >= seqLen || k >= dHead) return;

    float sum = 0.0;
    for (uint j = 0; j < seqLen; j++) {
        sum += A[i * seqLen + j] * V[j * dHead + k];
    }
    O[i * dHead + k] = sum;
}
