#include <metal_stdlib>
using namespace metal;

// GEMM, Q4 dequant fused into the inner loop — the batched generalization
// of gemv_q4_fused.metal. Same W storage (block-64, asymmetric, 8 lanes
// per 32-bit word), same per-row dequant-and-accumulate logic; the only
// real change is a second grid dimension for tokens, so one dispatch
// computes Y = X @ W.T for every row of X at once instead of one x
// vector at a time. No threadgroup-memory tiling here either, matching
// the fused-only (not fused+tiled) GEMV 3 this was built from.
constant uint GROUP_SIZE = 64;
constant uint LANES_PER_WORD = 8;                      // 32 bits / 4 bits per lane
constant uint WORDS_PER_GROUP = GROUP_SIZE / LANES_PER_WORD; // 8

kernel void gemm_q4_fused(device const uint*  W_packed [[buffer(0)]],
                          device const half*  scales   [[buffer(1)]],
                          device const half*  biases   [[buffer(2)]],
                          device const float* X        [[buffer(3)]], // [seqLen, numCols], row-major
                          device float*       Y        [[buffer(4)]], // [seqLen, numRows], row-major
                          constant uint&      numCols  [[buffer(5)]],
                          constant uint&      numRows  [[buffer(6)]],
                          uint2               gid      [[thread_position_in_grid]])
{
    uint row   = gid.x; // which output feature, 0..numRows
    uint token = gid.y; // which row of X / row of Y, 0..seqLen

    uint numGroups   = numCols / GROUP_SIZE;
    uint wordsPerRow = numGroups * WORDS_PER_GROUP;

    device const float* x = X + token * numCols; // this token's input row

    float sum = 0.0;

    for (uint g = 0; g < numGroups; g++) {
        float scale = float(scales[row * numGroups + g]);
        float bias  = float(biases[row * numGroups + g]);

        for (uint w = 0; w < WORDS_PER_GROUP; w++) {
            uint word = W_packed[row * wordsPerRow + g * WORDS_PER_GROUP + w];

            for (uint lane = 0; lane < LANES_PER_WORD; lane++) {
                uint q = (word >> (lane * 4)) & 0xF;
                float weight = float(q) * scale + bias;
                uint col = g * GROUP_SIZE + w * LANES_PER_WORD + lane;
                sum += weight * x[col];
            }
        }
    }

    Y[token * numRows + row] = sum;
}
