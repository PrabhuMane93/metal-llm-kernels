#include <metal_stdlib>
using namespace metal;

// GEMV 3 — Q4 dequant fused into the inner loop. W is stored packed:
// block-64, asymmetric (scale + bias, not a symmetric zero-point), 8
// lanes per 32-bit word (metal_lane_strided_q4). No threadgroup-memory
// tiling of x yet — this version isolates the dequant logic on its own,
// same one-thread-per-row structure as GEMV 1/2, before tiling gets
// layered back in.
constant uint GROUP_SIZE = 64;
constant uint LANES_PER_WORD = 8;                      // 32 bits / 4 bits per lane
constant uint WORDS_PER_GROUP = GROUP_SIZE / LANES_PER_WORD; // 8

kernel void gemv_q4_fused(device const uint*  W_packed [[buffer(0)]],
                          device const half*  scales   [[buffer(1)]],
                          device const half*  biases   [[buffer(2)]],
                          device const float* x        [[buffer(3)]],
                          device float*       y        [[buffer(4)]],
                          constant uint&      numCols  [[buffer(5)]],
                          uint                row      [[thread_position_in_grid]])
{
    uint numGroups   = numCols / GROUP_SIZE;           // 64
    uint wordsPerRow = numGroups * WORDS_PER_GROUP;    // 512

    float sum = 0.0;

    for (uint g = 0; g < numGroups; g++) {
        // Read once per block — shared by all 8 words / 64 lanes below.
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

    y[row] = sum;
}
