#include <metal_stdlib>
using namespace metal;

// Fused SiLU — one dispatch, two accumulators. GEMV 3's exact block-64,
// asymmetric, lane-strided Q4 unpack loop, factored into `dot_q4_dequant`
// so it's literally the same loop called twice (once for W_gate, once for
// W_up) against the same x — not a second, separately-written copy. Both
// dot products land in registers, SiLU + multiply happen there too; only
// the combined result h ever reaches global memory. See "Fused SiLU,
// decoded" for the full derivation of why this shape, not two kernels.
constant uint GROUP_SIZE = 64;
constant uint LANES_PER_WORD = 8;                      // 32 bits / 4 bits per lane
constant uint WORDS_PER_GROUP = GROUP_SIZE / LANES_PER_WORD; // 8

inline float dot_q4_dequant(device const uint*  W_packed,
                             device const half*  scales,
                             device const half*  biases,
                             device const float* x,
                             uint                row,
                             uint                numGroups,
                             uint                wordsPerRow)
{
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

    return sum;
}

kernel void gemv_silu_fused(device const uint*  W_gate_packed  [[buffer(0)]],
                             device const half*  gate_scales   [[buffer(1)]],
                             device const half*  gate_biases   [[buffer(2)]],
                             device const uint*  W_up_packed   [[buffer(3)]],
                             device const half*  up_scales     [[buffer(4)]],
                             device const half*  up_biases     [[buffer(5)]],
                             device const float* x             [[buffer(6)]],
                             device float*       h             [[buffer(7)]],
                             constant uint&      numCols       [[buffer(8)]],
                             uint                row           [[thread_position_in_grid]])
{
    uint numGroups   = numCols / GROUP_SIZE;
    uint wordsPerRow = numGroups * WORDS_PER_GROUP;

    float gate = dot_q4_dequant(W_gate_packed, gate_scales, gate_biases, x, row, numGroups, wordsPerRow);
    float up   = dot_q4_dequant(W_up_packed,   up_scales,   up_biases,   x, row, numGroups, wordsPerRow);

    // SiLU(gate) = gate * sigmoid(gate) — no built-in in MSL, three ops.
    float silu_gate = gate / (1.0f + exp(-gate));
    h[row] = silu_gate * up;
}
