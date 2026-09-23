#include <metal_stdlib>
using namespace metal;

// Tiled single-head attention with online softmax — Rung B. Same math as
// Rung A (attention_naive), computed without ever materializing the full
// (seq_len, seq_len) score matrix: one threadgroup owns one Br-row block of
// Q for its whole lifetime, and loops over Bc-row blocks of K/V, folding
// each block's contribution into a running max/sum (softmax) and a running
// weighted sum of V (the output) as it goes. Non-causal — causal masking is
// Rung C, layered on top of this same kernel, not here.
//
// Br = Bc = 4 chosen so threads_per_threadgroup = Br * D_HEAD = 512, safely
// under Metal's 1024-thread threadgroup cap while keeping Br and Bc equal —
// that equality is what lets the same tid -> (row, feat) split load Q, K,
// and V tiles alike, and fold the output, all without a second mapping.
kernel void flash_attention(device const float* Q     [[buffer(0)]],
                             device const float* K     [[buffer(1)]],
                             device const float* V     [[buffer(2)]],
                             device float*       O     [[buffer(3)]],
                             constant float&     scale [[buffer(4)]],
                             uint                tgId  [[threadgroup_position_in_grid]],
                             uint                tid   [[thread_position_in_threadgroup]])
{
    constexpr uint SEQ_LEN = 128;
    constexpr uint D_HEAD  = 128;
    constexpr uint BR = 4;
    constexpr uint BC = 4;
    constexpr uint NUM_KV_BLOCKS = SEQ_LEN / BC;

    threadgroup float Q_tile[BR * D_HEAD];
    threadgroup float K_tile[BC * D_HEAD];
    threadgroup float V_tile[BC * D_HEAD];
    threadgroup float S_tile[BR * BC]; // this block's scores, overwritten in place with exp'd weights
    threadgroup float m_row[BR];       // running row max, carried across the whole j loop
    threadgroup float l_row[BR];       // running row sum of exp, carried across the whole j loop
    threadgroup float corr[BR];        // this iteration's rescale factor for the O accumulator

    // Valid for every thread: BR * D_HEAD == 512 == threads_per_threadgroup,
    // so this split covers all 512 threads with none left over.
    uint r    = tid / D_HEAD; // row within this Q/O block, 0..BR-1
    uint feat = tid % D_HEAD; // feature/column, 0..D_HEAD-1
    uint qRow = tgId * BR + r; // this thread's actual row in the full (seq_len, d_head) matrix

    Q_tile[tid] = Q[qRow * D_HEAD + feat];
    if (tid < BR) {
        m_row[tid] = -INFINITY;
        l_row[tid] = 0.0;
    }
    float o_acc = 0.0; // O[qRow][feat], accumulated across every K/V block — a plain register,
                        // never read by another thread, so it never needs threadgroup memory.

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint j = 0; j < NUM_KV_BLOCKS; j++) {
        // Br == Bc, so the same (r, feat) split used for Q above also
        // covers this K/V block's rows exactly.
        uint kvRow = j * BC + r;
        K_tile[tid] = K[kvRow * D_HEAD + feat];
        V_tile[tid] = V[kvRow * D_HEAD + feat];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // S_ij = Q_i @ K_j^T * scale — one thread per (row, col) output,
        // full D_HEAD-length dot product, same "one thread, serial
        // reduction" shape as Rung A's qk_scaled_matmul. Only BR*BC=16 of
        // the 512 threads do work here; the rest sit out this phase.
        if (tid < BR * BC) {
            uint sr = tid / BC;
            uint sc = tid % BC;
            float dot = 0.0;
            for (uint d = 0; d < D_HEAD; d++) {
                dot += Q_tile[sr * D_HEAD + d] * K_tile[sc * D_HEAD + d];
            }
            S_tile[sr * BC + sc] = dot * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax update — one thread per row (only BR=4 active),
        // same "one thread, serial passes over its row" shape as Rung A's
        // row_softmax, just over this block's BC columns instead of the
        // whole seq_len row, and folding the running max/sum forward
        // instead of finishing them in one pass.
        if (tid < BR) {
            uint row = tid;
            float m_block = -INFINITY;
            for (uint c = 0; c < BC; c++) {
                m_block = max(m_block, S_tile[row * BC + c]);
            }
            float m_new = max(m_row[row], m_block);
            float correction = exp(m_row[row] - m_new); // exp(-inf - finite) = 0 on the first block, as intended

            float l_block = 0.0;
            for (uint c = 0; c < BC; c++) {
                float p = exp(S_tile[row * BC + c] - m_new);
                S_tile[row * BC + c] = p; // reuse S_tile in place, same "overwrite the buffer" habit as Rung A
                l_block += p;
            }

            m_row[row] = m_new;
            l_row[row] = l_row[row] * correction + l_block;
            corr[row]  = correction;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Fold this block into every thread's own O entry: rescale what's
        // already accumulated, then add this block's P @ V contribution.
        // All 512 threads active again here (same split as the Q/K/V loads).
        float pv = 0.0;
        for (uint c = 0; c < BC; c++) {
            pv += S_tile[r * BC + c] * V_tile[c * D_HEAD + feat];
        }
        o_acc = o_acc * corr[r] + pv;

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    O[qRow * D_HEAD + feat] = o_acc / l_row[r];
}
