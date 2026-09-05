#include <metal_stdlib>
using namespace metal;

// GEMV 2 — tiled, using threadgroup memory so all 256 threads in a group
// cooperatively load and share one cached tile of x, instead of each thread
// independently re-reading the whole vector from device memory itself.
kernel void gemv_tiled(device const float* W       [[buffer(0)]],
                       device const float* x       [[buffer(1)]],
                       device float*       y       [[buffer(2)]],
                       constant uint&      numCols [[buffer(3)]],
                       uint                row     [[thread_position_in_grid]],
                       uint                tid     [[thread_position_in_threadgroup]])
{
    constexpr uint TILE_SIZE = 256;
    threadgroup float tile[TILE_SIZE];

    float sum = 0.0;
    uint numTiles = numCols / TILE_SIZE;

    for (uint t = 0; t < numTiles; t++) {
        tile[tid] = x[t * TILE_SIZE + tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint c = 0; c < TILE_SIZE; c++) {
            sum += W[row * numCols + t * TILE_SIZE + c] * tile[c];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    y[row] = sum;
}
