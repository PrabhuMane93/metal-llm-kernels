#include <metal_stdlib>
using namespace metal;

kernel void gemv_naive(device const float* W       [[buffer(0)]],
                        device const float* x       [[buffer(1)]],
                        device float*       y       [[buffer(2)]],
                        constant uint&      numCols [[buffer(3)]],
                        uint                row     [[thread_position_in_grid]])
{
    float sum = 0.0;
    for (uint col = 0; col < numCols; col++) {
        sum += W[row * numCols + col] * x[col];
    }
    y[row] = sum;
}