#include "spmm_opt.h"
#include <stdio.h>
const int BLOCK_X = 16;
const int BLOCK_Y = 32;
const int NUM_THREADS = BLOCK_X * BLOCK_Y;

inline int ceil_div(int a, int b)
{
    return (a + b - 1) / b;
}

__global__ void spmm_kernel_opt(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int feat_in)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int x = tid / BLOCK_Y;
    if (x >= num_v)
        return;
    int lane_id = tid & (BLOCK_Y - 1);
    int y = blockIdx.y * BLOCK_Y + lane_id;
    int out_idx = x * feat_in + y;
    const float *vin_offset = vin + y;

    int begin = __ldg(ptr + x), end = __ldg(ptr + x + 1);

    float result = 0.f, v = 0.f;
    float val_temp[BLOCK_Y];
    float mul_temp[BLOCK_Y];
    int col_temp[BLOCK_Y];

    int ii, col;
    for (int i = begin; i < end; i += BLOCK_Y)
    {
        ii = i + lane_id;
        if (ii < end)
        {
            col = __ldg(idx + ii) * feat_in;
            v = __ldg(val + ii);
        }
        else
        {
            col = 0;
            v = 0;
        }
#pragma unroll
        for (int j = 0; j < BLOCK_Y; ++j)
        {
            col_temp[j] = __shfl_sync(0xFFFFFFFF, col, j);
            val_temp[j] = __shfl_sync(0xFFFFFFFF, v, j);
            mul_temp[j] = val_temp[j] * __ldg(vin_offset + col_temp[j]);
        }
#pragma unroll
        for (int j = 0; j < BLOCK_Y; ++j)
        {
            result += mul_temp[j];
        }
    }
    vout[out_idx] = result;
}

void SpMMOpt::preprocess(float *vin, float *vout)
{
    grid.x = ceil_div(num_v, BLOCK_X);
    grid.y = ceil_div(feat_in, BLOCK_Y);
    grid.z = 1;
    block.x = NUM_THREADS;
    block.y = 1;
    block.z = 1;
}

void SpMMOpt::run(float *vin, float *vout)
{
    spmm_kernel_opt<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
}