#include "spmm_opt.h"
#include <stdio.h>
#include <cuda.h>

const int BLOCK_X = 32;
const int BLOCK_Y = 32;

inline int ceil_div(int a, int b)
{
    return (a + b - 1) / b;
}

__global__ void spmm_kernel_opt(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int feat_in)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_v)
        return;
    int begin = ptr[tid], end = ptr[tid + 1];
    for (int j = 0; j < feat_in; ++j)
    {
        float result = 0.0f;
        for (int i = begin; i < end; ++i)
        {
            // Transposing the vin maybe cache-friendly
            result += vin[idx[i] * feat_in + j] * val[i];
        }
        vout[tid * feat_in + j] = result;
    }
}

__global__ void spmm_kernel_merge(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int feat_in)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    // if (x == 0 && y == 0)
    // {
    //     printf("gridDim = <%d, %d, %d>\n", gridDim.x, gridDim.y, gridDim.z);
    //     printf("blockDim = <%d, %d, %d>\n", blockDim.x, blockDim.y, blockDim.z);
    // }
    if (x >= feat_in || y >= num_v)
        return;

    int out_idx = y * feat_in + x;
    vin += x;
    int begin = __ldg(ptr + y), end = __ldg(ptr + y + 1);

    float result = 0.f, v;
    float val_temp[BLOCK_X];
    float mul_temp[BLOCK_X];
    int col_temp[BLOCK_X];

    int ii, col;
    for (int i = begin; i < end; i += BLOCK_X)
    {
        ii = i + threadIdx.x;
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
        for (int j = 0; j < BLOCK_X; ++j)
        {
            col_temp[j] = __shfl_sync(0xFFFFFFFF, col, j);
            val_temp[j] = __shfl_sync(0xFFFFFFFF, v, j);
            mul_temp[j] = val_temp[j] * __ldg(vin + col_temp[j]);
        }
#pragma unroll
        for (int j = 0; j < BLOCK_X; ++j)
        {
            result += mul_temp[j];
        }
    }
    vout[out_idx] = result;
}

void SpMMOpt::preprocess(float *vin, float *vout)
{
    // dbg("TODO");
    grid.x = ceil_div(feat_in, BLOCK_X);
    grid.y = ceil_div(num_v, BLOCK_Y);
    grid.z = 1;
    block.x = BLOCK_X;
    block.y = BLOCK_Y;
    block.z = 1;
}

void SpMMOpt::run(float *vin, float *vout)
{
    // dbg("TODO");
    // spmm_kernel_opt<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
    // printf("num_v = %d, feat_in = %d\n", num_v, feat_in);
    // printf("Grid = <%d, %d, %d>\n", grid.x, grid.y, grid.z);
    // printf("Block = <%d, %d, %d>\n", block.x, block.y, block.z);
    spmm_kernel_merge<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
}