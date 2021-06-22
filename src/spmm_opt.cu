#include "spmm_opt.h"
#include <stdio.h>
#include <cuda.h>

const int STRIDE = 32;
const int BLOCK_X = 32;
const int BLOCK_Y = 8;
inline int ceil_div(int a, int b)
{
    return (a + b - 1) / b;
}

__global__ void spmm_kernel_notopt(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int INFEATURE)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_v) return;
    int begin = ptr[tid], end = ptr[tid + 1];
    for (int j = 0; j < INFEATURE; ++j)
    {
        float result = 0.0f;
        for (int i = begin; i < end; ++i)
        {
            result += vin[idx[i] * INFEATURE + j] * val[i];
        }
        vout[tid * INFEATURE + j] = result;
    }
}


__global__ void spmm_kernel_opt(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int feat_in)
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
    int begin = __ldg(ptr + y), end = __ldg(ptr + y + 1);

    float result = 0.f;
    __shared__ float val_temp[BLOCK_Y][STRIDE];
    __shared__ int col_temp[BLOCK_Y][STRIDE];
    float vin_temp[STRIDE];

    int ii;
    for (int i = begin; i < end; i += STRIDE)
    {
        ii = i + threadIdx.x;
        if (ii < end)
        {
            col_temp[threadIdx.y][threadIdx.x] = __ldg(idx + ii) * feat_in;
            val_temp[threadIdx.y][threadIdx.x] = __ldg(val + ii);
        }
        else
        {
            col_temp[threadIdx.y][threadIdx.x] = 0;
            val_temp[threadIdx.y][threadIdx.x] = 0;
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < STRIDE; ++j)
        {
            vin_temp[j] = __ldg(vin + col_temp[threadIdx.y][j] + x);
        }
#pragma unroll
        for (int j = 0; j < STRIDE; ++j)
        {
            result += val_temp[threadIdx.y][j] * vin_temp[j];
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
    block.x = STRIDE;
    block.y = BLOCK_Y;
    block.z = 1;

    // int BLOCK_SIZE = 128;
    // grid.x = (num_v + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // block.x = BLOCK_SIZE;
}

void SpMMOpt::run(float *vin, float *vout)
{
    // dbg("TODO");
    // printf("num_v = %d, feat_in = %d\n", num_v, feat_in);
    // printf("Grid = <%d, %d, %d>\n", grid.x, grid.y, grid.z);
    // printf("Block = <%d, %d, %d>\n", block.x, block.y, block.z);
    spmm_kernel_opt<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
    // spmm_kernel_notopt<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
}