#include "spmm_opt.h"
#include <stdio.h>
__global__ void spmm_kernel_opt(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int INFEATURE)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_v)
        return;
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

void SpMMOpt::preprocess(float *vin, float *vout)
{
    // dbg("TODO");
    int BLOCK_SIZE = 1024;
    grid.x = (num_v + BLOCK_SIZE - 1) / BLOCK_SIZE;
    block.x = BLOCK_SIZE;
}

void SpMMOpt::run(float *vin, float *vout)
{
    // dbg("TODO");
    printf("Grid = <%d, %d, %d>\n", grid.x, grid.y, grid.z);
    printf("Block = <%d, %d, %d>\n", block.x, block.y, block.z);
    spmm_kernel_opt<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
}