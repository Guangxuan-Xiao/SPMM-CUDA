#include "spmm_opt.h"
#include <stdio.h>

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
            result += vin[idx[i] + j * num_v] * val[i];
        }
        vout[tid * feat_in + j] = result;
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
    // printf("Grid = <%d, %d, %d>\n", grid.x, grid.y, grid.z);
    // printf("Block = <%d, %d, %d>\n", block.x, block.y, block.z);
    float *new_vin;
    cudaMalloc(&new_vin, feat_in * num_v * sizeof(float));
    cublasCreate(&handle);
    float alpha = 1, beta = 0;
    cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, num_v, feat_in, &alpha, vin, feat_in,
                        &beta,
                        nullptr, num_v,
                        new_vin, num_v);
    spmm_kernel_opt<<<grid, block>>>(d_ptr, d_idx, d_val, new_vin, vout, num_v, feat_in);
}