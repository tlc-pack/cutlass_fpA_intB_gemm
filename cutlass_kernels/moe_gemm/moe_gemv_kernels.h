/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include "weightOnlyBatchedGemv/kernel.h"

namespace tensorrt_llm
{
namespace kernels
{

template <WeightOnlyQuantType QType, typename WeightOnlyFlag, template <typename T> class ActOp, bool Zero, bool Bias,
    int NPerBlock, int Batch, int BlockSize>
__global__ void moe_weight_only_batched_gemv(const uint8_t* qweight, const half* scales, const half* zeros,
    const half* in, const half* bias, half* out, int64_t* total_rows_before_expert, int64_t total_rows, int64_t n,
    int64_t k, int num_experts)
{
    static_assert(!Zero && !Bias, "Not implemented");
    static_assert(NPerBlock == 1 || (NPerBlock % 2 == 0));
    int bias_stride = 0;
    using Details = WeightOnlyKernelDetails<QType>;

    extern __shared__ uint8_t shmem[];

    const int gid = blockIdx.y;
    for (int i = 0; i < num_experts; i++)
    {
        if (total_rows_before_expert[i] >= (gid + 1))
        {
            qweight += i * n * k / Details::kElemsPerByte;
            scales += i * n;
            in += gid * k;
            out += gid * n;
            break;
        }
    }

    weight_only_batched_gemv_impl<QType, WeightOnlyFlag, ActOp, Zero, Bias, NPerBlock, Batch, BlockSize>(
        qweight, scales, zeros, in, bias, out, n, k, bias_stride, shmem);
}

template <WeightOnlyQuantType QType, typename WeightOnlyFlag>
void moe_gemv(const half* A, const uint8_t* B, const half* weight_scales, half* C, int64_t* total_rows_before_expert,
    int64_t total_rows, int64_t gemm_n, int64_t gemm_k, int num_experts, cudaStream_t stream)
{
    static constexpr int kInterleave = WeightLayoutDetails<QType>::kInterleave;
    constexpr int Batch = 1;
    // TODO: tuning the config
    constexpr int BlockSize = 128;
    constexpr int NPerBlock = 1;
    dim3 grid(gemm_n / NPerBlock / kInterleave, total_rows);
    dim3 block(BlockSize);
    int size = sizeof(float) * BlockSize / 32 * Batch * NPerBlock * kInterleave;
    moe_weight_only_batched_gemv<QType, WeightOnlyFlag, IdentityActivation, false, false, NPerBlock, Batch, BlockSize>
        <<<grid, block, size, stream>>>(B, weight_scales, nullptr, A, nullptr, C, total_rows_before_expert, total_rows,
            gemm_n, gemm_k, num_experts);
}

void moe_gemv(const half* A, const half* B, half* C, int64_t* total_rows_before_expert, int64_t total_rows,
    int64_t gemm_n, int64_t gemm_k, int num_experts, cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm
