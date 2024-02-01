/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "./moe_gemm_kernels_template.h"

namespace fastertransformer
{

template void moe_gemm(const half* A, const cutlass::uint4b_t* B, const half* weight_scales, half* C,
    int64_t* total_rows_before_expert, int64_t total_rows, int64_t gemm_n, int64_t gemm_k, int num_experts,
    cudaStream_t stream);

// unused. commented out to speed up compilation
// template
// void moe_gemm_bias_act(const half* A, const cutlass::uint4b_t* B, const half* weight_scales, const half* biases,
//     half* C, int64_t* total_rows_before_expert, int64_t total_rows, int64_t gemm_n, int64_t gemm_k, int num_experts,
//     std::optional<std::string> activation, cudaStream_t stream);

} // namespace fastertransformer
