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

#include <cutlass_kernels/fpA_intB_gemm.h>
#include <cutlass_kernels/moe_gemm/moe_gemm_kernels.h>
#include <dlpack/dlpack.h>
#include <optional>
#include <string>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#define SWITCH_QUANT_OP(group_size, k, ...)                                                                            \
    if (group_size == k)                                                                                               \
    {                                                                                                                  \
        constexpr auto quant_op = cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY;                                   \
        __VA_ARGS__                                                                                                    \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        constexpr auto quant_op = cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY;                                  \
        __VA_ARGS__                                                                                                    \
    }

int _fastertransformer_gemm_fp16_int(DLTensor* x, DLTensor* weight, DLTensor* scale, std::string activation, int m,
    int n, int k, int group_size, DLTensor* output)
{
    CHECK_GT(group_size, 0);
    CHECK_LE(group_size, k);

    auto func = tvm::runtime::Registry::Get("runtime.get_cuda_stream");
    ICHECK(func != nullptr);
    cudaStream_t stream = static_cast<cudaStream_t>((*func)().operator void*());

    std::optional<std::string> activation_opt = activation;
    if (activation == "identity")
        activation_opt = std::nullopt;

    SWITCH_QUANT_OP(
        group_size, k,
        fastertransformer::gemm_fp16_int_bias_act<cutlass::uint4b_t, quant_op>(static_cast<cutlass::half_t*>(x->data),
            static_cast<cutlass::uint4b_t*>(weight->data), static_cast<cutlass::half_t*>(scale->data), nullptr,
            static_cast<cutlass::half_t*>(output->data), activation_opt, m, n, k, group_size, 0, nullptr, 0, stream););

    return 0;
}

int _fastertransformer_gemm_fp16_int_bias(DLTensor* x, DLTensor* weight, DLTensor* scale, DLTensor* bias,
    std::string activation, int m, int n, int k, int group_size, int bias_stride, DLTensor* output)
{
    CHECK_GT(group_size, 0);
    CHECK_LE(group_size, k);

    auto func = tvm::runtime::Registry::Get("runtime.get_cuda_stream");
    ICHECK(func != nullptr);
    cudaStream_t stream = static_cast<cudaStream_t>((*func)().operator void*());

    std::optional<std::string> activation_opt = activation;
    if (activation == "identity")
        activation_opt = std::nullopt;

    SWITCH_QUANT_OP(
        group_size, k,
        fastertransformer::gemm_fp16_int_bias_act<cutlass::uint4b_t, quant_op>(static_cast<cutlass::half_t*>(x->data),
            static_cast<cutlass::uint4b_t*>(weight->data), static_cast<cutlass::half_t*>(scale->data),
            static_cast<cutlass::half_t*>(bias->data), static_cast<cutlass::half_t*>(output->data), activation_opt, m,
            n, k, group_size, bias_stride, nullptr, 0, stream););

    return 0;
}

int _fastertransformer_gemm_fp16_int_bias_residual(DLTensor* x, DLTensor* weight, DLTensor* scale, DLTensor* bias,
    DLTensor* residual, std::string activation, std::string binary_op, std::string unary_op, int m, int n, int k,
    int group_size, DLTensor* output)
{
    CHECK_GT(group_size, 0);
    CHECK_LE(group_size, k);

    auto func = tvm::runtime::Registry::Get("runtime.get_cuda_stream");
    ICHECK(func != nullptr);
    cudaStream_t stream = static_cast<cudaStream_t>((*func)().operator void*());

    SWITCH_QUANT_OP(group_size, k,
                    fastertransformer::gemm_fp16_int_bias_act_residual<cutlass::uint4b_t, quant_op>(
                        static_cast<cutlass::half_t*>(x->data), static_cast<cutlass::uint4b_t*>(weight->data),
                        static_cast<cutlass::half_t*>(scale->data), static_cast<cutlass::half_t*>(bias->data),
                        static_cast<cutlass::half_t*>(residual->data), static_cast<cutlass::half_t*>(output->data),
                        activation, binary_op, unary_op, m, n, k, group_size, nullptr, 0, stream););

    return 0;
}

void _fastertransformer_moe_gemm_fp16_fp16(DLTensor* x, DLTensor* weight, DLTensor* total_rows_before_expert,
                                        int64_t total_rows, int64_t n, int64_t k, int num_experts,
                                        DLTensor* out) {
      auto func = tvm::runtime::Registry::Get("runtime.get_cuda_stream");
      ICHECK(func != nullptr);
      cudaStream_t stream = static_cast<cudaStream_t>((*func)().operator void*());

      fastertransformer::moe_gemm<half, half>(
          static_cast<half*>(x->data), static_cast<half*>(weight->data), nullptr, 
          static_cast<half*>(out->data),
          static_cast<int64_t*>(total_rows_before_expert->data), total_rows, n, k, num_experts,
          stream);
    };

void _fastertransformer_moe_gemm_fp16_int(DLTensor* x, DLTensor* weight, DLTensor* scale, DLTensor* total_rows_before_expert,
                                        int64_t total_rows, int64_t n, int64_t k, int num_experts, int group_size,
                                        DLTensor* out) {
      auto func = tvm::runtime::Registry::Get("runtime.get_cuda_stream");
      ICHECK(func != nullptr);
      cudaStream_t stream = static_cast<cudaStream_t>((*func)().operator void*());

      ICHECK(group_size == k) << "group quantization not supported yet";

      fastertransformer::moe_gemm<half, cutlass::uint4b_t>(
          static_cast<half*>(x->data), static_cast<cutlass::uint4b_t*>(weight->data), static_cast<half*>(scale->data), 
          static_cast<half*>(out->data),
          static_cast<int64_t*>(total_rows_before_expert->data), total_rows, n, k, num_experts,
          stream);
    };

TVM_REGISTER_GLOBAL("fastertransformer.gemm_fp16_int").set_body_typed(_fastertransformer_gemm_fp16_int);
TVM_REGISTER_GLOBAL("fastertransformer.gemm_fp16_int_bias").set_body_typed(_fastertransformer_gemm_fp16_int_bias);
TVM_REGISTER_GLOBAL("fastertransformer.gemm_fp16_int_bias_residual")
    .set_body_typed(_fastertransformer_gemm_fp16_int_bias_residual);
TVM_REGISTER_GLOBAL("fastertransformer.moe_gemm_fp16_fp16").set_body_typed(_fastertransformer_moe_gemm_fp16_fp16);
TVM_REGISTER_GLOBAL("fastertransformer.moe_gemm_fp16_int").set_body_typed(_fastertransformer_moe_gemm_fp16_int);
