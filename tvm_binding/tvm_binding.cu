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

int _fastertransformer_gemm_fp16_int(
    DLTensor* x, DLTensor* weight, DLTensor* scale, int m, int n, int k, int group_size, DLTensor* output)
{
    CHECK_GT(group_size, 0);
    CHECK_LE(group_size, k);

    auto func = tvm::runtime::Registry::Get("runtime.get_cuda_stream");
    ICHECK(func != nullptr);
    cudaStream_t stream = static_cast<cudaStream_t>((*func)().operator void*());

    SWITCH_QUANT_OP(
        group_size, k,
        fastertransformer::gemm_fp16_int_bias_act<cutlass::uint4b_t, quant_op>(static_cast<cutlass::half_t*>(x->data),
            static_cast<cutlass::uint4b_t*>(weight->data), static_cast<cutlass::half_t*>(scale->data), nullptr,
            static_cast<cutlass::half_t*>(output->data), std::nullopt, m, n, k, group_size, 0, nullptr, 0, stream););

    return 0;
}

int _fastertransformer_gemm_fp16_int_bias(DLTensor* x, DLTensor* weight, DLTensor* scale, DLTensor* bias, int m, int n,
    int k, int group_size, int bias_stride, DLTensor* output)
{
    CHECK_GT(group_size, 0);
    CHECK_LE(group_size, k);

    auto func = tvm::runtime::Registry::Get("runtime.get_cuda_stream");
    ICHECK(func != nullptr);
    cudaStream_t stream = static_cast<cudaStream_t>((*func)().operator void*());

    SWITCH_QUANT_OP(
        group_size, k,
        fastertransformer::gemm_fp16_int_bias_act<cutlass::uint4b_t, quant_op>(static_cast<cutlass::half_t*>(x->data),
            static_cast<cutlass::uint4b_t*>(weight->data), static_cast<cutlass::half_t*>(scale->data),
            static_cast<cutlass::half_t*>(bias->data), static_cast<cutlass::half_t*>(output->data), std::nullopt, m, n,
            k, group_size, bias_stride, nullptr, 0, stream););

    return 0;
}

TVM_REGISTER_GLOBAL("fastertransformer.gemm_fp16_int").set_body_typed(_fastertransformer_gemm_fp16_int);
TVM_REGISTER_GLOBAL("fastertransformer.gemm_fp16_int_bias").set_body_typed(_fastertransformer_gemm_fp16_int_bias);

TVM_REGISTER_GLOBAL("fastertransformer.preprocess_weights")
    .set_body_typed(
        [](DLTensor* packed_weight, int sm, bool is_int4, DLTensor* output)
        {
            bool is_2d = packed_weight->ndim == 2;
            int num_experts = is_2d ? 1 : packed_weight->shape[0];
            int rows = packed_weight->shape[is_2d ? 0 : 1];
            int cols = packed_weight->shape[is_2d ? 1 : 2];
            // multiply cols by 2 since the "col" params in preprocess_weights refers to the column of
            // the unpacked weight.
            if (is_int4)
            {
                cols *= 2;
            }
            fastertransformer::preprocess_weights(static_cast<int8_t*>(output->data),
                static_cast<int8_t*>(packed_weight->data), num_experts, rows, cols, is_int4, sm);
            return 0;
        });

template <typename WeightType>
int _moe_gemm_fp16(DLTensor* x, DLTensor* weight, DLTensor* scale, DLTensor* bias, DLTensor* total_rows_before_expert,
    int64_t total_rows, int64_t n, int64_t k, int64_t num_experts, DLTensor* output)
{
    auto func = tvm::runtime::Registry::Get("runtime.get_cuda_stream");
    ICHECK(func != nullptr);
    cudaStream_t stream = static_cast<cudaStream_t>((*func)().operator void*());

    fastertransformer::moe_gemm_bias_act<cutlass::half_t, cutlass::half_t>(static_cast<cutlass::half_t*>(x->data),
        static_cast<WeightType*>(weight->data), scale == nullptr ? nullptr : static_cast<cutlass::half_t*>(scale->data),
        bias == nullptr ? nullptr : static_cast<cutlass::half_t*>(bias->data),
        static_cast<cutlass::half_t*>(output->data), static_cast<int64_t*>(total_rows_before_expert->data), total_rows,
        n, k, num_experts, std::nullopt, stream);
}

TVM_REGISTER_GLOBAL("fastertransformer.moe_gemm_fp16_fp16")
    .set_body_typed(
        [](DLTensor* x, DLTensor* weight, DLTensor* total_rows_before_expert, int64_t total_rows, int64_t n, int64_t k,
            int64_t num_experts, DLTensor* output)
        {
            _moe_gemm_fp16<cutlass::half_t>(
                x, weight, nullptr, nullptr, total_rows_before_expert, total_rows, n, k, num_experts, output);
        });

TVM_REGISTER_GLOBAL("fastertransformer.moe_gemm_fp16_fp16_bias")
    .set_body_typed(
        [](DLTensor* x, DLTensor* weight, DLTensor* bias, DLTensor* total_rows_before_expert, int64_t total_rows,
            int64_t n, int64_t k, int64_t num_experts, DLTensor* output)
        {
            _moe_gemm_fp16<cutlass::half_t>(
                x, weight, nullptr, bias, total_rows_before_expert, total_rows, n, k, num_experts, output);
        });

TVM_REGISTER_GLOBAL("fastertransformer.moe_gemm_fp16_uint4")
    .set_body_typed(
        [](DLTensor* x, DLTensor* weight, DLTensor* scale, DLTensor* total_rows_before_expert, int64_t total_rows,
            int64_t n, int64_t k, int64_t num_experts, DLTensor* output)
        {
            _moe_gemm_fp16<cutlass::uint4b_t>(
                x, weight, scale, nullptr, total_rows_before_expert, total_rows, n, k, num_experts, output);
        });

TVM_REGISTER_GLOBAL("fastertransformer.moe_gemm_fp16_uint4_bias")
    .set_body_typed(
        [](DLTensor* x, DLTensor* weight, DLTensor* scale, DLTensor* bias, DLTensor* total_rows_before_expert,
            int64_t total_rows, int64_t n, int64_t k, int64_t num_experts, DLTensor* output)
        {
            _moe_gemm_fp16<cutlass::uint4b_t>(
                x, weight, scale, bias, total_rows_before_expert, total_rows, n, k, num_experts, output);
        });

TVM_REGISTER_GLOBAL("fastertransformer.moe_gemm_fp16_uint8")
    .set_body_typed(
        [](DLTensor* x, DLTensor* weight, DLTensor* scale, DLTensor* total_rows_before_expert, int64_t total_rows,
            int64_t n, int64_t k, int64_t num_experts, DLTensor* output)
        {
            _moe_gemm_fp16<uint8_t>(
                x, weight, scale, nullptr, total_rows_before_expert, total_rows, n, k, num_experts, output);
        });

TVM_REGISTER_GLOBAL("fastertransformer.moe_gemm_fp16_uint8_bias")
    .set_body_typed(
        [](DLTensor* x, DLTensor* weight, DLTensor* scale, DLTensor* bias, DLTensor* total_rows_before_expert,
            int64_t total_rows, int64_t n, int64_t k, int64_t num_experts, DLTensor* output) {
            _moe_gemm_fp16<uint8_t>(
                x, weight, scale, bias, total_rows_before_expert, total_rows, n, k, num_experts, output);
        });