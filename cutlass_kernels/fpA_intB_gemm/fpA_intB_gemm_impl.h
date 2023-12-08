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

#pragma once

#include "../fpA_intB_gemm.h"
#include "./fpA_intB_gemm_template.h"
#include "weightOnlyBatchedGemv/kernelLauncher.h"

#include <type_traits>

namespace fastertransformer
{

template <typename WeightType, cutlass::WeightOnlyQuantOp QuantOp>
void dispatch_to_weight_only_batched_gemv(const half* A, const WeightType* B, const half* weight_scales,
    const half* bias, half* C, std::optional<std::string> activation, int m, int n, int k, int group_size,
    int bias_stride, char* workspace_ptr, size_t workspace_bytes, cudaStream_t stream)
{
    using namespace tensorrt_llm::kernels;

    // Convert WeightType
    WeightOnlyQuantType weight_type
        = std::is_same_v<WeightType, uint4b_t> ? WeightOnlyQuantType::Int4b : WeightOnlyQuantType::Int8b;

    // Convert QuantType
    WeightOnlyType quant_type = QuantOp == cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY
        ? WeightOnlyType::PerChannel
        : WeightOnlyType::GroupWise;

    // Convert Activation
    WeightOnlyActivationType weight_only_activation = WeightOnlyActivationType::Identity;
    {
        const std::unordered_map<ActivationType, WeightOnlyActivationType> activation_mapping{
            {ActivationType::Identity, WeightOnlyActivationType::Identity},
            {ActivationType::Relu, WeightOnlyActivationType::Relu},
            // {ActivationType::Silu, WeightOnlyActivationType::Silu},  // TODO: add silu
            {ActivationType::Gelu, WeightOnlyActivationType::Gelu},
        };
        if (activation.has_value())
        {
            weight_only_activation = activation_mapping.at(get_activation(*activation));
        }
    }

    WeightOnlyParams params{
        /*qweight=*/reinterpret_cast<const uint8_t*>(B),
        /*scales=*/reinterpret_cast<const ::half*>(weight_scales),
        /*zeros=*/nullptr,
        /*in=*/reinterpret_cast<const ::half*>(A),
        /*bias=*/reinterpret_cast<const ::half*>(bias),
        /*out=*/reinterpret_cast<::half*>(C),
        m,
        n,
        k,
        group_size,
        bias_stride,
    };

    weight_only_batched_gemv_launcher(weight_type, quant_type, weight_only_activation, params, stream);
}

template <typename WeightType, cutlass::WeightOnlyQuantOp QuantOp>
void gemm_fp16_int_bias_act(const half* A, const WeightType* B, const half* weight_scales, const half* bias, half* C,
    std::optional<std::string> activation, int m, int n, int k, int group_size, int bias_stride, char* workspace_ptr,
    size_t workspace_bytes, cudaStream_t stream)
{
    if (m <= 4 && (!activation.has_value() || activation.value() != "silu"))
    {
        dispatch_to_weight_only_batched_gemv<WeightType, QuantOp>(A, B, weight_scales, bias, C, activation, m, n, k,
            group_size, bias_stride, workspace_ptr, workspace_bytes, stream);
        return;
    }
    CutlassFpAIntBGemmRunner<half, WeightType, QuantOp> runner;

    if (!activation && bias == nullptr)
    {
        runner.gemm(A, B, weight_scales, C, m, n, k, group_size, workspace_ptr, workspace_bytes, stream);
    }
    else if (!activation)
    {
        runner.gemm_bias_act(A, B, weight_scales, bias, C, m, n, k, group_size, bias_stride, ActivationType::Identity,
            workspace_ptr, workspace_bytes, stream);
    }
    else
    {
        runner.gemm_bias_act(A, B, weight_scales, bias, C, m, n, k, group_size, bias_stride,
            get_activation(*activation), workspace_ptr, workspace_bytes, stream);
    }
}

template <typename WeightType, cutlass::WeightOnlyQuantOp QuantOp>
void gemm_fp16_int_bias_act_residual(const half* A, const WeightType* B, const half* weight_scales, const half* bias,
    const half* residual, half* C, const std::string& activation, const std::string& binary_op,
    const std::string& unary_op, int m, int n, int k, int group_size, char* workspace_ptr, size_t workspace_bytes,
    cudaStream_t stream)
{
    CutlassFpAIntBGemmRunner<half, WeightType, QuantOp> runner;

    runner.gemm_bias_act_residual(A, B, weight_scales, bias, residual, C, m, n, k, group_size, activation, binary_op,
        unary_op, workspace_ptr, workspace_bytes, stream);
}

} // namespace fastertransformer
