#pragma once

#include <optional>
#include <string>

#include "cutlass/half.h"
// clang-format off
#include "cutlass/numeric_types.h"
#include "cutlass/integer_subbyte.h"
// clang-format on
#include "cutlass_extensions/include/cutlass_extensions/weight_only_quant_op.h"
#include <cuda_runtime.h>

#include "./cutlass_preprocessors.h"

namespace fastertransformer
{

using half = cutlass::half_t;
using uint4b_t = cutlass::uint4b_t;

template <typename WeightType, cutlass::WeightOnlyQuantOp QuantOp>
void gemm_fp16_int_bias_act(const half* A, const WeightType* B, const half* weight_scales, const half* bias, half* C,
    std::optional<std::string> activation, int m, int n, int k, int group_size, int bias_stride, char* workspace_ptr,
    size_t workspace_bytes, cudaStream_t stream);

template <typename WeightType, cutlass::WeightOnlyQuantOp QuantOp>
void gemm_fp16_int_bias_act_residual(const half* A, const WeightType* B, const half* weight_scales, const half* bias,
    const half* residual, half* C, const std::string& activation, const std::string& binary_op,
    const std::string& unary_op, int m, int n, int k, int group_size, char* workspace_ptr, size_t workspace_bytes,
    cudaStream_t stream);

} // namespace fastertransformer
