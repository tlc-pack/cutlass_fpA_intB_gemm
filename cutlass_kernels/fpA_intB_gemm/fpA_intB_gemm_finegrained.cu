#include "./fpA_intB_gemm_impl.h"

namespace fastertransformer
{

template void gemm_fp16_int_bias_act<uint4b_t, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY>(const half* A,
    const uint4b_t* B, const half* weight_scales, const half* bias, half* C, std::optional<std::string> activation,
    int m, int n, int k, int group_size, int bias_stride, char* workspace_ptr, size_t workspace_bytes,
    cudaStream_t stream);

template void gemm_fp16_int_bias_act_residual<uint4b_t, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY>(
    const half* A, const uint4b_t* B, const half* weight_scales, const half* bias, const half* residual, half* C,
    const std::string& activation, const std::string& binary_op, const std::string& unary_op, int m, int n, int k,
    int group_size, char* workspace_ptr, size_t workspace_bytes, cudaStream_t stream);

template void gemm_fp16_int_bias_act<uint8_t, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY>(const half* A,
    const uint8_t* B, const half* weight_scales, const half* bias, half* C, std::optional<std::string> activation,
    int m, int n, int k, int group_size, int bias_stride, char* workspace_ptr, size_t workspace_bytes,
    cudaStream_t stream);

template void gemm_fp16_int_bias_act_residual<uint8_t, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY>(
    const half* A, const uint8_t* B, const half* weight_scales, const half* bias, const half* residual, half* C,
    const std::string& activation, const std::string& binary_op, const std::string& unary_op, int m, int n, int k,
    int group_size, char* workspace_ptr, size_t workspace_bytes, cudaStream_t stream);

} // namespace fastertransformer
