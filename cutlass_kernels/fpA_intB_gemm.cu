#include "fpA_intB_gemm.h"
#include "fpA_intB_gemm/fpA_intB_gemm_template.h"

namespace fastertransformer {

void gemm_fp16_int4(const half*  A,
		    const uint4b_t* B,
		    const half* weight_scales,
		    half* C,
		    int m, int n, int k, char* workspace_ptr,
		    size_t workspace_bytes,
		    cudaStream_t stream) {
  CutlassFpAIntBGemmRunner<half, uint4b_t> runner;

  runner.gemm(A, B, weight_scales,
	      C, m, n, k, workspace_ptr, workspace_bytes, stream);
}

ActivationType get_activation(const std::string& activation_name) {
  if (activation_name == "identity") return ActivationType::Identity;
  if (activation_name == "relu") return ActivationType::Relu;
  if (activation_name == "silu") return ActivationType::Silu;
  // todo: more
  return ActivationType::Identity;
}

void gemm_fp16_int4_bias_act(const half*  A,
		    const uint4b_t* B,
		    const half* weight_scales,
		    const half* biases,
		    half* C,
     	            const std::string& activation,
		    int m, int n, int k, char* workspace_ptr,
		    size_t workspace_bytes,
		    cudaStream_t stream) {
  CutlassFpAIntBGemmRunner<half, uint4b_t> runner;

  runner.gemm_bias_act(A, B, weight_scales, biases,
		       C, m, n, k, get_activation(activation), workspace_ptr, workspace_bytes, stream);
}

void gemm_fp16_int4_bias(const half*  A,
		    const uint4b_t* B,
		    const half* weight_scales,
		    const half* biases,
		    half* C,
		    int m, int n, int k, char* workspace_ptr,
		    size_t workspace_bytes,
		    cudaStream_t stream) {
  gemm_fp16_int4_bias_act(A, B, weight_scales, biases, C, "identity", m, n, k, workspace_ptr, workspace_bytes, stream);
}

void gemm_fp16_int4_bias_act_residual(
    const half *A, const uint4b_t *B, const half *weight_scales,
    const half *biases, const half *residual, half *C, const std::string& activation, const std::string& binary_op,
    const std::string& unary_op, int m, int n,
    int k, char *workspace_ptr, size_t workspace_bytes, cudaStream_t stream) {
  CutlassFpAIntBGemmRunner<half, uint4b_t> runner;

  runner.gemm_bias_act_residual(A, B, weight_scales, biases, residual,
				C, m, n, k, activation, binary_op, unary_op, workspace_ptr, workspace_bytes, stream);

}


} // namespace fastertransformer
