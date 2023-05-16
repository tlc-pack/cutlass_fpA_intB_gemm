#include "fpA_intB_gemm.h"
#include "fpA_intB_gemm/fpA_intB_gemm_template.h"

namespace fastertransformer {

void gemm_fp16_int8(const half*  A,
		    const uint8_t* B,
		    const half* weight_scales,
		    half* C,
		    int m, int n, int k, char* workspace_ptr,
		    size_t workspace_bytes,
		    cudaStream_t stream) {
  CutlassFpAIntBGemmRunner<half, uint8_t> runner;

  runner.gemm(A, B, weight_scales,
	      C, m, n, k, workspace_ptr, workspace_bytes, stream);

}

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

} // namespace fastertransformer
