#pragma once

#include <cuda_runtime.h>
#include "cutlass/numeric_types.h"
#include "cutlass/half.h"
#include "cutlass/integer_subbyte.h"

namespace fastertransformer {

using half = cutlass::half_t;
using uint4b_t = cutlass::uint4b_t;

void gemm_fp16_int8(const half*  A,
		    const uint8_t* B,
		    const half* weight_scales,
		    half* C,
		    int m, int n, int k, char* workspace_ptr,
		    size_t workspace_bytes,
		    cudaStream_t stream);

void gemm_fp16_int4(const half*  A,
		    const uint4b_t* B,
		    const half* weight_scales,
		    half* C,
		    int m, int n, int k, char* workspace_ptr,
		    size_t workspace_bytes,
		    cudaStream_t stream);

} // namespace fastertransformer
