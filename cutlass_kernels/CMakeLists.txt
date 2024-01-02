# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


include_directories(../cutlass_extensions/include/)
include_directories(${CUTLASS_DIR}/include)
include_directories(../utils)
include_directories(../)
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

file(GLOB WEIGHT_ONLY_BATCHED_GEMV_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../weightOnlyBatchedGemv/*.cu)

file(GLOB MOE_GEMM_SRC moe_gemm/*.cu)

set(FPA_INTB_GEMM_SRC
    fpA_intB_gemm/fpA_intB_gemm_per_col.cu
    fpA_intB_gemm/fpA_intB_gemm_finegrained.cu
    cutlass_heuristic.cc
    cutlass_preprocessors.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/logger.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils/cuda_utils.cc
    ${WEIGHT_ONLY_BATCHED_GEMV_SRC}
    ${MOE_GEMM_SRC}
    )

add_library(fpA_intB_gemm SHARED ${FPA_INTB_GEMM_SRC})
target_compile_options(
  fpA_intB_gemm PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-expt-relaxed-constexpr>
)
