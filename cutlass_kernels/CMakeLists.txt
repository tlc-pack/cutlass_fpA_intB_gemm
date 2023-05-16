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
include_directories(/usr/local/cuda-12.1/targets/x86_64-linux/include)

add_library(cutlass_heuristic STATIC cutlass_heuristic.cc)
set_property(TARGET cutlass_heuristic PROPERTY POSITION_INDEPENDENT_CODE ON)

find_package(CUDA)
include_directories(../)
cuda_add_library(fpA_intB_gemm STATIC fpA_intB_gemm/fpA_intB_gemm_fp16_int4.cu fpA_intB_gemm/fpA_intB_gemm_fp16_int8.cu fpA_intB_gemm.cu)
target_link_libraries(fpA_intB_gemm PRIVATE cutlass_heuristic)
