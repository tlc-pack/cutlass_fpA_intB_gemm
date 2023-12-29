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


if(FPA_INTB_GEMM_TVM_BINDING)
  message(STATUS "Compile tvm binding.")
  if(NOT FPA_INTB_GEMM_TVM_HOME STREQUAL "")
    set(TVM_HOME_SET ${FPA_INTB_GEMM_TVM_HOME})
  elseif(DEFINED ENV{TVM_HOME})
    set(TVM_HOME_SET $ENV{TVM_HOME})
  else()
    message(FATAL_ERROR "Error: Cannot find TVM. Please set the path to TVM by 1) adding `-DFPA_INTB_GEMM_TVM_HOME=path/to/tvm` in the cmake command, or 2) setting the environment variable `TVM_HOME` to the tvm path.")
  endif()
  message(STATUS "FPA_INTB_GEMM uses TVM home ${TVM_HOME_SET}.")

  file(GLOB_RECURSE TVM_BINDING_SRCS ${PROJECT_SOURCE_DIR}/tvm_binding/tvm_binding.cu)
  add_library(fpA_intB_gemm_tvm OBJECT ${TVM_BINDING_SRCS})
  target_compile_definitions(fpA_intB_gemm_tvm PRIVATE -DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\>)
  target_include_directories(fpA_intB_gemm_tvm PRIVATE ${CUTLASS_DIR}/include)
  target_include_directories(fpA_intB_gemm_tvm PRIVATE ${PROJECT_SOURCE_DIR})
  target_include_directories(fpA_intB_gemm_tvm PRIVATE ${TVM_HOME_SET}/include)
  target_include_directories(fpA_intB_gemm_tvm PRIVATE ${TVM_HOME_SET}/3rdparty/dlpack/include)
  target_include_directories(fpA_intB_gemm_tvm PRIVATE ${TVM_HOME_SET}/3rdparty/dmlc-core/include)
  target_compile_options(fpA_intB_gemm_tvm PRIVATE -Xcompiler=-fPIC)
endif(FPA_INTB_GEMM_TVM_BINDING)
