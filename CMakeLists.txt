project(FasterTransformer LANGUAGES CXX CUDA)

include(cmake/utils/Utils.cmake)

set(CUTLASS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cutlass)

fpA_intB_gemm_option(FPA_INTB_GEMM_TVM_BINDING "Whether to compile tvm binding or not." OFF)
fpA_intB_gemm_option(FPA_INTB_GEMM_HOME "The path to tvm for building tvm binding." "")

add_subdirectory(cutlass_kernels)
add_subdirectory(tvm_binding)
