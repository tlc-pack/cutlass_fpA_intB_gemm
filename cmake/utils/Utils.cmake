# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

macro(__fpA_intB_gemm_option variable description value)
  if(NOT DEFINED ${variable})
    set(${variable} ${value} CACHE STRING ${description})
  endif()
endmacro()

set(FPA_INTB_GEMM_ALL_OPTIONS)

#######################################################
# An option that the user can select. Can accept condition to control when option is available for user.
# Usage:
#   fpA_intB_gemm_option(<option_variable> "doc string" <initial value or boolean expression> [IF <condition>])
macro(fpA_intB_gemm_option variable description value)
  set(__value ${value})
  set(__condition "")
  set(__varname "__value")
  list(APPEND FPA_INTB_GEMM_ALL_OPTIONS ${variable})
  foreach(arg ${ARGN})
    if(arg STREQUAL "IF" OR arg STREQUAL "if")
      set(__varname "__condition")
    else()
      list(APPEND ${__varname} ${arg})
    endif()
  endforeach()
  unset(__varname)
  if("${__condition}" STREQUAL "")
    set(__condition 2 GREATER 1)
  endif()

  if(${__condition})
    if("${__value}" MATCHES ";")
      if(${__value})
        __fpA_intB_gemm_option(${variable} "${description}" ON)
      else()
        __fpA_intB_gemm_option(${variable} "${description}" OFF)
      endif()
    elseif(DEFINED ${__value})
      if(${__value})
        __fpA_intB_gemm_option(${variable} "${description}" ON)
      else()
        __fpA_intB_gemm_option(${variable} "${description}" OFF)
      endif()
    else()
      __fpA_intB_gemm_option(${variable} "${description}" "${__value}")
    endif()
  else()
    unset(${variable} CACHE)
  endif()
endmacro()