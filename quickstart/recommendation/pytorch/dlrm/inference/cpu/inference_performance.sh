#!/usr/bin/env bash
#
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

MODEL_DIR=${MODEL_DIR-$PWD}

echo "PRECISION: ${PRECISION}"
echo "DATASET_DIR: ${DATASET_DIR}"
echo "WEIGHT_PATH: ${WEIGHT_PATH}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"

if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set"
  exit 1
fi

if [ -z "${DATASET_DIR}" ]; then
  echo "The required environment variable DATASET_DIR has not been set"
  exit 1
fi

if [ ! -d "${DATASET_DIR}" ]; then
  echo "The DATASET_DIR '${DATASET_DIR}' does not exist"
  exit 1
fi

if [ -z "${WEIGHT_PATH}" ]; then
  echo "The required environment variable WEIGHT_PATH has not been set"
  exit 1
fi

if [ ! -f "${WEIGHT_PATH}" ]; then
  echo "The WEIGHT_PATH ${WEIGHT_PATH} file does not exist "
  exit 1
fi

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

if [ -z "${PRECISION}" ]; then
  echo "The required environment variable PRECISION has not been set"
  echo "Please set PRECISION to fp32, int8, or bf16."
  exit 1
fi

# Set paths that the run_inference_performance.sh scripts expect
export DATASET_PATH=${DATASET_DIR}
export work_space=${OUTPUT_DIR}
export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
export KMP_SETTINGS=1
export KMP_AFFINITY="granularity=fine,compact,1,0"
export KMP_BLOCKTIME=1

if [[ $PRECISION == "bf16" ]]; then
    cd ${MODEL_DIR}/models/dlrm/dlrm
    export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
    bash run_inference_performance.sh bf16 2>&1 | tee -a ${OUTPUT_DIR}/dlrm-inference-performance-bf16.log
elif [[ $PRECISION == "fp32" ]]; then
    cd ${MODEL_DIR}/models/dlrm/dlrm
    bash run_inference_performance.sh 2>&1 | tee -a ${OUTPUT_DIR}/dlrm-inference-performance-fp32.log
elif [[ $PRECISION == "int8" ]]; then
    cd ${MODEL_DIR}/models/dlrm-int8/dlrm
    export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
    bash run_inference_performance.sh int8 2>&1 | tee -a ${OUTPUT_DIR}/dlrm-inference-performance-int8.log
else
    echo "The specified precision '${PRECISION}' is unsupported."
    echo "Supported precisions are: fp32, int8, and bf16"
    exit 1
fi