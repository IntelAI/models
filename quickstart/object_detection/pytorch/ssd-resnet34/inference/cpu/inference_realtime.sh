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

echo "DATASET_DIR: ${DATASET_DIR}"
echo "PRETRAINED_MODEL: ${PRETRAINED_MODEL}"
echo "PRECISION: ${PRECISION}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"

if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set"
  exit 1
fi

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

if [ -z "${DATASET_DIR}" ]; then
  echo "The required environment variable DATASET_DIR has not been set"
  exit 1
fi

if [ ! -d "${DATASET_DIR}" ]; then
  echo "The DATASET_DIR '${DATASET_DIR}' does not exist"
  exit 1
fi

if [ -z "${PRECISION}" ]; then
  echo "The required environment variable PRECISION has not been set"
  echo "Please set PRECISION to fp32, avx-fp32, int8, avx-int8, or bf16."
  exit 1
fi

cd ${MODEL_DIR}/models/ssd/inference/others/cloud/single_stage_detector/pytorch

# Set env vars that the bash script looks for
export DATA_DIR=${DATASET_DIR}
export MODEL_DIR=${PRETRAINED_MODEL}
export work_space=${OUTPUT_DIR}

if [[ "$PRECISION" == *"avx"* ]]; then
    unset DNNL_MAX_CPU_ISA
fi

if [[ $PRECISION == "int8" || $PRECISION == "avx-int8" ]]; then
    bash run_multi_instance_latency_ipex.sh int8 jit ./pytorch_default_recipe_ssd_configure.json 2>&1 | tee -a ${OUTPUT_DIR}/ssd-resnet34-inference-realtime-int8.log
elif [[ $PRECISION == "bf16" ]]; then
    bash run_multi_instance_latency_ipex.sh bf16 jit 2>&1 | tee -a ${OUTPUT_DIR}/ssd-resnet34-inference-realtime-bf16.log
elif [[ $PRECISION == "fp32" || $PRECISION == "avx-fp32" ]]; then
    bash run_multi_instance_latency_ipex.sh fp32 jit 2>&1 | tee -a ${OUTPUT_DIR}/ssd-resnet34-inference-realtime-fp32.log
else
    echo "The specified precision '${PRECISION}' is unsupported."
    echo "Supported precisions are: fp32, avx-fp32, bf16, int8, and avx-int8"
    exit 1
fi