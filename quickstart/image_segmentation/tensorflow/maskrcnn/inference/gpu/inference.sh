#!/usr/bin/env bash
#
# Copyright (c) 2023 Intel Corporation
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

echo 'MODEL_DIR='$MODEL_DIR
echo 'PRECISION='$PRECISION
echo 'OUTPUT_DIR='$OUTPUT_DIR
echo 'BATCH_SIZE='$BATCH_SIZE
# Create an array of input directories that are expected and then verify that they exist
declare -A input_envs
input_envs[PRECISION]=${PRECISION}
input_envs[OUTPUT_DIR]=${OUTPUT_DIR}
input_envs[PRETRAINED_DIR]=${PRETRAINED_DIR}
input_envs[BATCH_SIZE]=${BATCH_SIZE}

for i in "${!input_envs[@]}"; do
  var_name=$i
  env_param=${input_envs[$i]}
 
  if [[ -z $env_param ]]; then
    echo "The required environment variable $var_name is not set" >&2
    exit 1
  fi
done

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}


if [[ ! -d ${MODEL_DIR}/DeepLearningExamples ]]; then
    echo "ERROR:https://github.com/NVIDIA/DeepLearningExamples.git repo is not cloned and patched. Please clone the repo and apply the patch"
    exit 1
fi

#Download pre-trained model
mkdir -p ${MODEL_DIR}/pretrained_weights
python -u ${MODEL_DIR}/DeepLearningExamples/TensorFlow2/Segmentation/MaskRCNN/scripts/download_weights.py --save_dir=${MODEL_DIR}/pretrained_weights

if [ ${PRECISION} == "fp16" ]; then
    python -u ${MODEL_DIR}/DeepLearningExamples/TensorFlow2/Segmentation/MaskRCNN/scripts/inference.py --data_dir=$DATASET_DIR --batch_size=$BATCH_SIZE --no_xla --weights_dir=${MODEL_DIR}/pretrained_weights --amp
else
    echo "MaskRCNN Inference currently supports FP16 inference"
    exit 1
fi