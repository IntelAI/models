#!/usr/bin/env bash
#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 Intel Corporation
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
# SPDX-License-Identifier: EPL-2.0
#

echo 'Running with parameters:'
echo "    USE_CASE: ${USE_CASE}"
echo "    FRAMEWORK: ${FRAMEWORK}"
echo "    WORKSPACE: ${WORKSPACE}"
echo "    DATASET_LOCATION: ${DATASET_LOCATION}"
echo "    CHECKPOINT_DIRECTORY: ${CHECKPOINT_DIRECTORY}"
echo "    IN_GRAPH: ${IN_GRAPH}"
echo '    Mounted volumes:'
echo "        ${BENCHMARK_SCRIPTS} mounted on: ${MOUNT_BENCHMARK}"
echo "        ${EXTERNAL_MODELS_SOURCE_DIRECTORY} mounted on: ${MOUNT_EXTERNAL_MODELS_SOURCE}"
echo "        ${INTELAI_MODELS} mounted on: ${MOUNT_INTELAI_MODELS_SOURCE}"
echo "        ${DATASET_LOCATION_VOL} mounted on: ${DATASET_LOCATION}"
echo "        ${CHECKPOINT_DIRECTORY_VOL} mounted on: ${CHECKPOINT_DIRECTORY}"
echo "    SOCKET_ID: ${SOCKET_ID}"
echo "    MODEL_NAME: ${MODEL_NAME}"
echo "    MODE: ${MODE}"
echo "    PRECISION: ${PRECISION}"
echo "    BATCH_SIZE: ${BATCH_SIZE}"
echo "    NUM_CORES: ${NUM_CORES}"
echo "    BENCHMARK_ONLY: ${BENCHMARK_ONLY}"
echo "    ACCURACY_ONLY: ${ACCURACY_ONLY}"
echo "    NOINSTALL: ${NOINSTALL}"
echo "    OUTPUT_DIR: ${OUTPUT_DIR}"

# Only inference is supported right now
if [ ${MODE} != "inference" ]; then
  echo "${MODE} mode is not supported"
  exit 1
fi

if [[ ${NOINSTALL} != "True" ]]; then
  ## install common dependencies
  apt update
  apt full-upgrade -y
  apt-get install python-tk numactl -y
  apt install -y libsm6 libxext6
  pip install --upgrade pip
  pip install requests
fi

verbose_arg=""
if [ ${VERBOSE} == "True" ]; then
  verbose_arg="--verbose"
fi

accuracy_only_arg=""
if [ ${ACCURACY_ONLY} == "True" ]; then
  accuracy_only_arg="--accuracy-only"
fi

benchmark_only_arg=""
if [ ${BENCHMARK_ONLY} == "True" ]; then
  benchmark_only_arg="--benchmark-only"
fi

RUN_SCRIPT_PATH="common/${FRAMEWORK}/run_tf_benchmark.py"

timestamp=`date +%Y%m%d_%H%M%S`
LOG_FILENAME="benchmark_${MODEL_NAME}_${MODE}_${PRECISION}_${timestamp}.log"
if [ ! -d "${OUTPUT_DIR}" ]; then
  mkdir ${OUTPUT_DIR}
fi

export PYTHONPATH=${PYTHONPATH}:${MOUNT_INTELAI_MODELS_SOURCE}

# Common execution command used by all models
function run_model() {
  # Navigate to the main benchmark directory before executing the script,
  # since the scripts use the benchmark/common scripts as well.
  cd ${MOUNT_BENCHMARK}

  # Start benchmarking
  eval ${CMD} 2>&1 | tee ${LOGFILE}

  if [ ${VERBOSE} == "True" ]; then
    echo "PYTHONPATH: ${PYTHONPATH}" | tee -a ${LOGFILE}
    echo "RUNCMD: ${CMD} " | tee -a ${LOGFILE}
    echo "Batch Size: ${BATCH_SIZE}" | tee -a ${LOGFILE}
  fi
  echo "Ran ${MODE} with batch size ${BATCH_SIZE}" | tee -a ${LOGFILE}

  # if it starts with /workspace then it's not a separate mounted dir
  # so it's custom and is in same spot as LOGFILE is, otherwise it's mounted in a different place
  if [[ "${OUTPUT_DIR}" = "/workspace"* ]]; then
    LOG_LOCATION_OUTSIDE_CONTAINER=${BENCHMARK_SCRIPTS}/common/${FRAMEWORK}/logs/${LOG_FILENAME}
  else
    LOG_LOCATION_OUTSIDE_CONTAINER=${LOGFILE}
  fi
  echo "Log location outside container: ${LOG_LOCATION_OUTSIDE_CONTAINER}" | tee -a ${LOGFILE}
}

# basic run command with commonly used args
CMD="python ${RUN_SCRIPT_PATH} \
--framework=${FRAMEWORK} \
--use-case=${USE_CASE} \
--model-name=${MODEL_NAME} \
--precision=${PRECISION} \
--mode=${MODE} \
--model-source-dir=${MOUNT_EXTERNAL_MODELS_SOURCE} \
--benchmark-dir=${MOUNT_BENCHMARK} \
--intelai-models=${MOUNT_INTELAI_MODELS_SOURCE} \
--num-cores=${NUM_CORES} \
--batch-size=${BATCH_SIZE} \
--socket-id=${SOCKET_ID} \
${accuracy_only_arg} \
${benchmark_only_arg} \
${verbose_arg}"

# Add on --in-graph and --data-location for int8 inference
if [ ${MODE} == "inference" ] && [ ${PRECISION} == "int8" ]; then
    CMD="${CMD} --in-graph=${IN_GRAPH} --data-location=${DATASET_LOCATION}"
fi

function install_protoc() {
  # install protoc, if necessary, then compile protoc files
  if [ ! -f "bin/protoc" ]; then
    install_location=$1
    echo "protoc not found, installing protoc from ${install_location}"
    apt-get -y install wget
    wget -O protobuf.zip ${install_location}
    unzip -o protobuf.zip
    rm protobuf.zip
  else
    echo "protoc already found"
  fi

}

# 3D UNet model
function 3d_unet() {
  if [ ${PRECISION} == "fp32" ]; then
    if [ ${NOINSTALL} != "True" ]; then
      pip install -r "${MOUNT_BENCHMARK}/${USE_CASE}/${FRAMEWORK}/${MODEL_NAME}/requirements.txt"
    fi
    export PYTHONPATH=${PYTHONPATH}:${MOUNT_INTELAI_MODELS_SOURCE}/inference/fp32
    CMD="${CMD} --in-graph=${IN_GRAPH} --data-location=${DATASET_LOCATION}"
    PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
  else
    echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
    exit 1
  fi
}

# DCGAN model
function dcgan() {
  if [ ${PRECISION} == "fp32" ]; then

    export PYTHONPATH=${PYTHONPATH}:${MOUNT_EXTERNAL_MODELS_SOURCE}/research:${MOUNT_EXTERNAL_MODELS_SOURCE}/research/slim:${MOUNT_EXTERNAL_MODELS_SOURCE}/research/gan/cifar

    CMD="${CMD} --checkpoint=${CHECKPOINT_DIRECTORY} --data-location=${DATASET_LOCATION}"

    PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
  else
    echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
    exit 1
  fi
}

# DRAW model
function draw() {
  if [ ${PRECISION} == "fp32" ]; then
    CMD="${CMD} --checkpoint=${CHECKPOINT_DIRECTORY} --data-location=${DATASET_LOCATION}"
    PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
  else
    echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
    exit 1
  fi
}

# Fast R-CNN (ResNet50) model
function fastrcnn() {
    export PYTHONPATH=$PYTHONPATH:${MOUNT_EXTERNAL_MODELS_SOURCE}/research:${MOUNT_EXTERNAL_MODELS_SOURCE}/research/slim

    if [ ${NOINSTALL} != "True" ]; then
      # install dependencies
      pip install -r "${MOUNT_BENCHMARK}/object_detection/tensorflow/fastrcnn/requirements.txt"
      original_dir=$(pwd)
      cd "${MOUNT_EXTERNAL_MODELS_SOURCE}/research"
      # install protoc v3.3.0, if necessary, then compile protoc files
      install_protoc "https://github.com/google/protobuf/releases/download/v3.3.0/protoc-3.3.0-linux-x86_64.zip"
      echo "Compiling protoc files"
      ./bin/protoc object_detection/protos/*.proto --python_out=.

      # install cocoapi
      cd ${MOUNT_EXTERNAL_MODELS_SOURCE}/cocoapi/PythonAPI
      echo "Installing COCO API"
      make
      cp -r pycocotools ${MOUNT_EXTERNAL_MODELS_SOURCE}/research/
    fi

    export PYTHONPATH=${PYTHONPATH}:${MOUNT_EXTERNAL_MODELS_SOURCE}

    if [ ${PRECISION} == "fp32" ]; then
      config_file_arg=""
      if [ -n "${config_file}" ]; then
        config_file_arg="--config_file=${config_file}"
      fi

      if [[ -z "${config_file}" ]] && [ ${BENCHMARK_ONLY} == "True" ]; then
        echo "Fast R-CNN requires --config_file arg to be defined"
        exit 1
      fi
      cd $original_dir
      CMD="${CMD} --checkpoint=${CHECKPOINT_DIRECTORY} \
      --data-location=${DATASET_LOCATION} \
      --in-graph=${IN_GRAPH} ${config_file_arg}"
    elif [ ${PRECISION} == "int8" ]; then
      number_of_steps_arg=""
      if [ -n "${number_of_steps}" ] && [ ${BENCHMARK_ONLY} == "True" ]; then
        CMD="${CMD} --number-of-steps=${number_of_steps}"
      fi
      cd $original_dir
      PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
    else
      echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
      exit 1
    fi
}

# inceptionv3 model
function inceptionv3() {
  if [ ${PRECISION} == "int8" ]; then
    # For accuracy, dataset location is required, see README for more information.
    if [ ! -d "${DATASET_LOCATION}" ] && [ ${ACCURACY_ONLY} == "True" ]; then
      echo "No Data directory specified, accuracy will not be calculated."
      exit 1
    fi

    export PYTHONPATH=${PYTHONPATH}:${MOUNT_EXTERNAL_MODELS_SOURCE}
    input_height_arg=""
    input_width_arg=""

    if [ -n "${input_height}" ]; then
      input_height_arg="--input-height=${input_height}"
    fi

    if [ -n "${input_width}" ]; then
      input_width_arg="--input-width=${input_width}"
    fi

    CMD="${CMD} ${input_height_arg} ${input_width_arg}"
    PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model

  elif [ ${PRECISION} == "fp32" ]; then
    # Run inception v3 fp32 inference
    CMD="${CMD} --in-graph=${IN_GRAPH} --data-location=${DATASET_LOCATION}"
    PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
  else
    echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
    exit 1
  fi
}

# inception_resnet_v2 model
function inception_resnet_v2() {
  # For accuracy, dataset location is required, see README for more information.
  if [ "${DATASET_LOCATION_VOL}" == None ] && [ ${ACCURACY_ONLY} == "True" ]; then
    echo "No Data directory specified, accuracy will not be calculated."
    exit 1
  fi

  if [ ${PRECISION} == "fp32" ]; then
    # Add on --in-graph and --data-location for int8 inference
    if [ ${MODE} == "inference" ] && [ ${ACCURACY_ONLY} == "True" ]; then
      CMD="${CMD} --in-graph=${IN_GRAPH} --data-location=${DATASET_LOCATION}"
    elif [ ${MODE} == "inference" ] && [ ${BENCHMARK_ONLY} == "True" ]; then
      CMD="${CMD} --checkpoint=${CHECKPOINT_DIRECTORY} --data-location=${DATASET_LOCATION}"
    fi
    PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
  else
    echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
    exit 1
  fi
}

# Mask R-CNN model
function maskrcnn() {
  if [ ${PRECISION} == "fp32" ]; then
    original_dir=$(pwd)

    if [ ${NOINSTALL} != "True" ]; then
      # install dependencies
      pip3 install -r ${MOUNT_EXTERNAL_MODELS_SOURCE}/requirements.txt

      # install cocoapi
      cd ${MOUNT_EXTERNAL_MODELS_SOURCE}/coco/PythonAPI
      echo "Installing COCO API"
      make
      cp -r pycocotools ${MOUNT_EXTERNAL_MODELS_SOURCE}/samples/coco
    fi
    export PYTHONPATH=${PYTHONPATH}:${MOUNT_EXTERNAL_MODELS_SOURCE}:${MOUNT_EXTERNAL_MODELS_SOURCE}/samples/coco:${MOUNT_EXTERNAL_MODELS_SOURCE}/mrcnn
    cd ${original_dir}
    CMD="${CMD} --data-location=${DATASET_LOCATION}"
    PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
  else
    echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
    exit 1
  fi
}

# mobilenet_v1 model
function mobilenet_v1() {
  if [ ${PRECISION} == "fp32" ]; then
    CMD="${CMD} --checkpoint=${CHECKPOINT_DIRECTORY} --in-graph=${IN_GRAPH} --data-location=${DATASET_LOCATION}"
    export PYTHONPATH=${PYTHONPATH}:${MOUNT_EXTERNAL_MODELS_SOURCE}:${MOUNT_EXTERNAL_MODELS_SOURCE}/research:${MOUNT_EXTERNAL_MODELS_SOURCE}/research/slim
    PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
  else
    echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
    exit 1
  fi
}

# NCF model
function ncf() {
  if [ ${PRECISION} == "fp32" ]; then
    # For nfc, if dataset location is empty, script downloads dataset at given location.
    if [ ! -d "${DATASET_LOCATION}" ]; then
      mkdir -p /dataset
    fi

    export PYTHONPATH=${PYTHONPATH}:${MOUNT_EXTERNAL_MODELS_SOURCE}

    if [ ${NOINSTALL} != "True" ]; then
      pip install -r ${MOUNT_EXTERNAL_MODELS_SOURCE}/official/requirements.txt
    fi

    CMD="${CMD} --checkpoint=${CHECKPOINT_DIRECTORY} \
    --data-location=${DATASET_LOCATION}"

    PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
  else
    echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
    exit 1
  fi
}

# ResNet101 model
function resnet101() {
    export PYTHONPATH=${PYTHONPATH}:$(pwd):${MOUNT_BENCHMARK}

    # For accuracy, dataset location is required.
    if [ "${DATASET_LOCATION_VOL}" == "None" ] && [ ${ACCURACY_ONLY} == "True" ]; then
      echo "No Data directory specified, accuracy will not be calculated."
      exit 1
    fi

    if [ ${PRECISION} == "int8" ]; then
        PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
    elif [ ${PRECISION} == "fp32" ]; then
      CMD="${CMD} --in-graph=${IN_GRAPH} --data-location=${DATASET_LOCATION}"
      PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
    else
      echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
      exit 1
    fi
}

# Resnet50 int8 and fp32 models
function resnet50() {
    export PYTHONPATH=${PYTHONPATH}:$(pwd):${MOUNT_BENCHMARK}

    if [ ${PRECISION} == "int8" ]; then
        # For accuracy, dataset location is required, see README for more information.
        if [ "${DATASET_LOCATION_VOL}" == None ] && [ ${ACCURACY_ONLY} == "True" ]; then
          echo "No Data directory specified, accuracy will not be calculated."
          exit 1
        fi
        PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model

    elif [ ${PRECISION} == "fp32" ]; then
        # Run resnet50 fp32 inference
        CMD="${CMD} --in-graph=${IN_GRAPH} --data-location=${DATASET_LOCATION}"
        PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
    else
        echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
        exit 1
    fi
}

# R-FCN (ResNet101) model
function rfcn() {
  export PYTHONPATH=$PYTHONPATH:${MOUNT_EXTERNAL_MODELS_SOURCE}/research:${MOUNT_EXTERNAL_MODELS_SOURCE}/research/slim:${MOUNT_EXTERNAL_MODELS_SOURCE}

  if [ ${NOINSTALL} != "True" ]; then
    # install dependencies
    pip install -r "${MOUNT_BENCHMARK}/object_detection/tensorflow/rfcn/requirements.txt"
    original_dir=$(pwd)

    cd "${MOUNT_EXTERNAL_MODELS_SOURCE}/research"
    # install protoc v3.3.0, if necessary, then compile protoc files
    install_protoc "https://github.com/google/protobuf/releases/download/v3.3.0/protoc-3.3.0-linux-x86_64.zip"
    echo "Compiling protoc files"
    ./bin/protoc object_detection/protos/*.proto --python_out=.

    # install cocoapi
    cd ${MOUNT_EXTERNAL_MODELS_SOURCE}/cocoapi/PythonAPI
    echo "Installing COCO API"
    make
    cp -r pycocotools ${MOUNT_EXTERNAL_MODELS_SOURCE}/research/
  fi

  split_arg=""
  if [ -n "${split}" ] && [ ${ACCURACY_ONLY} == "True" ]; then
      split_arg="--split=${split}"
  fi

  if [ ${PRECISION} == "fp32" ]; then
      if [[ -z "${config_file}" ]] && [ ${BENCHMARK_ONLY} == "True" ]; then
          echo "R-FCN requires --config_file arg to be defined"
          exit 1
      fi

      CMD="${CMD} --checkpoint=${CHECKPOINT_DIRECTORY} \
      --config_file=${config_file} --data-location=${DATASET_LOCATION} \
      --in-graph=${IN_GRAPH} ${split_arg}"
   else
      echo "MODE:${MODE} and PRECISION=${PRECISION} not supported"
  fi
  cd $original_dir
  PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
}

# SqueezeNet model
function squeezenet() {
  if [ ${PRECISION} == "fp32" ]; then
    CMD="${CMD} --checkpoint=${CHECKPOINT_DIRECTORY} \
    --data-location=${DATASET_LOCATION}"

    PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
  else
    echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
    exit 1
  fi
}

# SSD-MobileNet model
function ssd_mobilenet() {
  if [ ${PRECISION} == "fp32" ]; then
    if [ ${BATCH_SIZE} != "-1" ]; then
      echo "Warning: SSD-MobileNet inference script does not use the batch_size arg"
    fi

    export PYTHONPATH=$PYTHONPATH:${MOUNT_EXTERNAL_MODELS_SOURCE}/research:${MOUNT_EXTERNAL_MODELS_SOURCE}/research/slim:${MOUNT_EXTERNAL_MODELS_SOURCE}/research/object_detection

    if [ ${NOINSTALL} != "True" ]; then
      # install dependencies
      pip install -r "${MOUNT_BENCHMARK}/object_detection/tensorflow/ssd-mobilenet/requirements.txt"

      pushd "${MOUNT_EXTERNAL_MODELS_SOURCE}/research"

      # install protoc, if necessary, then compile protoc files
      install_protoc "https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip"

      echo "Compiling protoc files"
      ./bin/protoc object_detection/protos/*.proto --python_out=.

      popd
    fi

    CMD="${CMD} --in-graph=${IN_GRAPH} \
    --data-location=${DATASET_LOCATION}"
    CMD=${CMD} run_model
  else
    echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
    exit 1
  fi
}

# UNet model
function unet() {
  if [ ${PRECISION} == "fp32" ]; then
    if [[ -z "${checkpoint_name}" ]]; then
      echo "wavenet requires --checkpoint_name arg to be defined"
      exit 1
    fi
    if [ ${ACCURACY_ONLY} == "True" ]; then
      echo "Accuracy testing is not supported for ${MODEL_NAME}"
      exit 1
    fi
    CMD="${CMD} --checkpoint=${CHECKPOINT_DIRECTORY} --checkpoint_name=${checkpoint_name}"
    export PYTHONPATH=${PYTHONPATH}:${MOUNT_EXTERNAL_MODELS_SOURCE}
    PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
  else
    echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
    exit 1
  fi
}

# transformer language model
function transformer_language() {
  if [ ${PRECISION} == "fp32" ]; then

    if [[ -z "${decode_from_file}" ]]; then
        echo "transformer-language requires --decode_from_file arg to be defined"
        exit 1
    fi
    if [[ -z "${reference}" ]]; then
        echo "transformer-language requires --reference arg to be defined"
        exit 1
    fi
    if [[ -z "${CHECKPOINT_DIRECTORY}" ]]; then
        echo "transformer-language requires --checkpoint arg to be defined"
        exit 1
    fi
    if [[ -z "${DATASET_LOCATION}" ]]; then
        echo "transformer-language requires --data-location arg to be defined"
        exit 1
    fi

    if [ ${NOINSTALL} != "True" ]; then
      # install dependencies
      echo "Installing tensor2tensor for CPU..."
      pip install tensor2tensor[tensorflow]
    fi

    cp ${MOUNT_INTELAI_MODELS_SOURCE}/${MODE}/${PRECISION}/decoding.py ${MOUNT_EXTERNAL_MODELS_SOURCE}/tensor2tensor/utils/decoding.py

    CMD="${CMD} --checkpoint=${CHECKPOINT_DIRECTORY} \
    --data-location=${DATASET_LOCATION} \
    --decode_from_file=${CHECKPOINT_DIRECTORY}/${decode_from_file} \
    --reference=${CHECKPOINT_DIRECTORY}/${reference}"

    PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
  else
    echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
    exit 1
  fi
}

# Wavenet model
function wavenet() {
  if [ ${PRECISION} == "fp32" ]; then
    if [[ -z "${checkpoint_name}" ]]; then
      echo "wavenet requires --checkpoint_name arg to be defined"
      exit 1
    fi

    if [[ -z "${sample}" ]]; then
      echo "wavenet requires --sample arg to be defined"
      exit 1
    fi

    export PYTHONPATH=${PYTHONPATH}:${MOUNT_EXTERNAL_MODELS_SOURCE}

    if [ ${NOINSTALL} != "True" ]; then
      pip install -r ${MOUNT_EXTERNAL_MODELS_SOURCE}/requirements.txt
    fi

    CMD="${CMD} --checkpoint=${CHECKPOINT_DIRECTORY} \
        --checkpoint_name=${checkpoint_name} \
        --sample=${sample}"

    PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
  else
    echo "PRECISION=${PRECISION} is not supported for ${MODEL_NAME}"
    exit 1
  fi
}

# Wide & Deep model
function wide_deep() {
    if [ ${PRECISION} == "fp32" ]; then
      export PYTHONPATH=${PYTHONPATH}:${MOUNT_EXTERNAL_MODELS_SOURCE}

      if [ ${NOINSTALL} != "True" ]; then
        # install dependencies
        pip install -r "${MOUNT_BENCHMARK}/classification/tensorflow/wide_deep/requirements.txt"
      fi

      CMD="${CMD} --checkpoint=${CHECKPOINT_DIRECTORY} \
      --data-location=${DATASET_LOCATION}"
      CMD=${CMD} run_model
    else
      echo "PRECISION=${PRECISION} not supported for ${MODEL_NAME}"
      exit 1
    fi
}

LOGFILE=${OUTPUT_DIR}/${LOG_FILENAME}
echo "Log output location: ${LOGFILE}"

MODEL_NAME=$(echo ${MODEL_NAME} | tr 'A-Z' 'a-z')
if [ ${MODEL_NAME} == "3d_unet" ]; then
  3d_unet
elif [ ${MODEL_NAME} == "dcgan" ]; then
  dcgan
elif [ ${MODEL_NAME} == "draw" ]; then
  draw
elif [ ${MODEL_NAME} == "fastrcnn" ]; then
  fastrcnn
elif [ ${MODEL_NAME} == "inceptionv3" ]; then
  inceptionv3
elif [ ${MODEL_NAME} == "inception_resnet_v2" ]; then
  inception_resnet_v2
elif [ ${MODEL_NAME} == "maskrcnn" ]; then
  maskrcnn
elif [ ${MODEL_NAME} == "mobilenet_v1" ]; then
  mobilenet_v1
elif [ ${MODEL_NAME} == "ncf" ]; then
  ncf
elif [ ${MODEL_NAME} == "resnet101" ]; then
  resnet101
elif [ ${MODEL_NAME} == "resnet50" ]; then
  resnet50
elif [ ${MODEL_NAME} == "rfcn" ]; then
  rfcn
elif [ ${MODEL_NAME} == "squeezenet" ]; then
  squeezenet
elif [ ${MODEL_NAME} == "ssd-mobilenet" ]; then
  ssd_mobilenet
elif [ ${MODEL_NAME} == "unet" ]; then
  unet
elif [ ${MODEL_NAME} == "transformer_language" ]; then
  transformer_language
elif [ ${MODEL_NAME} == "wavenet" ]; then
  wavenet
elif [ ${MODEL_NAME} == "wide_deep" ]; then
  wide_deep
else
  echo "Unsupported model: ${MODEL_NAME}"
  exit 1
fi
