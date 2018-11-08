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
echo "    FRAMEWORK: ${FRAMEWORK}"
echo "    WORKSPACE: ${WORKSPACE}"
echo "    DATASET_LOCATION: ${DATASET_LOCATION}"
echo "    CHECKPOINT_DIRECTORY: ${CHECKPOINT_DIRECTORY}"
echo "    IN_GRAPH: ${IN_GRAPH}"
echo '    Mounted volumes:'
echo "        ${BENCHMARK_SCRIPTS} mounted on: ${MOUNT_BENCHMARK}"
echo "        ${MODELS_SOURCE_DIRECTORY} mounted on: ${MOUNT_MODELS_SOURCE}"
echo "        ${DATASET_LOCATION_VOL} mounted on: ${DATASET_LOCATION}"
echo "        ${CHECKPOINT_DIRECTORY_VOL} mounted on: ${CHECKPOINT_DIRECTORY}"
echo "    SINGLE_SOCKET: ${SINGLE_SOCKET}"
echo "    MODEL_NAME: ${MODEL_NAME}"
echo "    MODE: ${MODE}"
echo "    PLATFORM: ${PLATFORM}"
echo "    BATCH_SIZE: ${BATCH_SIZE}"
echo "    BENCHMARK_ONLY: ${BENCHMARK_ONLY}"
echo "    ACCURACY_ONLY: ${ACCURACY_ONLY}"

## install common dependencies
apt update ; apt full-upgrade -y
apt-get install python-tk numactl -y
apt install -y libsm6 libxext6
pip install --upgrade pip
pip install requests

single_socket_arg=""
if [ ${SINGLE_SOCKET} == "true" ]; then
    single_socket_arg="--single-socket"
fi

RUN_SCRIPT_PATH="common/${FRAMEWORK}/run_tf_benchmark.py"

DIR=${WORKSPACE}/${MODEL_NAME}/${MODE}/${PLATFORM}

LOG_OUTPUT=${WORKSPACE}/logs
if [ ! -d "${LOG_OUTPUT}" ];then
    mkdir ${LOG_OUTPUT}
fi

# Common execution command used by all models
function run_model() {
    # Navigate to the main benchmark directory before executing the script,
    # since the scripts use the benchmark/common scripts as well.
    cd ${MOUNT_BENCHMARK}

    # Start benchmarking
    eval ${CMD} 2>&1 | tee ${LOGFILE}

    echo "PYTHONPATH: ${PYTHONPATH}" | tee -a ${LOGFILE}
    echo "RUNCMD: ${CMD} " | tee -a ${LOGFILE}
    echo "Batch Size: ${BATCH_SIZE}" | tee -a ${LOGFILE}
    echo "Ran inference with batch size ${BATCH_SIZE}" | tee -a ${LOGFILE}

    LOG_LOCATION_OUTSIDE_CONTAINER="${BENCHMARK_SCRIPTS}/common/${FRAMEWORK}/logs/benchmark_${MODEL_NAME}_${MODE}.log"
    echo "Log location outside container: ${LOG_LOCATION_OUTSIDE_CONTAINER}" | tee -a ${LOGFILE}
}

# NCF model
function ncf() {
    # For nfc, if dataset location is empty, script downloads dataset at given location.
    if [ ! -d "${DATASET_LOCATION}" ];then
        mkdir -p /dataset
    fi

    export PYTHONPATH=${PYTHONPATH}:${MOUNT_MODELS_SOURCE}
    pip install -r ${MOUNT_MODELS_SOURCE}/official/requirements.txt

    CMD="python ${RUN_SCRIPT_PATH} \
    --framework=${FRAMEWORK} \
    --model-name=${MODEL_NAME} \
    --platform=${PLATFORM} \
    --mode=${MODE} \
    --model-source-dir=${MOUNT_MODELS_SOURCE} \
    --batch-size=${BATCH_SIZE} \
    ${single_socket_arg} \
    --data-location=${DATASET_LOCATION} \
    --checkpoint=${CHECKPOINT_DIRECTORY} \
    --verbose"

    PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
}

# SSD-MobileNet models
function ssd_mobilenet() {
    if [ ${MODE} == "inference" ] && [ ${PLATFORM} == "fp32" ]; then
        # install dependencies
        pip install -r "${MOUNT_BENCHMARK}/object_detection/tensorflow/ssd-mobilenet/requirements.txt"

        original_dir=$(pwd)
        cd "${MOUNT_MODELS_SOURCE}/research"

        # install protoc, if necessary, then compile protoc files
        if [ ! -f "bin/protoc" ]; then
            echo "protoc not found, installing protoc 3.0.0"
            apt-get -y install wget
            wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
            unzip -f -o protobuf.zip
            rm protobuf.zip
        else
            echo "protoc already found"
        fi

        echo "Compiling protoc files"
        ./bin/protoc object_detection/protos/*.proto --python_out=.

        export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

        cd $original_dir
        CMD="python ${RUN_SCRIPT_PATH} \
        --framework=${FRAMEWORK} \
        --model-name=${MODEL_NAME} \
        --platform=${PLATFORM} \
        --mode=${MODE} \
        --model-source-dir=${MOUNT_MODELS_SOURCE} \
        --batch-size=${BATCH_SIZE} \
        ${single_socket_arg} \
        --data-location=${DATASET_LOCATION} \
        --in-graph=${IN_GRAPH} \
        --verbose"

        CMD=${CMD} run_model
    else
        echo "MODE:${MODE} and PLATFORM=${PLATFORM} not supported"
    fi
}

# Resnet50 int8 model
function resnet50() {
    if [ ${MODE} == "inference" ] && [ ${PLATFORM} == "int8" ]; then
        # For accuracy, dataset location is required, see README for more information.
        if [ ! -d "${DATASET_LOCATION}" ] && [ ${ACCURACY_ONLY} == "True" ];then
            echo "No Data directory specified, accuracy will not be calculated."
            exit 1
        fi

        accuracy_only_arg=""
        if [ ${ACCURACY_ONLY} == "True" ]; then
            accuracy_only_arg="--accuracy-only"
        fi

        benchmark_only_arg=""
        if [ ${BENCHMARK_ONLY} == "True" ]; then
            benchmark_only_arg="--benchmark-only"
        fi
 
        export PYTHONPATH=${PYTHONPATH}:`pwd`:${MOUNT_BENCHMARK}
        CMD="python ${RUN_SCRIPT_PATH} \
        --framework=${FRAMEWORK} \
        --model-name=${MODEL_NAME} \
        --platform=${PLATFORM} \
        --mode=${MODE} \
        --model-source-dir=${MOUNT_MODELS_SOURCE} \
        --batch-size=${BATCH_SIZE} \
        ${single_socket_arg} \
        ${accuracy_only_arg} \
        ${benchmark_only_arg} \
        --in-graph=${IN_GRAPH}"

        PYTHONPATH=${PYTHONPATH} CMD=${CMD} run_model
    else
        echo "MODE:${MODE} and PLATFORM=${PLATFORM} not supported"
    fi
}

LOGFILE=${LOG_OUTPUT}/benchmark_${MODEL_NAME}_${MODE}.log
echo 'Log output location: ${LOGFILE}'

MODEL_NAME=`echo ${MODEL_NAME} | tr 'A-Z' 'a-z'`
if [ ${MODEL_NAME} == "ncf" ]; then
    ncf
elif [ ${MODEL_NAME} == "ssd-mobilenet" ]; then
    ssd_mobilenet
elif [ ${MODEL_NAME} == "resnet50" ]; then
    resnet50
else
    echo "Unsupported model: ${MODEL_NAME}"
    exit 1
fi
