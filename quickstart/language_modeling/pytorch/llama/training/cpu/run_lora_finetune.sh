
#!/bin/bash

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


ARGS=""

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
#export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set, please create the output path and set it to OUTPUT_DIR"
  exit 1
fi

if [[ "$1" == "bf16" ]]
then
    precision="bf16"
    ARGS="$ARGS --bf16 "
    echo "### running bf16 mode"
elif [[ "$1" == "fp32" ]]
then
    echo "### running fp32 mode"
elif [[ "$1" == "fp16" ]]
then
    precision=fp16
    ARGS="$ARGS --fp16 "
    echo "### running fp16 mode"
elif [[ "$1" == "bf32" ]]
then
    precision=bf32
    ARGS="$ARGS --bf32 "
    echo "### running bf32 mode"
else
    echo "The specified precision '$1' is unsupported."
    echo "Supported precisions are: fp32, bf32, bf16, fp16"
    exit 1
fi

python -m intel_extension_for_pytorch.cpu.launch --throughput-mode --enable_tcmalloc --log_path=${OUTPUT_DIR} --log_file_prefix="./training_log_${precision}_${mode}"  ../../../../../../models/language_modeling/pytorch/llama/training/cpu/finetune.py  $ARGS \
    --base_model 'decapoda-research/llama-7b-hf'\
    --data_path '../../../../../../models/language_modeling/pytorch/llama/training/cpu/alpaca_data.json' \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 32 \
    --micro_batch_size 32 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --val_set_size 2000 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs \
    --group_by_length \