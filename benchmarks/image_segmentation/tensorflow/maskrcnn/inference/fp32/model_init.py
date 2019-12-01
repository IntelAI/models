#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Intel Corporation
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

#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from common.base_model_init import BaseModelInitializer
from common.base_model_init import set_env_var

import os


class ModelInitializer(BaseModelInitializer):
    """initialize model and run benchmark"""

    def __init__(self, args, custom_args=[], platform_util=None):
        super(ModelInitializer, self).__init__(args, custom_args, platform_util)

        # set num_inter_threads and num_intra_threads
        self.set_num_inter_intra_threads()

        # Set KMP env vars, if they haven't already been set
        config_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        self.set_kmp_vars(config_file_path)

        benchmark_script = os.path.join(
            self.args.intelai_models, "coco.py")
        self.benchmark_command = self.get_command_prefix(args.socket_id) + \
            self.python_exe + " " + benchmark_script + " evaluate "

        set_env_var("OMP_NUM_THREADS", self.args.num_intra_threads)

        self.benchmark_command = self.benchmark_command + \
            " --dataset=" + str(self.args.data_location) + \
            " --num_inter_threads " + str(self.args.num_inter_threads) + \
            " --num_intra_threads " + str(self.args.num_intra_threads) + \
            " --nw 5 --nb 50 --model=coco" + \
            " --infbs " + str(self.args.batch_size)

    def run(self):
        if self.benchmark_command:
            self.run_command(self.benchmark_command)
