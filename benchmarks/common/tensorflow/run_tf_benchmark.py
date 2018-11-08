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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
from argparse import ArgumentParser
from common.base_benchmark_util import BaseBenchmarkUtil


class ModelBenchmarkUtil(BaseBenchmarkUtil):
    """Benchmark util for int8 and fp32 models """

    def main(self):
        super(ModelBenchmarkUtil, self).define_args()

        arg_parser = ArgumentParser(parents=[self._common_arg_parser],
                                    description='Parse args for benchmark interface')

        # checkpoint directory location
        arg_parser.add_argument('-c', "--checkpoint",
                                help='Specify the location of trained model checkpoint directory. '
                                     'If mode=training model/weights will be '
                                     'written to this location.'
                                     'If mode=inference assumes that the location '
                                     'points to a model that has already been trained. ',
                                dest='checkpoint', default=None)

        # in graph directory location
        arg_parser.add_argument('-g', "--in-graph",
                                help='Full path to the input graph ',
                                dest='input_graph', default=None)

        arg_parser.add_argument('-k', "--benchmark-only",
                                help='For benchmark measurement only in int8 models.',
                                dest='benchmark_only',
                                action='store_true')

        arg_parser.add_argument("--accuracy-only",
                                help='For accuracy measurement only in int8 models.',
                                dest='accuracy_only',
                                action='store_true')

        args, unknown = arg_parser.parse_known_args()
        mi = super(ModelBenchmarkUtil, self).initialize_model(args, unknown)
        if mi is not None:  # start model initializer if not None
            mi.run()

if __name__ == "__main__":
    util = ModelBenchmarkUtil()
    util.main()
