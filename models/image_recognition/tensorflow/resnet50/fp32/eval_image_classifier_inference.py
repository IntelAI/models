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

import time
import numpy as np
from argparse import ArgumentParser

import tensorflow as tf
import tensorflow.tools.graph_transforms as graph_transforms

import datasets
import preprocessing

INPUTS = 'input'
OUTPUTS = 'predict'
OPTIMIZATION = 'strip_unused_nodes remove_nodes(op=Identity, op=CheckNumerics) fold_constants(ignore_errors=true) fold_batch_norms fold_old_batch_norms'

RESNET_IMAGE_SIZE = 224
IMAGENET_VALIDATION_IMAGES = 50000


class eval_classifier_optimized_graph:
  """Evaluate image classifier with optimized TensorFlow graph"""

  def __init__(self):

    arg_parser = ArgumentParser(description='Parse args')

    arg_parser.add_argument('-b', "--batch-size",
                            help="Specify the batch size. If this " \
                                 "parameter is not specified or is -1, the " \
                                 "largest ideal batch size for the model will " \
                                 "be used.",
                            dest="batch_size", type=int, default=-1)

    arg_parser.add_argument('-e', "--inter-op-parallelism-threads",
                            help='The number of inter-thread.',
                            dest='num_inter_threads', type=int, default=0)

    arg_parser.add_argument('-a', "--intra-op-parallelism-threads",
                            help='The number of intra-thread.',
                            dest='num_intra_threads', type=int, default=0)

    arg_parser.add_argument('-m', "--model-name",
                            help='Specify the model name to run benchmark for',
                            dest='model_name')

    arg_parser.add_argument('-g', "--input-graph",
                            help='Specify the input graph for the transform tool',
                            dest='input_graph')

    arg_parser.add_argument('-d', "--data-location",
                            help='Specify the location of the data. '
                                 'If this parameter is not specified, '
                                 'the benchmark will use random/dummy data.',
                            dest="data_location", default=None)

    arg_parser.add_argument('-s', "--steps",
                            help='Specify the number of inference steps.',
                            dest="steps", default=40)

    arg_parser.add_argument('-r', "--accuracy-only",
                            help='For accuracy measurement only.',
                            dest='accuracy_only', action='store_true')
    # parse the arguments
    self.args = arg_parser.parse_args()
    # validate the arguments
    self.validate_args()

  def run(self):
    """run benchmark with optimized graph"""

    with tf.Graph().as_default() as graph:

      config = tf.ConfigProto()
      config.allow_soft_placement = True
      config.intra_op_parallelism_threads = self.args.num_intra_threads
      config.inter_op_parallelism_threads = self.args.num_inter_threads

      with tf.Session(config=config) as sess:

        # convert the frozen graph to optimized graph
        graph_def = tf.GraphDef()
        with tf.gfile.FastGFile(self.args.input_graph, 'rb') as input_file:
          input_graph_content = input_file.read()
          graph_def.ParseFromString(input_graph_content)

        output_graph = graph_transforms.TransformGraph(graph_def,
                                                       [INPUTS], [OUTPUTS], [OPTIMIZATION])
        sess.graph.as_default()
        tf.import_graph_def(output_graph, name='')

        # Definite input and output Tensors for detection_graph
        input_tensor = graph.get_tensor_by_name('input:0')
        output_tensor = graph.get_tensor_by_name('predict:0')
        tf.global_variables_initializer()

        num_processed_images = 0

        if (self.args.data_location):
          print("Inference with real data.")
          dataset = datasets.ImagenetData(self.args.data_location)
          preprocessor = preprocessing.ImagePreprocessor(
            RESNET_IMAGE_SIZE, RESNET_IMAGE_SIZE, self.args.batch_size,
            1,  # device count
            tf.float32,  # data_type for input fed to the graph
            train=False,  # doing inference
            resize_method='crop')
          images, labels = preprocessor.minibatch(dataset, subset='validation')
          num_remaining_images = dataset.num_examples_per_epoch(subset='validation') \
                                 - num_processed_images
        else:
          print("Inference with dummy data.")
          num_remaining_images = IMAGENET_VALIDATION_IMAGES

        if (not self.args.accuracy_only):  # performance check
          step = 0
          warm_up_iteration = 10
          total_time = 0

          while num_remaining_images >= self.args.batch_size and step < self.args.steps:
            step += 1

            # Reads and preprocess data
            if (self.args.data_location):
              preprocessed_images = sess.run([images[0]])
              image_np = preprocessed_images[0]
            else:
              image_np = np.random.rand(self.args.batch_size, RESNET_IMAGE_SIZE, RESNET_IMAGE_SIZE, 3).astype(np.uint8)

            num_processed_images += self.args.batch_size
            num_remaining_images -= self.args.batch_size

            start_time = time.time()
            (predicts) = sess.run([output_tensor], feed_dict={input_tensor: image_np})
            time_consume = time.time() - start_time

            print('Iteration %d: %.3f sec' % (step, time_consume))
            if step > warm_up_iteration:
              total_time += time_consume

          time_average = total_time / (step - warm_up_iteration)
          print('Average time: %.3f sec' % (time_average))

          print('Batch size = %d' % self.args.batch_size)
          if (self.args.batch_size == 1):
            print('Latency: %.3f ms' % (time_average * 1000))
          # print throughput for both batch size 1 and 128
          print('Throughput: %.3f images/sec' % (self.args.batch_size / time_average))

        else:  # accuracy check
          total_accuracy1, total_accuracy5 = (0.0, 0.0)

          while num_remaining_images >= self.args.batch_size:
            # Reads and preprocess data
            np_images, np_labels = sess.run([images[0], labels[0]])
            num_processed_images += self.args.batch_size
            num_remaining_images -= self.args.batch_size

            # Compute inference on the preprocessed data
            predictions = sess.run(output_tensor,
                                   {input_tensor: np_images})
            accuracy1 = tf.reduce_sum(
              tf.cast(tf.nn.in_top_k(tf.constant(predictions),
                                     tf.constant(np_labels), 1), tf.float32))

            accuracy5 = tf.reduce_sum(
              tf.cast(tf.nn.in_top_k(tf.constant(predictions),
                                     tf.constant(np_labels), 5), tf.float32))
            np_accuracy1, np_accuracy5 = sess.run([accuracy1, accuracy5])
            total_accuracy1 += np_accuracy1
            total_accuracy5 += np_accuracy5
            print("Processed %d images. (Top1 accuracy, Top5 accuracy) = (%0.4f, %0.4f)" \
                  % (num_processed_images, total_accuracy1 / num_processed_images,
                     total_accuracy5 / num_processed_images))

  def validate_args(self):
    """validate the arguments"""

    if not self.args.data_location:
      if self.args.accuracy_only:
        raise ValueError("You must use real data for accuracy measurement.")


if __name__ == "__main__":
  evaluate_opt_graph = eval_classifier_optimized_graph()
  evaluate_opt_graph.run()
