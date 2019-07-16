# Image Recognition with TensorFlow Serving on CPU

### Online and Batch Inference
Model and Precision: InceptionV3 FP32, ResNet50 FP32, and ResNet50 Int8

## Goal

This tutorial will introduce you to the CPU performance considerations for image recognition deep learning models with different precisions and
how to use Intel® Optimizations for [TensorFlow Serving](https://www.tensorflow.org/serving/) to improve inference time on CPUs. 
It also provides sample code that you can use to get your optimized TensorFlow model server and GRPC client up and running quickly.

## Prerequisites

This tutorial assumes you have already:
* [Installed TensorFlow Serving](/docs/general/tensorflow_serving/InstallationGuide.md)
* Read and understood the [General Best Practices](/docs/general/tensorflow_serving/GeneralBestPractices.md),
  especially these sections:
   * **Performance Metrics**
   * **TensorFlow Serving Configuration Settings**
* Ran an example end-to-end using a GRPC client, such as the one in the [Installation Guide](/docs/general/tensorflow_serving/InstallationGuide.md#option-2-using-grpc-this-is-the-fastest-method-but-the-client-has-more-dependencies)
  
## Background

Convolutional neural networks (CNNs) for image recognition are computationally expensive. 
The Intel® Math Kernel Library for Deep Neural Networks (Intel® MKL-DNN) offers significant performance improvements for convolution, pooling, normalization, activation, and other operations via efficient vectorization and multi-threading.
Tuning TensorFlow Serving to take full advantage of your hardware for image recognition deep learning inference involves:
1. Working through this tutorial to set up servable versions of the well-known [ResNet50](https://arxiv.org/pdf/1512.03385.pdf) and [InceptionV3](https://arxiv.org/pdf/1512.00567v1.pdf) CNN models with different precisions.
2. Running a TensorFlow Serving docker container configured for performance given your hardware resources
3. Running a client script to measure online and batch inference performance
4. Experimenting with the TensorFlow Serving settings on your own to further optimize for your model and use case

## Hands-on Tutorial - InceptionV3 and Resnet50

This section shows a step-by-step example for how to serve one of the following Image Recognition models
`(ResNet50 FP32, ResNet50 Int8, and InceptionV3 FP32)` using TensorFlow Serving.
It also explains the possible ways to manage the available CPU resources and tune it for the optimal performance.

For steps 1 and 2, refer to the Intel Model Zoo READMEs:
*  **FP32 precision:** use the Intel Model Zoo `FP32` README sections,
    * [InceptionV3 FP32 README](/benchmarks/image_recognition/tensorflow/inceptionv3#fp32-inference-instructions), and 
    * [ResNet50 FP32 README](/benchmarks/image_recognition/tensorflow/resnet50#fp32-inference-instructions)

*  **Int8 precision:** use the Intel Model Zoo `Int8` README sections,
    * [ResNet50 Int8 README](/benchmarks/image_recognition/tensorflow/resnet50#int8-inference-instructions)

>NOTE: The below example shows InceptionV3 (FP32). The same code snippets will work for ResNet50 (FP32 and Int8) by replacing the model name to `resnet50`.

1. **Download the Model**: Download and extract the InceptionV3 pre-trained model, using the instructions in above README.

2. **(Optional) Download Data**: If you are interested only in testing performance, not accuracy, you can skip this step and use synthetic data.
   If you want to verify prediction accuracy by testing on real data, follow the instructions in one of the READMEs above to download the ImageNet dataset.

3. **Clone this repository**: Clone the [intelai/models](https://github.com/intelai/models) repository and `cd` into the `models/benchmarks/image_recognition/tensorflow_serving/inceptionv3/inference/fp32` directory.
   ```
   $ git clone https://github.com/IntelAI/models.git
   $ cd models/benchmarks/image_recognition/tensorflow_serving/inceptionv3/inference/fp32
   ```

4. **Set up your environment**: In this tutorial, we use a virtual environment to install a few required Python packages. 
   If you do not have pip or virtualenv, you will need to get them first:
   ```
   $ sudo apt-get install python-pip
   $ pip install virtualenv
   $ virtualenv venv
   ```
   Then activate the virtual environment and install `requests` and `tensorflow-serving-api`:
   ```
   $ source venv/bin/activate
   (venv)$ pip install requests tensorflow-serving-api
   ```
5. **Create a SavedModel**: Using the conversion script `model_graph_to_saved_model.py`, convert the pre-trained model graph to a SavedModel.
   (For ResNet50, substitute the name of the ResNet50 FP32 or the ResNet50 Int8 pre-trained model.)
   
   Example:
   ```
   (venv)$ python model_graph_to_saved_model.py --import_path inceptionv3_fp32_pretrained_model.pb
   2018-12-11 15:55:33.018355: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
   2018-12-11 15:55:33.033707: I tensorflow/core/common_runtime/process_util.cc:69] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
   2018-12-11 15:55:33.447678: I tensorflow/tools/graph_transforms/transform_graph.cc:318] Applying strip_unused_nodes
   2018-12-11 15:55:33.515980: I tensorflow/tools/graph_transforms/transform_graph.cc:318] Applying remove_nodes
   2018-12-11 15:55:33.556799: I tensorflow/tools/graph_transforms/transform_graph.cc:318] Applying fold_constants
   2018-12-11 15:55:33.575557: I tensorflow/tools/graph_transforms/transform_graph.cc:318] Applying fold_batch_norms
   2018-12-11 15:55:33.586407: I tensorflow/tools/graph_transforms/transform_graph.cc:318] Applying fold_old_batch_norms
   Exporting trained model to /tmp/1
   Done!
   ```
   This will create a `/tmp/1/` directory with a `saved_model.pb` file in it. This is the file we will serve from TensorFlow Serving.
   The `model_graph_to_saved_model.py` script has applied some transform optimizations and attached a signature definition to the model
   in order to make it compatible with TensorFlow Serving. You can take a look at the script, its flags/options, and these resources for more information:
   * [SavedModel](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/saved_model)
   * [SignatureDefs](https://www.tensorflow.org/serving/signature_defs) 
   
6. **Discover the number of physical cores**: Compute *num_physical_cores* by executing the `lscpu` command and multiplying `Core(s) per socket` by `Socket(s)`.
   For example, for a machine with `Core(s) per socket: 28` and `Socket(s): 2`, *num_physical_cores* = 28 * 2 = 56.

7. **Start the server**: Now let's start up the TensorFlow model server. To optimize overall performance, use the following recommended settings from the
   [General Best Practices](/docs/general/tensorflow_serving/GeneralBestPractices.md):
   * OMP_NUM_THREADS=*num_physical_cores*
   * TENSORFLOW_INTER_OP_PARALLELISM=2
   * TENSORFLOW_INTRA_OP_PARALLELISM=*num_physical_cores*
   
   For our example with 56 physical cores:
   ```
   (venv)$ docker run \
           --name=tfserving \
           --rm \
           -d \
           -p 8500:8500 \
           -v "/tmp:/models/inceptionv3" \
           -e MODEL_NAME=inceptionv3 \
           -e OMP_NUM_THREADS=56 \
           -e TENSORFLOW_INTER_OP_PARALLELISM=2 \
           -e TENSORFLOW_INTRA_OP_PARALLELISM=56 \
           tensorflow/serving:mkl
   ```
   Note: For some models, playing around with these settings values can improve performance even further. 
   We recommend that you experiment with your own hardware and model if you have strict performance requirements.

8. **Run a Test**: Now we can run a test client that downloads a cat picture and sends it for recognition.
   The script has an option for sending a local JPG, if you would prefer to test a different image.
   Run `python image_recognition_client.py --help` for more usage information.
   ```
   (venv)$ python image_recognition_client.py --model inceptionv3
   ```
   The output should be a tensor of class probabilities and `Predicted class:  286`.

9. **Online inference**: Online (or real-time) inference is usually defined as the time it takes to return a prediction for batch size 1.
   To see average online inference performance (in ms), run the script `image_recognition_benchmark.py` using batch_size 1:
   ```
   (venv)$ python image_recognition_benchmark.py --batch_size 1 --model inceptionv3
   Iteration 1: ... sec
   ...
   Iteration 40: ... sec
   Average time: ... sec
   Batch size = 1
   Latency: ... ms
   Throughput: ... images/sec
   ```
   
   In some cases, it is desirable to constrain the inference server to a single core or socket. 
   Docker has many runtime flags that allow you to control the container's access to the host system's CPUs, memory, and other resources.
   See the [Docker document on this topic](https://docs.docker.com/config/containers/resource_constraints/#cpu) for all the options and their definitions.
   For example, to run the container so that a single CPU is used, you can use these settings:
   * `--cpuset-cpus="0"`
   * `--cpus="1"`
   * `OMP_NUM_THREADS=1`
   * `TENSORFLOW_INTER_OP_PARALLELISM=1`
   * `TENSORFLOW_INTRA_OP_PARALLELISM=1`
   ```
   (venv)$ docker run \
           --name=tfserving \
           --rm \
           --cpuset-cpus="0" \
           --cpus="1" \
           -d \
           -p 8500:8500 \
           -v "/tmp:/models/inceptionv3" \
           -e MODEL_NAME=inceptionv3 \
           -e OMP_NUM_THREADS=1 \
           -e TENSORFLOW_INTER_OP_PARALLELISM=1 \
           -e TENSORFLOW_INTRA_OP_PARALLELISM=1 \
           tensorflow/serving:mkl
   ```

10. **Batch inference**: Regardless of hardware, the best batch size is 128. 
    To see average batch inference performance (in images/sec), run the script `image_recognition_benchmark.py` using batch_size 128:
    ```
    (venv)$ python image_recognition_benchmark.py --batch_size 128 --model inceptionv3
    Iteration 1: ... sec
    ...
    Iteration 40: ... sec
    Average time: ... sec
    Batch size = 128
    Throughput: ... images/sec
    ```

11. **Clean up**: 
    * To shut off the server, stop the docker process that is running it. View your running containers with `docker ps`.
      To stop one, copy the Container ID and run `docker stop <container_id>`.
    * Deactivate your virtual environment with `deactivate`.
    
## Conclusion

You have now seen three end-to-end examples of serving an image recognition model for inference using TensorFlow Serving, and learned:
1. How to create a SavedModel from a TensorFlow model graph
2. How to choose good values for the performance-related runtime parameters exposed by the `docker run` command
3. How to verify that the served model can correctly classify an image using a GRPC client
4. How to measure online and batch inference metrics using a GRPC client

With this knowledge and the example code provided, 
you should be able to get started serving your own custom image recognition model with good performance. 
If desired, you should also be able to investigate a variety of different settings combinations to see if further performance improvement are possible.
