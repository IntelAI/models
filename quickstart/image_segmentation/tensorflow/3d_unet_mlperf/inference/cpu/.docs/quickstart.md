<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `inference_realtime_multi_instance.sh` | Runs multi instance realtime inference using 4 cores per instance for the specified precision (int8, fp32, bfloat32 or bfloat16) with 100 steps and 50 warmup steps to compute latency. Dummy data is used for performance evaluation. Waits for all instances to complete, then prints a summarized throughput value. |
| `inference_throughput_multi_instance.sh` | Runs multi instance batch inference using 1 instance per socket for the specified precision (int8, fp32, bfloat32 or bfloat16) with 100 steps and 50 warmup steps to compute throughput. Dummy data is used for performance evaluation. Waits for all instances to complete, then prints a summarized throughput value. |
| `accuracy.sh` | Measures the inference accuracy (providing a `DATASET_DIR` environment variable is required) for the specified precision (int8, fp32, bfloat32 or bfloat16). |
