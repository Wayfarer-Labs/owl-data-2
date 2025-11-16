#!/bin/bash

# Launch VLLM server with smaller qwen to do mass labelling

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
PORT=${1:-8000}
vllm serve Qwen/Qwen2.5-VL-3B-Instruct \
  --enable-prefix-caching \
  --prefix-caching-hash-algo sha256 \
  --quantization fp8 \
  --trust-remote-code \
  --host 0.0.0.0 --port $PORT \
  --data-parallel-size 8 \
  --tensor-parallel-size 1 \
  --limit-mm-per-prompt '{"image":10,"video":10}' 