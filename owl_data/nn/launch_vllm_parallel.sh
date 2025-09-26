#!/bin/bash

# Launch 8 independent vLLM servers, one per GPU, for true parallel processing
# Each server runs on a different port (8000-8007)

echo "Launching 8 independent vLLM servers for parallel processing..."

# Kill any existing vLLM servers first
echo "Cleaning up any existing vLLM processes..."
pkill -f "vllm serve" 2>/dev/null
sleep 2

# Array to store PIDs
declare -a PIDS

# Launch 8 servers, one per GPU
for gpu_id in {0..7}; do
    port=$((8000 + gpu_id))
    echo "Starting vLLM server on GPU $gpu_id, port $port..."

    CUDA_VISIBLE_DEVICES=$gpu_id vllm serve Qwen/Qwen2.5-VL-3B-Instruct \
        --enable-prefix-caching \
        --kv-cache-dtype fp8 \
        --quantization fp8 \
        --trust-remote-code \
        --host 0.0.0.0 \
        --port $port \
        --gpu-memory-utilization 0.95 \
        --max-model-len 8192 \
        --limit-mm-per-prompt '{"image":10,"video":10}' \
        > /tmp/vllm_gpu_${gpu_id}.log 2>&1 &

    PIDS[$gpu_id]=$!
    echo "  PID: ${PIDS[$gpu_id]}"
done

echo ""
echo "All 8 vLLM servers launched!"
echo "Servers running on ports: 8000-8007"
echo ""
echo "Log files available at: /tmp/vllm_gpu_*.log"
echo ""
echo "Waiting for servers to initialize (30 seconds)..."
sleep 30

# Check if all servers are running
echo ""
echo "Checking server status..."
all_running=true
for gpu_id in {0..7}; do
    port=$((8000 + gpu_id))
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:$port/health | grep -q "200"; then
        echo "  GPU $gpu_id (port $port): ✓ Running"
    else
        echo "  GPU $gpu_id (port $port): ✗ Not responding"
        all_running=false
    fi
done

if $all_running; then
    echo ""
    echo "✓ All servers successfully initialized and ready!"
else
    echo ""
    echo "⚠ Some servers failed to start. Check logs at /tmp/vllm_gpu_*.log"
fi

echo ""
echo "To stop all servers, run: pkill -f 'vllm serve'"
echo ""
echo "PIDs saved for reference:"
for gpu_id in {0..7}; do
    echo "  GPU $gpu_id: ${PIDS[$gpu_id]}"
done