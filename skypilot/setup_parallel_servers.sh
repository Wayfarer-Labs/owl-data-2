#!/bin/bash
# Skypilot-compatible setup script for parallel vLLM servers
# This script is designed to work across multiple nodes in a Skypilot cluster

set -e

echo "==================== Parallel vLLM Setup for Skypilot ===================="
echo "Node Rank: ${SKYPILOT_NODE_RANK:-0}"
echo "Total Nodes: ${SKYPILOT_NUM_NODES:-1}"
echo "=========================================================================="

# Configuration
NODE_RANK=${SKYPILOT_NODE_RANK:-0}
NUM_NODES=${SKYPILOT_NUM_NODES:-1}
BASE_PORT=$((8000 + NODE_RANK * 100))  # Each node gets its own port range

# Cleanup existing servers
echo "Cleaning up any existing vLLM processes..."
pkill -f "vllm serve" 2>/dev/null || true
sleep 2

# Launch 8 independent vLLM servers on this node
echo "Launching 8 parallel vLLM servers on node ${NODE_RANK}..."
echo "Using ports ${BASE_PORT}-$((BASE_PORT + 7))"

declare -a PIDS

for gpu_id in {0..7}; do
    port=$((BASE_PORT + gpu_id))
    log_file="/tmp/vllm_node${NODE_RANK}_gpu${gpu_id}.log"

    echo "Starting server on GPU ${gpu_id}, port ${port}..."

    CUDA_VISIBLE_DEVICES=$gpu_id vllm serve Qwen/Qwen2.5-VL-3B-Instruct \
        --enable-prefix-caching \
        --prefix-caching-hash-algo sha256 \
        --quantization fp8 \
        --trust-remote-code \
        --host 0.0.0.0 \
        --port $port \
        --gpu-memory-utilization 0.95 \
        --max-model-len 8192 \
        --limit-mm-per-prompt '{"image":10,"video":10}' \
        --disable-log-requests \
        > ${log_file} 2>&1 &

    PIDS[$gpu_id]=$!
    echo "  PID: ${PIDS[$gpu_id]}, Log: ${log_file}"
    sleep 1
done

# Wait for servers to initialize
echo ""
echo "Waiting for servers to initialize (30 seconds)..."
sleep 30

# Health check
echo ""
echo "Performing health checks..."
all_healthy=true

for gpu_id in {0..7}; do
    port=$((BASE_PORT + gpu_id))
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:$port/health 2>/dev/null | grep -q "200"; then
        echo "  GPU $gpu_id (port $port): ✓ Healthy"
    else
        echo "  GPU $gpu_id (port $port): ✗ Not responding"
        all_healthy=false
        # Show last few lines of log for debugging
        log_file="/tmp/vllm_node${NODE_RANK}_gpu${gpu_id}.log"
        if [ -f "$log_file" ]; then
            echo "    Last log entries:"
            tail -n 5 "$log_file" | sed 's/^/      /'
        fi
    fi
done

if $all_healthy; then
    echo ""
    echo "✓ All servers successfully initialized on node ${NODE_RANK}!"
    echo ""
    echo "Server endpoints:"
    for gpu_id in {0..7}; do
        port=$((BASE_PORT + gpu_id))
        echo "  GPU $gpu_id: http://localhost:$port"
    done
else
    echo ""
    echo "⚠ Some servers failed to start. Check logs in /tmp/vllm_node${NODE_RANK}_gpu*.log"
    exit 1
fi

echo ""
echo "PIDs for reference:"
for gpu_id in {0..7}; do
    echo "  GPU $gpu_id: ${PIDS[$gpu_id]}"
done

echo ""
echo "==================== Setup Complete ===================="
echo "Node ${NODE_RANK} is ready with 8 parallel vLLM servers"
echo "Ports: ${BASE_PORT}-$((BASE_PORT + 7))"
echo "========================================================"