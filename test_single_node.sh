#!/bin/bash
# Quick test script for single node parallel captioning
# Run this directly on the node after SSH-ing in

set -e

echo "==================== Single Node Parallel vLLM Test ===================="
echo "This script will:"
echo "1. Launch 8 parallel vLLM servers (one per GPU)"
echo "2. Run a quick benchmark to verify performance"
echo "3. Process a small batch of videos"
echo "========================================================================="
echo ""

# Configuration
DATA_DIR="/mnt/data/waypoint_1/data/egoexplore_360P"
OUTPUT_DIR="/mnt/data/waypoint_1/data_pt/egoexplore_360P_test"
BATCH_SIZE=64

# Step 1: Kill any existing servers
echo "[1/4] Cleaning up existing vLLM processes..."
pkill -f "vllm serve" 2>/dev/null || true
sleep 2

# Step 2: Launch servers
echo "[2/4] Launching 8 parallel vLLM servers..."
source /mnt/data/envs/.venv/bin/activate

for gpu_id in {0..7}; do
    port=$((8000 + gpu_id))
    echo "  Starting server on GPU $gpu_id (port $port)..."

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

    sleep 1
done

echo "  Waiting for initialization (30 seconds)..."
sleep 30

# Step 3: Health check
echo "[3/4] Verifying server health..."
all_healthy=true
for gpu_id in {0..7}; do
    port=$((8000 + gpu_id))
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:$port/health 2>/dev/null | grep -q "200"; then
        echo "  ✓ GPU $gpu_id (port $port): Healthy"
    else
        echo "  ✗ GPU $gpu_id (port $port): Not responding"
        all_healthy=false
    fi
done

if ! $all_healthy; then
    echo ""
    echo "ERROR: Some servers failed. Check /tmp/vllm_gpu_*.log"
    echo "Showing recent errors:"
    for gpu_id in {0..7}; do
        if [ -f /tmp/vllm_gpu_${gpu_id}.log ]; then
            echo "GPU $gpu_id errors:"
            tail -n 5 /tmp/vllm_gpu_${gpu_id}.log | grep -i error || true
        fi
    done
    exit 1
fi

echo ""
echo "✓ All servers healthy!"
echo ""

# Step 4: Run quick benchmark
echo "[4/4] Running performance test..."
source deactivate
source /mnt/data/shahbuland/venv/bin/activate
cd /mnt/data/shahbuland/owl-data-2

# Create a small test to verify throughput
python3 -c "
import time
import asyncio
from openai import AsyncOpenAI
import json

async def test_throughput():
    # Test with different batch sizes
    batch_sizes = [1, 8, 16, 32, 64]

    print('Testing throughput with different batch sizes:')
    print('-' * 50)

    for batch_size in batch_sizes:
        clients = [AsyncOpenAI(api_key='EMPTY', base_url=f'http://localhost:{8000+i}/v1') for i in range(8)]

        # Create simple test messages
        messages = []
        for i in range(batch_size):
            msg = [{
                'role': 'user',
                'content': 'Describe this game scene: A player running through a forest.'
            }]
            messages.append(msg)

        # Distribute across servers
        async def process_one(client, msg):
            try:
                resp = await client.chat.completions.create(
                    model='Qwen/Qwen2.5-VL-3B-Instruct',
                    messages=msg,
                    temperature=0.1,
                    max_tokens=50
                )
                return resp.choices[0].message.content
            except Exception as e:
                return f'Error: {e}'

        start = time.time()
        tasks = []
        for idx, msg in enumerate(messages):
            client = clients[idx % 8]
            tasks.append(process_one(client, msg))

        results = await asyncio.gather(*tasks)
        duration = time.time() - start
        throughput = batch_size / duration

        print(f'Batch {batch_size:3d}: {duration:6.2f}s ({throughput:6.2f} req/s)')

    print('-' * 50)

asyncio.run(test_throughput())
"

echo ""
echo "==================== Test Complete ===================="
echo ""
echo "To run full captioning, use:"
echo "python -m owl_data.waypoint_1.captions_from_tensors_parallel \\"
echo "  --root_dir $DATA_DIR \\"
echo "  --output_dir $OUTPUT_DIR \\"
echo "  --batch_size $BATCH_SIZE \\"
echo "  --num_servers 8"
echo ""
echo "To monitor GPU usage:"
echo "nvidia-smi dmon -s u -d 1"
echo "========================================================"