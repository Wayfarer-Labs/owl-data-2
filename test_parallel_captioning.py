#!/usr/bin/env python3
"""
Test script to verify parallel vLLM captioning performance and GPU utilization.

Usage:
1. First launch the parallel servers:
   bash owl_data/nn/launch_vllm_parallel.sh

2. Then run this test:
   python test_parallel_captioning.py

3. Monitor GPU utilization in another terminal:
   nvidia-smi -l 1
"""

import torch
import time
import asyncio
import json
from openai import AsyncOpenAI
import numpy as np
from PIL import Image
import io
import base64

# Test configuration
NUM_SERVERS = 8
BASE_PORT = 8000
BATCH_SIZES = [1, 4, 8, 16, 32, 64]
NUM_WARMUP = 2
NUM_RUNS = 5

def create_dummy_frames(batch_size, num_frames=5, height=224, width=224):
    """Create dummy video frames for testing."""
    frames = []
    for _ in range(batch_size):
        # Create random RGB frames
        frame_tensor = torch.randint(0, 256, (num_frames, 3, height, width), dtype=torch.uint8)
        frames.append(frame_tensor)
    return frames

def to_data_url_png(chw_uint8: torch.Tensor) -> str:
    """Convert tensor to data URL."""
    img = Image.fromarray(chw_uint8.permute(1,2,0).contiguous().cpu().numpy())
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

async def test_single_server(batch_size):
    """Test with single server (traditional approach)."""
    client = AsyncOpenAI(api_key="EMPTY", base_url=f"http://localhost:{BASE_PORT}/v1")
    frames = create_dummy_frames(batch_size)

    async def process_one(frame_tensor):
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this video game scene briefly."},
            ] + [{"type": "image_url", "image_url": {"url": to_data_url_png(frame)}}
                 for frame in frame_tensor]
        }]

        resp = await client.chat.completions.create(
            model="Qwen/Qwen2.5-VL-3B-Instruct",
            messages=messages,
            temperature=0.1,
            max_tokens=128
        )
        return resp.choices[0].message.content

    start = time.time()
    results = await asyncio.gather(*[process_one(f) for f in frames])
    end = time.time()

    return end - start, len(results)

async def test_parallel_servers(batch_size):
    """Test with parallel servers (new approach)."""
    # Create clients for all servers
    clients = []
    for i in range(NUM_SERVERS):
        port = BASE_PORT + i
        client = AsyncOpenAI(api_key="EMPTY", base_url=f"http://localhost:{port}/v1")
        clients.append(client)

    frames = create_dummy_frames(batch_size)

    async def process_one(client, frame_tensor):
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this video game scene briefly."},
            ] + [{"type": "image_url", "image_url": {"url": to_data_url_png(frame)}}
                 for frame in frame_tensor]
        }]

        resp = await client.chat.completions.create(
            model="Qwen/Qwen2.5-VL-3B-Instruct",
            messages=messages,
            temperature=0.1,
            max_tokens=128
        )
        return resp.choices[0].message.content

    # Distribute requests round-robin across servers
    tasks = []
    for idx, frame_tensor in enumerate(frames):
        client_idx = idx % NUM_SERVERS
        client = clients[client_idx]
        tasks.append(process_one(client, frame_tensor))

    start = time.time()
    results = await asyncio.gather(*tasks)
    end = time.time()

    return end - start, len(results)

def run_benchmark():
    """Run the full benchmark comparing single vs parallel servers."""
    print("=" * 80)
    print("vLLM Parallel Captioning Performance Test")
    print("=" * 80)
    print()

    # Check if servers are running
    print("Checking server availability...")
    import requests
    servers_available = 0
    for i in range(NUM_SERVERS):
        port = BASE_PORT + i
        try:
            resp = requests.get(f"http://localhost:{port}/health", timeout=1)
            if resp.status_code == 200:
                servers_available += 1
                print(f"  Server on port {port}: ✓")
            else:
                print(f"  Server on port {port}: ✗")
        except:
            print(f"  Server on port {port}: ✗")

    if servers_available == 0:
        print("\n❌ No servers available! Please run: bash owl_data/nn/launch_vllm_parallel.sh")
        return

    print(f"\n✓ {servers_available}/{NUM_SERVERS} servers available")
    print()

    results = {
        "single_server": {},
        "parallel_servers": {}
    }

    for batch_size in BATCH_SIZES:
        print(f"Testing batch size: {batch_size}")
        print("-" * 40)

        # Warmup
        print(f"  Warming up...")
        for _ in range(NUM_WARMUP):
            if servers_available == 1:
                asyncio.run(test_single_server(min(batch_size, 8)))
            else:
                asyncio.run(test_parallel_servers(min(batch_size, 8)))

        # Test single server (if only one available, or for comparison)
        if batch_size <= 16:  # Don't test huge batches on single server
            print(f"  Testing single server...")
            times = []
            for run in range(NUM_RUNS):
                duration, count = asyncio.run(test_single_server(batch_size))
                times.append(duration)
                throughput = count / duration
                print(f"    Run {run+1}: {duration:.2f}s ({throughput:.2f} captions/sec)")

            avg_time = np.mean(times)
            avg_throughput = batch_size / avg_time
            results["single_server"][batch_size] = {
                "avg_time": avg_time,
                "throughput": avg_throughput
            }
            print(f"  Single server avg: {avg_time:.2f}s ({avg_throughput:.2f} captions/sec)")

        # Test parallel servers
        if servers_available > 1:
            print(f"  Testing {servers_available} parallel servers...")
            times = []
            for run in range(NUM_RUNS):
                duration, count = asyncio.run(test_parallel_servers(batch_size))
                times.append(duration)
                throughput = count / duration
                print(f"    Run {run+1}: {duration:.2f}s ({throughput:.2f} captions/sec)")

            avg_time = np.mean(times)
            avg_throughput = batch_size / avg_time
            results["parallel_servers"][batch_size] = {
                "avg_time": avg_time,
                "throughput": avg_throughput
            }
            print(f"  Parallel servers avg: {avg_time:.2f}s ({avg_throughput:.2f} captions/sec)")

        print()

    # Print summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    print("Throughput (captions/second):")
    print("-" * 60)
    print(f"{'Batch Size':<12} {'Single Server':<20} {'Parallel Servers':<20} {'Speedup':<10}")
    print("-" * 60)

    for batch_size in BATCH_SIZES:
        single = results["single_server"].get(batch_size, {}).get("throughput", 0)
        parallel = results["parallel_servers"].get(batch_size, {}).get("throughput", 0)

        if single > 0 and parallel > 0:
            speedup = parallel / single
            print(f"{batch_size:<12} {single:<20.2f} {parallel:<20.2f} {speedup:<10.2f}x")
        elif parallel > 0:
            print(f"{batch_size:<12} {'N/A':<20} {parallel:<20.2f} {'N/A':<10}")
        elif single > 0:
            print(f"{batch_size:<12} {single:<20.2f} {'N/A':<20} {'N/A':<10}")

    print()
    print("✓ Benchmark complete!")

    # Save results
    with open("parallel_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: parallel_benchmark_results.json")

if __name__ == "__main__":
    run_benchmark()