#!/usr/bin/env python3
"""
Numba CUDA Benchmark
Outputs JSON results for the reporting system.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from numba import cuda

ARRAY_SIZE = 20_000_000
WARMUP_RUNS = 5
BENCH_RUNS = 10
SEED = 42
THREADS_PER_BLOCK = 256


@cuda.jit(fastmath=True)
def compute_cuda_fp64(arr, out):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for idx in range(start, arr.size, stride):
        val = arr[idx]
        out[idx] = math.sqrt(val * val + 1.0) * math.sin(val) + math.cos(val * 0.5)


@cuda.jit(fastmath=True)
def compute_cuda_fp32(arr, out):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for idx in range(start, arr.size, stride):
        val = arr[idx]
        out[idx] = math.sqrt(val * val + 1.0) * math.sin(val) + math.cos(val * 0.5)


def benchmark_cuda(kernel, d_arr, d_out, blocks, threads, warmup=WARMUP_RUNS, runs=BENCH_RUNS):
    for _ in range(warmup):
        kernel[blocks, threads](d_arr, d_out)
        cuda.synchronize()

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        kernel[blocks, threads](d_arr, d_out)
        cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    h_out = d_out.copy_to_host()
    return {
        "min": min(times),
        "max": max(times),
        "mean": sum(times) / len(times),
        "runs": times,
        "checksum": float(np.sum(h_out))
    }


def main():
    parser = argparse.ArgumentParser(description="Numba CUDA Benchmark")
    parser.add_argument("--output", "-o", type=str, help="Output JSON file")
    parser.add_argument("--size", type=int, default=ARRAY_SIZE, help="Array size")
    args = parser.parse_args()

    if not cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        sys.exit(1)

    device = cuda.get_current_device()
    
    np.random.seed(SEED)
    h_arr_f64 = np.ascontiguousarray(
        np.random.uniform(-10.0, 10.0, args.size).astype(np.float64)
    )
    h_arr_f32 = h_arr_f64.astype(np.float32)

    results = {
        "benchmark": "numba_cuda",
        "language": "python",
        "device": device.name.decode(),
        "compute_capability": list(device.compute_capability),
        "array_size": args.size,
        "warmup_runs": WARMUP_RUNS,
        "bench_runs": BENCH_RUNS,
        "implementations": {}
    }

    blocks = min((args.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK,
                 device.MULTIPROCESSOR_COUNT * 32)

    # FP64
    d_arr_f64 = cuda.to_device(h_arr_f64)
    d_out_f64 = cuda.device_array_like(h_arr_f64)
    results["implementations"]["numba_cuda_fp64"] = benchmark_cuda(
        compute_cuda_fp64, d_arr_f64, d_out_f64, blocks, THREADS_PER_BLOCK
    )
    results["implementations"]["numba_cuda_fp64"]["dtype"] = "float64"

    # FP32
    d_arr_f32 = cuda.to_device(h_arr_f32)
    d_out_f32 = cuda.device_array_like(h_arr_f32)
    results["implementations"]["numba_cuda_fp32"] = benchmark_cuda(
        compute_cuda_fp32, d_arr_f32, d_out_f32, blocks, THREADS_PER_BLOCK
    )
    results["implementations"]["numba_cuda_fp32"]["dtype"] = "float32"

    output = json.dumps(results, indent=2)
    if args.output:
        Path(args.output).write_text(output)
        print(f"Results written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
