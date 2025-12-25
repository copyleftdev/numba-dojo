#!/usr/bin/env python3
"""
Numba CPU Benchmark
Outputs JSON results for the reporting system.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit, prange, vectorize, float64, get_num_threads

ARRAY_SIZE = 20_000_000
WARMUP_RUNS = 3
BENCH_RUNS = 5
SEED = 42


def compute_pure_python(arr, out):
    for i in range(len(arr)):
        val = arr[i]
        out[i] = math.sqrt(val * val + 1.0) * math.sin(val) + math.cos(val * 0.5)


@njit(cache=True, boundscheck=False, fastmath=True)
def compute_numba_jit(arr, out):
    n = len(arr)
    for i in range(n):
        val = arr[i]
        out[i] = np.sqrt(val * val + 1.0) * np.sin(val) + np.cos(val * 0.5)


@njit(parallel=True, cache=True, boundscheck=False, fastmath=True, error_model='numpy')
def compute_numba_parallel(arr, out):
    n = len(arr)
    for i in prange(n):
        val = arr[i]
        out[i] = np.sqrt(val * val + 1.0) * np.sin(val) + np.cos(val * 0.5)


@vectorize([float64(float64)], target='parallel', cache=True, fastmath=True)
def compute_vectorize_parallel(val):
    return np.sqrt(val * val + 1.0) * np.sin(val) + np.cos(val * 0.5)


def benchmark_inplace(func, arr, out, warmup=WARMUP_RUNS, runs=BENCH_RUNS):
    for _ in range(warmup):
        func(arr, out)
    
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        func(arr, out)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    return {
        "min": min(times),
        "max": max(times),
        "mean": sum(times) / len(times),
        "runs": times,
        "checksum": float(np.sum(out))
    }


def benchmark_return(func, arr, warmup=WARMUP_RUNS, runs=BENCH_RUNS):
    for _ in range(warmup):
        _ = func(arr)
    
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        result = func(arr)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    return {
        "min": min(times),
        "max": max(times),
        "mean": sum(times) / len(times),
        "runs": times,
        "checksum": float(np.sum(result))
    }


def main():
    parser = argparse.ArgumentParser(description="Numba CPU Benchmark")
    parser.add_argument("--output", "-o", type=str, help="Output JSON file")
    parser.add_argument("--size", type=int, default=ARRAY_SIZE, help="Array size")
    args = parser.parse_args()

    np.random.seed(SEED)
    arr = np.ascontiguousarray(
        np.random.uniform(-10.0, 10.0, args.size).astype(np.float64)
    )
    out = np.empty_like(arr)

    results = {
        "benchmark": "numba_cpu",
        "language": "python",
        "array_size": args.size,
        "dtype": "float64",
        "threads": get_num_threads(),
        "warmup_runs": WARMUP_RUNS,
        "bench_runs": BENCH_RUNS,
        "implementations": {}
    }

    # Pure Python (extrapolated)
    small_arr = arr[:100_000]
    small_out = np.empty_like(small_arr)
    start = time.perf_counter()
    compute_pure_python(small_arr, small_out)
    elapsed = time.perf_counter() - start
    estimated = elapsed * (args.size / 100_000)
    results["implementations"]["pure_python"] = {
        "min": estimated,
        "estimated": True,
        "sample_size": 100_000
    }

    # Numba JIT
    results["implementations"]["numba_jit"] = benchmark_inplace(
        compute_numba_jit, arr, out
    )

    # Numba Parallel
    results["implementations"]["numba_parallel"] = benchmark_inplace(
        compute_numba_parallel, arr, out
    )

    # Numba @vectorize parallel
    results["implementations"]["numba_vectorize_parallel"] = benchmark_return(
        compute_vectorize_parallel, arr
    )

    # NumPy vectorized
    times = []
    for _ in range(BENCH_RUNS):
        start = time.perf_counter()
        result = np.sqrt(arr * arr + 1.0) * np.sin(arr) + np.cos(arr * 0.5)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    results["implementations"]["numpy_vectorized"] = {
        "min": min(times),
        "max": max(times),
        "mean": sum(times) / len(times),
        "runs": times,
        "checksum": float(np.sum(result))
    }

    # Output
    output = json.dumps(results, indent=2)
    if args.output:
        Path(args.output).write_text(output)
        print(f"Results written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
