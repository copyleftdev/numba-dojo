#!/usr/bin/env python3
"""
Benchmark Runner Script
Executes all benchmarks and collects results.
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime


def run_command(cmd: list, cwd: Path = None, capture: bool = True) -> tuple:
    """Run a command and return (success, output)."""
    print(f"  Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=capture,
            text=True,
            timeout=300  # 5 minute timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Timeout"
    except Exception as e:
        return False, "", str(e)


def main():
    parser = argparse.ArgumentParser(description="Run all benchmarks")
    parser.add_argument("--results-dir", "-r", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--size", type=int, default=20_000_000,
                        help="Array size for benchmarks")
    parser.add_argument("--skip-rust", action="store_true", help="Skip Rust benchmark")
    parser.add_argument("--skip-cuda", action="store_true", help="Skip CUDA benchmarks")
    parser.add_argument("--skip-python", action="store_true", help="Skip Python benchmarks")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    results_dir = project_root / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BENCHMARK RUNNER")
    print(f"Array size: {args.size:,}")
    print(f"Results dir: {results_dir}")
    print("=" * 60)

    results_summary = {
        "timestamp": datetime.now().isoformat(),
        "array_size": args.size,
        "benchmarks_run": [],
        "benchmarks_failed": [],
    }

    # =========================================================================
    # Python/Numba CPU Benchmark
    # =========================================================================
    if not args.skip_python:
        print("\n[1/4] Running Numba CPU benchmark...")
        output_file = results_dir / "numba_cpu.json"
        success, stdout, stderr = run_command(
            [sys.executable, "benchmarks/python/bench_numba.py",
             "-o", str(output_file), "--size", str(args.size)],
            cwd=project_root
        )
        if success:
            print(f"  ✓ Results saved to {output_file}")
            results_summary["benchmarks_run"].append("numba_cpu")
        else:
            print(f"  ✗ Failed: {stderr}")
            results_summary["benchmarks_failed"].append("numba_cpu")

    # =========================================================================
    # Python/Numba CUDA Benchmark
    # =========================================================================
    if not args.skip_cuda and not args.skip_python:
        print("\n[2/4] Running Numba CUDA benchmark...")
        output_file = results_dir / "numba_cuda.json"
        success, stdout, stderr = run_command(
            [sys.executable, "benchmarks/python/bench_cuda.py",
             "-o", str(output_file), "--size", str(args.size)],
            cwd=project_root
        )
        if success:
            print(f"  ✓ Results saved to {output_file}")
            results_summary["benchmarks_run"].append("numba_cuda")
        else:
            print(f"  ✗ Failed: {stderr}")
            results_summary["benchmarks_failed"].append("numba_cuda")

    # =========================================================================
    # Rust Benchmark
    # =========================================================================
    if not args.skip_rust:
        print("\n[3/4] Running Rust benchmark...")
        rust_binary = project_root / "benchmarks/rust/target/release/bench-rust"
        output_file = results_dir / "rust_cpu.json"
        
        if rust_binary.exists():
            success, stdout, stderr = run_command(
                [str(rust_binary), "-o", str(output_file), "--size", str(args.size)],
                cwd=project_root
            )
            if success:
                print(f"  ✓ Results saved to {output_file}")
                results_summary["benchmarks_run"].append("rust_cpu")
            else:
                print(f"  ✗ Failed: {stderr}")
                results_summary["benchmarks_failed"].append("rust_cpu")
        else:
            print(f"  ✗ Binary not found. Run 'make build-rust' first.")
            results_summary["benchmarks_failed"].append("rust_cpu")

    # =========================================================================
    # CUDA C++ Benchmark
    # =========================================================================
    if not args.skip_cuda:
        print("\n[4/4] Running CUDA C++ benchmark...")
        cuda_binary = project_root / "benchmarks/cuda/benchmark"
        output_file = results_dir / "cuda_cpp.json"
        
        if cuda_binary.exists():
            success, stdout, stderr = run_command(
                [str(cuda_binary), "-o", str(output_file), "--size", str(args.size)],
                cwd=project_root
            )
            if success:
                print(f"  ✓ Results saved to {output_file}")
                results_summary["benchmarks_run"].append("cuda_cpp")
            else:
                print(f"  ✗ Failed: {stderr}")
                results_summary["benchmarks_failed"].append("cuda_cpp")
        else:
            print(f"  ✗ Binary not found. Run 'make build-cuda' first.")
            results_summary["benchmarks_failed"].append("cuda_cpp")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Successful: {len(results_summary['benchmarks_run'])}")
    print(f"Failed: {len(results_summary['benchmarks_failed'])}")
    
    # Save summary
    summary_file = results_dir / "run_summary.json"
    with open(summary_file, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nRun summary saved to {summary_file}")

    if results_summary["benchmarks_failed"]:
        print(f"\nFailed benchmarks: {', '.join(results_summary['benchmarks_failed'])}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
