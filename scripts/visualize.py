#!/usr/bin/env python3
"""
Benchmark Visualization Script
Generates charts from benchmark JSON results.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Style configuration
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = {
    'python': '#3776ab',
    'rust': '#dea584', 
    'cuda_cpp': '#76b900',
    'numba_cuda': '#8bc34a',
}

CATEGORY_COLORS = {
    'pure_python': '#ff6b6b',
    'numpy': '#4ecdc4',
    'numba_cpu': '#3776ab',
    'rust_cpu': '#dea584',
    'cuda_gpu': '#76b900',
}


def load_results(results_dir: Path) -> dict:
    """Load all JSON result files from the results directory."""
    results = {}
    for json_file in results_dir.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)
            results[json_file.stem] = data
    return results


def extract_all_implementations(results: dict) -> list:
    """Extract all implementations with their timings."""
    implementations = []
    
    for bench_name, data in results.items():
        language = data.get("language", "unknown")
        for impl_name, impl_data in data.get("implementations", {}).items():
            min_time = impl_data.get("min", impl_data.get("estimated", 0))
            
            # Categorize
            if "pure_python" in impl_name:
                category = "pure_python"
            elif "numpy" in impl_name:
                category = "numpy"
            elif "cuda" in impl_name.lower() or "gpu" in bench_name.lower():
                category = "cuda_gpu"
            elif "rust" in impl_name.lower() or language == "rust":
                category = "rust_cpu"
            else:
                category = "numba_cpu"
            
            implementations.append({
                "name": impl_name,
                "benchmark": bench_name,
                "language": language,
                "category": category,
                "min_time": min_time,
                "dtype": impl_data.get("dtype", "float64"),
                "estimated": impl_data.get("estimated", False),
            })
    
    return implementations


def create_bar_chart(implementations: list, output_path: Path):
    """Create a horizontal bar chart of all implementations."""
    # Sort by time (fastest first)
    sorted_impls = sorted(implementations, key=lambda x: x["min_time"])
    
    # Filter out pure python for better visualization (it's too slow)
    display_impls = [i for i in sorted_impls if i["category"] != "pure_python"]
    
    fig, ax = plt.subplots(figsize=(12, max(8, len(display_impls) * 0.4)))
    
    names = [f"{i['name']}" for i in display_impls]
    times = [i["min_time"] * 1000 for i in display_impls]  # Convert to ms
    colors = [CATEGORY_COLORS.get(i["category"], "#888888") for i in display_impls]
    
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, times, color=colors, edgecolor='white', linewidth=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Time (ms)", fontsize=12)
    ax.set_title("Benchmark Results: 20 Million Element Computation\n(Lower is Better)", 
                 fontsize=14, fontweight='bold')
    
    # Add time labels on bars
    for bar, time_val in zip(bars, times):
        width = bar.get_width()
        label = f"{time_val:.2f} ms" if time_val >= 1 else f"{time_val*1000:.2f} μs"
        ax.text(width + max(times) * 0.02, bar.get_y() + bar.get_height()/2,
                label, va='center', fontsize=9)
    
    # Legend
    legend_patches = [
        mpatches.Patch(color=CATEGORY_COLORS['numba_cpu'], label='Numba CPU'),
        mpatches.Patch(color=CATEGORY_COLORS['rust_cpu'], label='Rust CPU'),
        mpatches.Patch(color=CATEGORY_COLORS['cuda_gpu'], label='CUDA GPU'),
        mpatches.Patch(color=CATEGORY_COLORS['numpy'], label='NumPy'),
    ]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path / "benchmark_bars.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_path / "benchmark_bars.svg", bbox_inches='tight')
    plt.close()
    print(f"Created: {output_path / 'benchmark_bars.png'}")


def create_speedup_chart(implementations: list, output_path: Path):
    """Create a speedup chart relative to NumPy baseline."""
    # Find numpy baseline
    numpy_impl = next((i for i in implementations if "numpy" in i["name"].lower()), None)
    if not numpy_impl:
        print("No NumPy baseline found, skipping speedup chart")
        return
    
    baseline_time = numpy_impl["min_time"]
    
    # Calculate speedups
    sorted_impls = sorted(implementations, key=lambda x: x["min_time"])
    display_impls = [i for i in sorted_impls if i["category"] != "pure_python"]
    
    fig, ax = plt.subplots(figsize=(12, max(8, len(display_impls) * 0.4)))
    
    names = [i["name"] for i in display_impls]
    speedups = [baseline_time / i["min_time"] for i in display_impls]
    colors = [CATEGORY_COLORS.get(i["category"], "#888888") for i in display_impls]
    
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, speedups, color=colors, edgecolor='white', linewidth=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Speedup vs NumPy", fontsize=12)
    ax.set_title("Speedup Relative to NumPy Vectorized\n(Higher is Better)", 
                 fontsize=14, fontweight='bold')
    ax.axvline(x=1, color='red', linestyle='--', linewidth=1, alpha=0.7, label='NumPy baseline')
    
    # Add speedup labels
    for bar, speedup in zip(bars, speedups):
        width = bar.get_width()
        ax.text(width + max(speedups) * 0.02, bar.get_y() + bar.get_height()/2,
                f"{speedup:.1f}x", va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path / "speedup_chart.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_path / "speedup_chart.svg", bbox_inches='tight')
    plt.close()
    print(f"Created: {output_path / 'speedup_chart.png'}")


def create_category_comparison(implementations: list, output_path: Path):
    """Create a grouped bar chart comparing categories."""
    categories = {}
    for impl in implementations:
        cat = impl["category"]
        if cat not in categories or impl["min_time"] < categories[cat]["min_time"]:
            categories[cat] = impl
    
    # Order categories
    category_order = ["pure_python", "numpy", "numba_cpu", "rust_cpu", "cuda_gpu"]
    ordered_cats = [c for c in category_order if c in categories]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = [c.replace("_", " ").title() for c in ordered_cats]
    times = [categories[c]["min_time"] * 1000 for c in ordered_cats]
    colors = [CATEGORY_COLORS[c] for c in ordered_cats]
    
    x_pos = np.arange(len(names))
    bars = ax.bar(x_pos, times, color=colors, edgecolor='white', linewidth=0.5)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.set_yscale('log')
    ax.set_title("Best Time per Category (Log Scale)\n(Lower is Better)", 
                 fontsize=14, fontweight='bold')
    
    # Add time labels
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        label = f"{time_val:.2f}ms" if time_val >= 1 else f"{time_val*1000:.1f}μs"
        ax.text(bar.get_x() + bar.get_width()/2, height * 1.1,
                label, ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path / "category_comparison.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_path / "category_comparison.svg", bbox_inches='tight')
    plt.close()
    print(f"Created: {output_path / 'category_comparison.png'}")


def create_summary_table(implementations: list, output_path: Path):
    """Create a markdown summary table."""
    sorted_impls = sorted(implementations, key=lambda x: x["min_time"])
    
    # Find baseline
    numpy_impl = next((i for i in implementations if "numpy" in i["name"].lower()), None)
    baseline_time = numpy_impl["min_time"] if numpy_impl else sorted_impls[-1]["min_time"]
    
    lines = [
        "# Benchmark Results Summary",
        "",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
        "## All Implementations (Sorted by Speed)",
        "",
        "| Rank | Implementation | Time | Speedup vs NumPy | Category |",
        "|------|---------------|------|------------------|----------|",
    ]
    
    for rank, impl in enumerate(sorted_impls, 1):
        time_str = f"{impl['min_time']*1000:.4f} ms" if impl['min_time'] >= 0.001 else f"{impl['min_time']*1e6:.2f} μs"
        speedup = baseline_time / impl["min_time"]
        est_marker = " *(est)*" if impl.get("estimated") else ""
        lines.append(f"| {rank} | {impl['name']}{est_marker} | {time_str} | {speedup:.1f}x | {impl['category']} |")
    
    lines.extend([
        "",
        "## Charts",
        "",
        "![Benchmark Results](benchmark_bars.png)",
        "",
        "![Speedup Chart](speedup_chart.png)",
        "",
        "![Category Comparison](category_comparison.png)",
    ])
    
    summary_path = output_path / "RESULTS.md"
    summary_path.write_text("\n".join(lines))
    print(f"Created: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark visualizations")
    parser.add_argument("--results-dir", "-r", type=str, default="results",
                        help="Directory containing JSON result files")
    parser.add_argument("--output-dir", "-o", type=str, default="results",
                        help="Output directory for charts")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.exists():
        print(f"Error: Results directory '{results_dir}' not found", file=sys.stderr)
        sys.exit(1)

    print(f"Loading results from {results_dir}...")
    results = load_results(results_dir)
    
    if not results:
        print("No result files found!", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(results)} result files")
    implementations = extract_all_implementations(results)
    print(f"Found {len(implementations)} implementations")

    print("\nGenerating visualizations...")
    create_bar_chart(implementations, output_dir)
    create_speedup_chart(implementations, output_dir)
    create_category_comparison(implementations, output_dir)
    create_summary_table(implementations, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
