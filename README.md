# 🥋 Numba Dojo

[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?logo=python&logoColor=white)](https://python.org)
[![Rust](https://img.shields.io/badge/Rust-1.70+-dea584?logo=rust&logoColor=white)](https://rust-lang.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.0+-76b900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A Performance Battle Royale: Python vs Rust vs CUDA**

An experimental benchmark suite exploring the boundaries of numerical computation performance across languages, paradigms, and hardware. This project asks a simple question: *How fast can we go?*

> *"Somewhere between algorithms and hardware, Numba didn't just make my code faster. It made exploration lighter."*  
> — [Shreyan Ghosh (@zenoguy)](https://dev.to/zenoguy)

This project is a direct response to Shreyan's wonderful article ["When Time Became a Variable"](https://dev.to/zenoguy/when-time-became-a-variable-notes-from-my-journey-with-numba-57oj), expanding his Numba experiments into a cross-language performance exploration.

## 🎯 The Challenge

Compute `sqrt(x² + 1) * sin(x) + cos(x/2)` for **20 million elements**.

Simple math. Maximum optimization. Who wins?

## 🏆 Results Preview

| Category | Winner | Time | Speedup vs NumPy |
|----------|--------|------|------------------|
| **GPU (FP32)** | CUDA C++ | ~0.2ms | ~3,000x |
| **GPU (FP64)** | CUDA C++ | ~3.4ms | ~200x |
| **CPU Parallel** | Rust (Rayon) | ~13ms | ~50x |
| **CPU Single** | Rust / Numba JIT | ~560ms | ~1.2x |
| **Baseline** | NumPy Vectorized | ~670ms | 1x |

## 📁 Project Structure

```
numba-dojo/
├── benchmarks/
│   ├── python/         # Numba CPU & CUDA benchmarks
│   ├── rust/           # Rust + Rayon benchmark
│   └── cuda/           # CUDA C++ benchmark
├── scripts/
│   ├── run_benchmarks.py   # Benchmark runner
│   └── visualize.py        # Chart generation
├── results/            # JSON results & charts (generated)
├── Makefile            # Build & run automation
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** with pip
- **Rust** (via [rustup](https://rustup.rs/))
- **CUDA Toolkit** (optional, for GPU benchmarks)

### Run Everything

```bash
# Clone the repo
git clone https://github.com/yourusername/numba-dojo.git
cd numba-dojo

# Build all benchmarks, run them, and generate charts
make all
```

That's it! Results will be in `results/`.

### Individual Steps

```bash
# Just build
make build

# Just run benchmarks
make run

# Just generate visualizations
make visualize

# Check what's available
make help
```

## 📊 What Gets Benchmarked

### Python (Numba)
- **Pure Python** - Baseline loop (extrapolated from 100k elements)
- **NumPy Vectorized** - Standard NumPy operations
- **Numba JIT** - Single-threaded compiled
- **Numba Parallel** - Multi-threaded with `prange`
- **Numba @vectorize** - Parallel ufunc

### Python (Numba CUDA)
- **Numba CUDA FP64** - Double precision GPU
- **Numba CUDA FP32** - Single precision GPU

### Rust
- **Single-threaded** - Idiomatic iterators
- **Parallel (Rayon)** - Work-stealing parallelism
- **Parallel Chunks** - Cache-optimized chunks

### CUDA C++
- **FP64** - Double precision
- **FP32** - Single precision
- **FP32 Intrinsics** - Hardware-optimized math

## 🔬 Key Findings

1. **GPU crushes CPU** for embarrassingly parallel workloads
2. **FP32 is 10-20x faster than FP64** on consumer GPUs (limited FP64 units)
3. **Rust ≈ Numba JIT** for single-threaded (both use LLVM)
4. **Rust beats Numba** by ~20% in parallel (Rayon vs Numba threading)
5. **Memory bandwidth is the limit** - FP32 CUDA hits 85% of theoretical bandwidth

## 🛠 Configuration

Override array size:

```bash
make run ARRAY_SIZE=50000000
```

Skip specific benchmarks:

```bash
python scripts/run_benchmarks.py --skip-cuda --skip-rust
```

## 📈 Sample Output

After running `make all`, you'll get:

- `results/RESULTS.md` - Markdown summary table
- `results/benchmark_bars.png` - Horizontal bar chart
- `results/speedup_chart.png` - Speedup vs NumPy
- `results/category_comparison.png` - Best per category
- `results/*.json` - Raw benchmark data

## 🔧 Development

```bash
# Check all dependencies
make check-deps

# Lint Rust code
make lint-rust

# Format Rust code
make fmt-rust

# Clean everything
make clean-all
```

## 📚 References

- [Numba Documentation](https://numba.pydata.org/)
- [Rayon (Rust)](https://docs.rs/rayon/latest/rayon/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Roofline Model](https://en.wikipedia.org/wiki/Roofline_model)

## 📄 License

MIT License — Use it, learn from it, build on it.

## 🙏 Acknowledgments

This project exists because of the spirit of open exploration in computing.

**Special thanks to [Shreyan Ghosh (@zenoguy)](https://dev.to/zenoguy)** for his article ["When Time Became a Variable — Notes From My Journey With Numba"](https://dev.to/zenoguy/when-time-became-a-variable-notes-from-my-journey-with-numba-57oj). His playful, experimental approach to performance optimization reminded me why computing is beautiful: we get to ask "what if?" and then actually find out.

Shreyan's piece wasn't just about Numba — it was about curiosity, about treating performance as part of expression, about the joy of watching something go *fast*. This project is my way of vibing off that energy and pushing the experiment further: What happens when we throw Rust into the mix? What about raw CUDA? Where does the hardware actually give up?

The answer, as it turns out, is *memory bandwidth*. But the journey there? That's the fun part.

---

**Other resources that made this possible:**
- [Numba](https://numba.pydata.org/) — The JIT that started it all
- [Rayon](https://docs.rs/rayon/) — Rust's beautiful parallel iterator library
- [NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit) — For letting us talk to the GPU
- The [Roofline Model](https://en.wikipedia.org/wiki/Roofline_model) — For helping us understand *why* we can't go faster

---

*Keep experimenting. Keep playing. That's what computing is for.* ✨
