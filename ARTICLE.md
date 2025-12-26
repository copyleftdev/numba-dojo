---
title: "When I Took Numba to the Dojo: A Battle Royale Against Rust and CUDA"
published: true
description: "Expanding on @zenoguy's Numba experiments by throwing Rust and CUDA into the ring. Who wins when we push 20 million elements to their limits?"
tags: python, rust, cuda, performance
cover_image: https://raw.githubusercontent.com/copyleftdev/numba-dojo/main/media/logo.png
canonical_url: null
series: "Performance Exploration"
---

## A Thank You Note

Before we dive in, I want to acknowledge [Shreyan Ghosh (@zenoguy)](https://dev.to/zenoguy) and his wonderful article ["When Time Became a Variable — Notes From My Journey With Numba"](https://dev.to/zenoguy/when-time-became-a-variable-notes-from-my-journey-with-numba-57oj).

His piece captured something beautiful about computing: the joy of experimentation, the thrill of watching code go *fast*, and the curiosity to ask "what if?"

This line stuck with me:

> *"Somewhere between algorithms and hardware, Numba didn't just make my code faster. It made exploration lighter."*

Reading his benchmarks, I couldn't help but wonder: **What happens when we throw Rust into the mix? What about raw CUDA? Where does the hardware actually give up?**

So I built a dojo. Let's spar.

---

## 🎯 The Challenge

Same challenge as Shreyan's original experiment:

```
f(x) = sqrt(x² + 1) × sin(x) + cos(x/2)
```

Compute this for **20 million elements**.

Simple math. Maximum optimization. *Who wins?*

---

## 🥊 The Contenders

I assembled fighters from different worlds:

### Team Python 🐍
- **Pure Python** — The baseline. Interpreter overhead. GIL-bound.
- **NumPy Vectorized** — The standard approach.
- **Numba JIT** — Single-threaded compiled.
- **Numba Parallel** — Multi-threaded with `prange`.
- **Numba @vectorize** — Parallel ufunc magic.

### Team Rust 🦀
- **Single-threaded** — Idiomatic iterators.
- **Parallel (Rayon)** — Work-stealing parallelism.
- **Parallel Chunks** — Cache-optimized chunking.

### Team GPU 🎮
- **Numba CUDA** — Python on the GPU.
- **CUDA C++ FP64** — Double precision native.
- **CUDA C++ FP32** — Single precision native.
- **CUDA C++ Intrinsics** — Hardware-optimized math.

---

## 🏗️ The Setup

I wanted this to be **reproducible and fair**:

- Same computation across all implementations
- Same array size (20 million float64 elements)
- Same random seed (42, obviously)
- Multiple warmup runs to eliminate JIT/cache effects
- Take the **minimum** of multiple runs (least noise)

The full benchmark suite is open source: [**github.com/copyleftdev/numba-dojo**](https://github.com/copyleftdev/numba-dojo)

```bash
# Run everything yourself
git clone https://github.com/copyleftdev/numba-dojo.git
cd numba-dojo
make all
```

---

## 📊 The Results

Let's see who survived the dojo.

### The Full Leaderboard

![Benchmark Results](https://raw.githubusercontent.com/copyleftdev/numba-dojo/main/results/benchmark_bars.png)

| Rank | Implementation | Time | Speedup vs NumPy |
|------|---------------|------|------------------|
| 🥇 | CUDA C++ FP32 | 0.21 ms | **3,255x** |
| 🥈 | Numba CUDA FP32 | 2.52 ms | 265x |
| 🥉 | CUDA C++ FP64 | 4.11 ms | 162x |
| 4 | Numba CUDA FP64 | 4.14 ms | 161x |
| 5 | Rust Parallel | 12.39 ms | 54x |
| 6 | Numba @vectorize | 14.86 ms | 45x |
| 7 | Numba Parallel | 15.55 ms | 43x |
| 8 | Rust Single | 555.62 ms | 1.2x |
| 9 | Numba JIT | 558.30 ms | 1.2x |
| 10 | NumPy Vectorized | 667.30 ms | 1.0x |
| 11 | Pure Python | ~6,650 ms | 0.1x |

### Speedup Visualization

![Speedup Chart](https://raw.githubusercontent.com/copyleftdev/numba-dojo/main/results/speedup_chart.png)

### Category Champions

![Category Comparison](https://raw.githubusercontent.com/copyleftdev/numba-dojo/main/results/category_comparison.png)

---

## 🔬 What I Learned

### 1. GPU Demolishes CPU (When It Fits)

The RTX 3080 Ti at **0.21ms** is **3,255x faster** than NumPy. That's not a typo.

For embarrassingly parallel workloads like element-wise computation, GPUs are in a different league. The massive parallelism (80 streaming multiprocessors, thousands of cores) absolutely crushes sequential execution.

### 2. FP32 is 20x Faster Than FP64 on Consumer GPUs

```
CUDA FP64: 4.11 ms
CUDA FP32: 0.21 ms  ← 20x faster!
```

Consumer GPUs (GeForce series) have limited FP64 units — typically 1/32 the throughput of FP32. If your computation can tolerate single precision, **use it**.

### 3. Rust ≈ Numba JIT (Single-Threaded)

```
Rust Single:  555.62 ms
Numba JIT:    558.30 ms
```

Both compile to LLVM IR. Both get similar codegen. The difference is noise. This validates Numba's claim: **"Feels like Python, behaves like C."**

### 4. Rust Beats Numba in Parallel (~20%)

```
Rust Parallel (Rayon):  12.39 ms
Numba Parallel:         15.55 ms
```

Rayon's work-stealing scheduler has lower overhead than Numba's threading. For CPU-parallel workloads in production, Rust has an edge.

### 5. We Hit the Memory Bandwidth Wall

This was the most interesting discovery.

When I profiled the FP32 CUDA kernel:

```
Time:       0.21 ms
Bandwidth:  ~777 GB/s achieved
Theoretical: 912 GB/s (RTX 3080 Ti)
Efficiency: 85%
```

We're running at **85% of peak memory bandwidth**. The GPU cores are actually *waiting for data*. No algorithm can beat physics.

This is the [Roofline Model](https://en.wikipedia.org/wiki/Roofline_model) in action:

```
                    Peak Compute
                         /
                        /
Performance            /
                      /  ← We're here (memory-bound)
                     /
                    /
            ──────────────────────
                Memory Bandwidth
```

For this workload with low arithmetic intensity (few ops per byte), we've hit the ceiling.

---

## 🧪 The Code

Here's what each implementation looks like:

### Numba (The Hero of The Original Article)

```python
from numba import njit, prange
import numpy as np

@njit(parallel=True, fastmath=True, cache=True)
def compute_numba_parallel(arr, out):
    n = len(arr)
    for i in prange(n):
        val = arr[i]
        out[i] = np.sqrt(val * val + 1.0) * np.sin(val) + np.cos(val * 0.5)
```

Just add `@njit`. That's it. Shreyan was right — this is magical.

### Rust (The Challenger)

```rust
use rayon::prelude::*;

fn compute_parallel(arr: &[f64], out: &mut [f64]) {
    out.par_iter_mut()
        .zip(arr.par_iter())
        .for_each(|(o, &v)| {
            *o = (v * v + 1.0).sqrt() * v.sin() + (v * 0.5).cos();
        });
}
```

Rayon makes parallelism feel as natural as iterators.

### CUDA C++ (The Champion)

```cuda
__global__ void compute_fp32(const float* arr, float* out, size_t n) {
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += blockDim.x * gridDim.x) {
        float val = arr[idx];
        out[idx] = sqrtf(val * val + 1.0f) * sinf(val) + cosf(val * 0.5f);
    }
}
```

Grid-stride loops for maximum occupancy.

---

## 🎯 When to Use What

| Scenario | Recommendation |
|----------|----------------|
| Quick prototyping | NumPy (it's fine, really) |
| Need 10-50x speedup, stay in Python | Numba parallel |
| Production CPU workloads | Rust + Rayon |
| Maximum performance, GPU available | CUDA (FP32 if possible) |
| GPU + Python ecosystem | Numba CUDA |

---

## 🙏 Final Thoughts

Shreyan's original article reminded me why I love computing: **we get to ask "what if?" and then actually find out.**

What if we compile this loop? *43x faster.*
What if we use all CPU cores? *54x faster.*
What if we throw a GPU at it? *3,255x faster.*
What if we hit the memory bandwidth wall? *Physics wins.*

The journey from Pure Python (6.6 seconds) to CUDA FP32 (0.2 milliseconds) is a **33,000x improvement**. That's not optimization — that's transformation.

---

## 🔗 Resources

- **Full source code**: [github.com/copyleftdev/numba-dojo](https://github.com/copyleftdev/numba-dojo)
- **Original inspiration**: [@zenoguy's Numba article](https://dev.to/zenoguy/when-time-became-a-variable-notes-from-my-journey-with-numba-57oj)
- **Numba docs**: [numba.pydata.org](https://numba.pydata.org/)
- **Rayon (Rust)**: [docs.rs/rayon](https://docs.rs/rayon/)
- **Roofline Model**: [Wikipedia](https://en.wikipedia.org/wiki/Roofline_model)

---

*Keep experimenting. Keep playing. That's what computing is for.* ✨

---

**What's your favorite performance optimization story? Drop it in the comments!**
