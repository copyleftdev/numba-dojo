/**
 * CUDA C++ Benchmark
 * Outputs JSON results for the reporting system.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <chrono>
#include <vector>
#include <random>
#include <string>

constexpr size_t ARRAY_SIZE = 20'000'000;
constexpr int WARMUP_RUNS = 5;
constexpr int BENCH_RUNS = 10;
constexpr int BLOCK_SIZE = 256;
constexpr uint64_t SEED = 42;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

__global__ void compute_fp64(const double* __restrict__ arr,
                             double* __restrict__ out,
                             size_t n) {
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += blockDim.x * gridDim.x) {
        double val = arr[idx];
        out[idx] = sqrt(val * val + 1.0) * sin(val) + cos(val * 0.5);
    }
}

__global__ void compute_fp32(const float* __restrict__ arr,
                             float* __restrict__ out,
                             size_t n) {
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += blockDim.x * gridDim.x) {
        float val = arr[idx];
        out[idx] = sqrtf(val * val + 1.0f) * sinf(val) + cosf(val * 0.5f);
    }
}

__global__ void compute_fp32_intrinsics(const float* __restrict__ arr,
                                         float* __restrict__ out,
                                         size_t n) {
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += blockDim.x * gridDim.x) {
        float val = arr[idx];
        out[idx] = __fsqrt_rn(__fmaf_rn(val, val, 1.0f)) * __sinf(val) + __cosf(val * 0.5f);
    }
}

struct BenchResult {
    double min_time;
    double max_time;
    double mean_time;
    std::vector<double> runs;
    double checksum;
    std::string dtype;
};

template<typename T, typename KernelFunc>
BenchResult benchmark_kernel(KernelFunc kernel,
                              const T* d_arr,
                              T* d_out,
                              T* h_out,
                              size_t n,
                              int num_blocks,
                              const char* dtype) {
    // Warmup
    for (int i = 0; i < WARMUP_RUNS; i++) {
        kernel<<<num_blocks, BLOCK_SIZE>>>(d_arr, d_out, n);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Benchmark
    std::vector<double> times;
    times.reserve(BENCH_RUNS);
    
    for (int i = 0; i < BENCH_RUNS; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        kernel<<<num_blocks, BLOCK_SIZE>>>(d_arr, d_out, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double>(end - start).count());
    }

    CUDA_CHECK(cudaMemcpy(h_out, d_out, n * sizeof(T), cudaMemcpyDeviceToHost));
    
    double checksum = 0.0;
    for (size_t i = 0; i < n; i++) {
        checksum += static_cast<double>(h_out[i]);
    }

    double min_t = times[0], max_t = times[0], sum_t = 0.0;
    for (double t : times) {
        min_t = std::min(min_t, t);
        max_t = std::max(max_t, t);
        sum_t += t;
    }

    return BenchResult{min_t, max_t, sum_t / times.size(), times, checksum, dtype};
}

void print_json_array(FILE* f, const std::vector<double>& arr) {
    fprintf(f, "[");
    for (size_t i = 0; i < arr.size(); i++) {
        fprintf(f, "%.6f%s", arr[i], i < arr.size() - 1 ? ", " : "");
    }
    fprintf(f, "]");
}

int main(int argc, char** argv) {
    const char* output_path = nullptr;
    size_t array_size = ARRAY_SIZE;

    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) && i + 1 < argc) {
            output_path = argv[++i];
        } else if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            array_size = std::stoull(argv[++i]);
        }
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    // Generate data
    std::mt19937_64 rng(SEED);
    std::uniform_real_distribution<double> dist(-10.0, 10.0);
    
    std::vector<double> h_arr_f64(array_size);
    std::vector<double> h_out_f64(array_size);
    std::vector<float> h_arr_f32(array_size);
    std::vector<float> h_out_f32(array_size);

    for (size_t i = 0; i < array_size; i++) {
        h_arr_f64[i] = dist(rng);
        h_arr_f32[i] = static_cast<float>(h_arr_f64[i]);
    }

    // Allocate GPU memory
    double *d_arr_f64, *d_out_f64;
    float *d_arr_f32, *d_out_f32;
    
    CUDA_CHECK(cudaMalloc(&d_arr_f64, array_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out_f64, array_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_arr_f32, array_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_f32, array_size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_arr_f64, h_arr_f64.data(), array_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_arr_f32, h_arr_f32.data(), array_size * sizeof(float), cudaMemcpyHostToDevice));

    int num_blocks = prop.multiProcessorCount * 32;

    // Run benchmarks
    std::vector<std::pair<std::string, BenchResult>> results;
    
    results.push_back({"cuda_fp64", benchmark_kernel(compute_fp64, d_arr_f64, d_out_f64, 
                                                      h_out_f64.data(), array_size, num_blocks, "float64")});
    results.push_back({"cuda_fp32", benchmark_kernel(compute_fp32, d_arr_f32, d_out_f32,
                                                      h_out_f32.data(), array_size, num_blocks, "float32")});
    results.push_back({"cuda_fp32_intrinsics", benchmark_kernel(compute_fp32_intrinsics, d_arr_f32, d_out_f32,
                                                                 h_out_f32.data(), array_size, num_blocks, "float32")});

    // Output JSON
    FILE* f = output_path ? fopen(output_path, "w") : stdout;
    if (!f) {
        fprintf(stderr, "Failed to open output file\n");
        return 1;
    }

    fprintf(f, "{\n");
    fprintf(f, "  \"benchmark\": \"cuda_cpp\",\n");
    fprintf(f, "  \"language\": \"cuda_cpp\",\n");
    fprintf(f, "  \"device\": \"%s\",\n", prop.name);
    fprintf(f, "  \"compute_capability\": [%d, %d],\n", prop.major, prop.minor);
    fprintf(f, "  \"multiprocessors\": %d,\n", prop.multiProcessorCount);
    fprintf(f, "  \"array_size\": %zu,\n", array_size);
    fprintf(f, "  \"warmup_runs\": %d,\n", WARMUP_RUNS);
    fprintf(f, "  \"bench_runs\": %d,\n", BENCH_RUNS);
    fprintf(f, "  \"implementations\": {\n");

    for (size_t i = 0; i < results.size(); i++) {
        const auto& [name, res] = results[i];
        fprintf(f, "    \"%s\": {\n", name.c_str());
        fprintf(f, "      \"min\": %.6f,\n", res.min_time);
        fprintf(f, "      \"max\": %.6f,\n", res.max_time);
        fprintf(f, "      \"mean\": %.6f,\n", res.mean_time);
        fprintf(f, "      \"runs\": ");
        print_json_array(f, res.runs);
        fprintf(f, ",\n");
        fprintf(f, "      \"checksum\": %.2f,\n", res.checksum);
        fprintf(f, "      \"dtype\": \"%s\"\n", res.dtype.c_str());
        fprintf(f, "    }%s\n", i < results.size() - 1 ? "," : "");
    }

    fprintf(f, "  }\n");
    fprintf(f, "}\n");

    if (output_path) {
        fclose(f);
        fprintf(stderr, "Results written to %s\n", output_path);
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_arr_f64));
    CUDA_CHECK(cudaFree(d_out_f64));
    CUDA_CHECK(cudaFree(d_arr_f32));
    CUDA_CHECK(cudaFree(d_out_f32));

    return 0;
}
