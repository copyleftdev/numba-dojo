use clap::Parser;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::Serialize;
use std::collections::HashMap;
use std::fs;
use std::hint::black_box;
use std::time::Instant;

const ARRAY_SIZE: usize = 20_000_000;
const WARMUP_RUNS: usize = 3;
const BENCH_RUNS: usize = 5;
const SEED: u64 = 42;

#[derive(Parser)]
#[command(name = "bench-rust")]
#[command(about = "Rust CPU Benchmark")]
struct Args {
    #[arg(short, long)]
    output: Option<String>,

    #[arg(long, default_value_t = ARRAY_SIZE)]
    size: usize,
}

#[derive(Serialize)]
struct BenchResult {
    min: f64,
    max: f64,
    mean: f64,
    runs: Vec<f64>,
    checksum: f64,
}

#[derive(Serialize)]
struct Results {
    benchmark: String,
    language: String,
    array_size: usize,
    dtype: String,
    threads: usize,
    warmup_runs: usize,
    bench_runs: usize,
    implementations: HashMap<String, BenchResult>,
}

#[inline(always)]
fn compute_element(val: f64) -> f64 {
    val.mul_add(val, 1.0).sqrt() * val.sin() + (val * 0.5).cos()
}

#[inline(never)]
fn compute_single(arr: &[f64], out: &mut [f64]) {
    for (o, &v) in out.iter_mut().zip(arr.iter()) {
        *o = compute_element(v);
    }
}

#[inline(never)]
fn compute_parallel(arr: &[f64], out: &mut [f64]) {
    out.par_iter_mut()
        .zip(arr.par_iter())
        .for_each(|(o, &v)| {
            *o = compute_element(v);
        });
}

#[inline(never)]
fn compute_parallel_chunks(arr: &[f64], out: &mut [f64]) {
    const CHUNK_SIZE: usize = 8192;
    out.par_chunks_mut(CHUNK_SIZE)
        .zip(arr.par_chunks(CHUNK_SIZE))
        .for_each(|(out_chunk, arr_chunk)| {
            for i in 0..out_chunk.len() {
                unsafe {
                    let val = *arr_chunk.get_unchecked(i);
                    *out_chunk.get_unchecked_mut(i) = compute_element(val);
                }
            }
        });
}

fn benchmark<F>(arr: &[f64], out: &mut [f64], f: F) -> BenchResult
where
    F: Fn(&[f64], &mut [f64]),
{
    // Warmup
    for _ in 0..WARMUP_RUNS {
        f(arr, out);
        black_box(&out);
    }

    // Benchmark
    let mut times = Vec::with_capacity(BENCH_RUNS);
    for _ in 0..BENCH_RUNS {
        let start = Instant::now();
        f(arr, out);
        black_box(&out);
        times.push(start.elapsed().as_secs_f64());
    }

    let checksum: f64 = black_box(out.iter().sum());
    let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean = times.iter().sum::<f64>() / times.len() as f64;

    BenchResult {
        min,
        max,
        mean,
        runs: times,
        checksum,
    }
}

fn main() {
    let args = Args::parse();

    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let arr: Vec<f64> = (0..args.size)
        .map(|_| rng.gen_range(-10.0..10.0))
        .collect();
    let mut out = vec![0.0f64; args.size];

    let num_threads = rayon::current_num_threads();

    let mut results = Results {
        benchmark: "rust_cpu".to_string(),
        language: "rust".to_string(),
        array_size: args.size,
        dtype: "float64".to_string(),
        threads: num_threads,
        warmup_runs: WARMUP_RUNS,
        bench_runs: BENCH_RUNS,
        implementations: HashMap::new(),
    };

    // Single-threaded
    results
        .implementations
        .insert("rust_single".to_string(), benchmark(&arr, &mut out, compute_single));

    // Parallel
    results
        .implementations
        .insert("rust_parallel".to_string(), benchmark(&arr, &mut out, compute_parallel));

    // Parallel chunks
    results.implementations.insert(
        "rust_parallel_chunks".to_string(),
        benchmark(&arr, &mut out, compute_parallel_chunks),
    );

    let json = serde_json::to_string_pretty(&results).unwrap();

    if let Some(output_path) = args.output {
        fs::write(&output_path, &json).expect("Failed to write output file");
        eprintln!("Results written to {}", output_path);
    } else {
        println!("{}", json);
    }
}
