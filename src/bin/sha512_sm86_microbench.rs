use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::error::Error;
use std::sync::Arc;
use std::time::Instant;

const KERNEL_SRC: &str = include_str!("../../research/sha512_sm86/kernel.cu");

const ABC_BLOCK: [u64; 16] = [
    0x6162638000000000,
    0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0,
    24,
];

const ABC_SHA512: [u64; 8] = [
    0xddaf35a193617aba,
    0xcc417349ae204131,
    0x12e6fa4e89a97ea2,
    0x0a9eeee64b55d39a,
    0x2192992a274fc1a8,
    0x36ba3c23a3feebbd,
    0x454d4423643ce80e,
    0x2a9ac94fa54ca49f,
];

struct GpuBench {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    u64_kernel: CudaFunction,
    pair_kernel: CudaFunction,
}

impl GpuBench {
    fn new() -> Result<Self, Box<dyn Error>> {
        let opts = CompileOptions {
            arch: Some("compute_86"),
            use_fast_math: Some(false),
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(KERNEL_SRC, opts)?;
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let module = ctx.load_module(ptx)?;
        let u64_kernel = module.load_function("sha512_u64_kernel")?;
        let pair_kernel = module.load_function("sha512_pair32_kernel")?;
        Ok(Self {
            ctx,
            stream,
            u64_kernel,
            pair_kernel,
        })
    }

    fn print_resources(&self) -> Result<(), Box<dyn Error>> {
        println!("Device: {}", self.ctx.name()?);
        for (name, f) in [
            ("u64", &self.u64_kernel),
            ("pair32", &self.pair_kernel),
        ] {
            println!(
                "Resources {name}: regs/thread={} local/thread={}B shared/block={}B max_threads/block={}",
                f.num_regs()?,
                f.local_size_bytes()?,
                f.shared_size_bytes()?,
                f.max_threads_per_block()?,
            );
        }
        Ok(())
    }

    fn launch(
        &self,
        f: &CudaFunction,
        d_block: &CudaSlice<u64>,
        d_out: &mut CudaSlice<u64>,
        threads: u32,
        block: u32,
        iterations: u32,
    ) -> Result<f64, Box<dyn Error>> {
        let grid = threads.div_ceil(block).max(1);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        self.stream.synchronize()?;
        let start = Instant::now();
        let mut launcher = self.stream.launch_builder(f);
        launcher.arg(d_block);
        launcher.arg(d_out);
        launcher.arg(&iterations);
        unsafe { launcher.launch(cfg)? };
        self.stream.synchronize()?;
        Ok(start.elapsed().as_secs_f64())
    }

    fn self_test_one(
        &self,
        name: &str,
        f: &CudaFunction,
    ) -> Result<(), Box<dyn Error>> {
        let d_block = self.stream.clone_htod(&ABC_BLOCK)?;
        let mut d_out = self.stream.alloc_zeros::<u64>(8)?;
        self.launch(f, &d_block, &mut d_out, 1, 1, 1)?;
        let got = self.stream.clone_dtoh(&d_out)?;
        if got.as_slice() != ABC_SHA512 {
            return Err(format!(
                "{name} self-test FAILED\nexpected={:016x?}\ngot={:016x?}",
                ABC_SHA512, got
            )
            .into());
        }
        println!("{name} self-test: PASS (SHA-512(\"abc\"))");
        Ok(())
    }

    fn self_test(&self) -> Result<(), Box<dyn Error>> {
        self.print_resources()?;
        self.self_test_one("u64", &self.u64_kernel)?;
        self.self_test_one("pair32", &self.pair_kernel)?;
        Ok(())
    }

    fn bench_one(
        &self,
        name: &str,
        f: &CudaFunction,
        threads: u32,
        block: u32,
        iterations: u32,
        samples: u32,
    ) -> Result<Vec<u64>, Box<dyn Error>> {
        let d_block = self.stream.clone_htod(&ABC_BLOCK)?;
        let mut d_out = self
            .stream
            .alloc_zeros::<u64>(threads as usize * 8)?;

        println!(
            "Bench {name}: threads={threads} block={block} iterations/thread={iterations} samples={samples}"
        );

        let mut all_secs = 0.0f64;
        let mut steady_secs = 0.0f64;
        let work = threads as f64 * iterations as f64;

        for sample in 0..samples {
            let secs = self.launch(
                f,
                &d_block,
                &mut d_out,
                threads,
                block,
                iterations,
            )?;
            let rate = work / secs;
            println!(
                "sample #{sample} elapsed={secs:.6}s rate={:.3} Mcompress/s",
                rate / 1_000_000.0
            );
            all_secs += secs;
            if sample > 0 {
                steady_secs += secs;
            }
        }

        let all_work = work * samples as f64;
        println!(
            "weighted_all = {:.3} Mcompress/s",
            all_work / all_secs / 1_000_000.0
        );
        if samples > 1 {
            let steady_work = work * (samples - 1) as f64;
            println!(
                "weighted_steady = {:.3} Mcompress/s",
                steady_work / steady_secs / 1_000_000.0
            );
        }

        Ok(self.stream.clone_dtoh(&d_out)?)
    }
}

#[derive(Clone, Copy)]
struct Args {
    threads: u32,
    block: u32,
    iterations: u32,
    samples: u32,
}

fn parse_u32(args: &[String], flag: &str, default: u32) -> Result<u32, Box<dyn Error>> {
    match args.iter().position(|x| x == flag) {
        Some(i) => {
            let raw = args
                .get(i + 1)
                .ok_or_else(|| format!("{flag} requires a value"))?;
            Ok(raw.parse()?)
        }
        None => Ok(default),
    }
}

fn usage() {
    eprintln!(
        "usage:\n  sha512_sm86_microbench --self-test\n  sha512_sm86_microbench --bench <u64|pair32|both> [--threads N] [--block N] [--iters N] [--samples N]"
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let argv: Vec<String> = std::env::args().collect();
    let gpu = GpuBench::new()?;

    if argv.iter().any(|a| a == "--self-test") {
        return gpu.self_test();
    }

    let bench_pos = match argv.iter().position(|a| a == "--bench") {
        Some(i) => i,
        None => {
            usage();
            return Ok(());
        }
    };
    let variant = argv.get(bench_pos + 1).map(String::as_str).unwrap_or("both");

    let args = Args {
        threads: parse_u32(&argv, "--threads", 262_144)?,
        block: parse_u32(&argv, "--block", 128)?,
        iterations: parse_u32(&argv, "--iters", 512)?,
        samples: parse_u32(&argv, "--samples", 5)?,
    };
    if args.threads == 0 || args.iterations == 0 || args.samples == 0 {
        return Err("threads/iters/samples must be non-zero".into());
    }
    if args.block == 0 || args.block > 1024 || args.block % 32 != 0 {
        return Err("block must be a non-zero warp multiple <= 1024".into());
    }

    gpu.self_test()?;

    match variant {
        "u64" => {
            let _ = gpu.bench_one(
                "u64",
                &gpu.u64_kernel,
                args.threads,
                args.block,
                args.iterations,
                args.samples,
            )?;
        }
        "pair32" => {
            let _ = gpu.bench_one(
                "pair32",
                &gpu.pair_kernel,
                args.threads,
                args.block,
                args.iterations,
                args.samples,
            )?;
        }
        "both" => {
            let a = gpu.bench_one(
                "u64",
                &gpu.u64_kernel,
                args.threads,
                args.block,
                args.iterations,
                args.samples,
            )?;
            let b = gpu.bench_one(
                "pair32",
                &gpu.pair_kernel,
                args.threads,
                args.block,
                args.iterations,
                args.samples,
            )?;
            if a != b {
                return Err("u64 and pair32 benchmark outputs differ".into());
            }
            println!("cross-check: PASS (u64 == pair32 output for benchmark workload)");
        }
        _ => {
            usage();
            return Err(format!("unknown benchmark variant {variant:?}").into());
        }
    }

    Ok(())
}
