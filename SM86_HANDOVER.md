# RTX 3070 / SM86 optimization handover

Repository: `TnzGit/btc-seedphrase-recovery-3070`

Primary branch: `opt/sm86-rtx3070-v1`  
Core A/B branch: `opt/sm86-rtx3070-core`  
Baseline: `main`

## Goal

Optimize the existing CUDA recovery pipeline specifically for RTX 3070 / GA104 / SM86 while preserving exact BIP39/BIP32/secp256k1 correctness.

Do not judge changes only by standalone SHA-512 throughput. The acceptance metric is end-to-end `--bench` candidates/s after the built-in GPU self-test succeeds.

## What has already been changed

### 1. SHA-512 message schedule

The original `sha512_compress()` allocated `uint64_t W[80]` per thread.

The optimized branch replaces it with:

- 16 scalar `u64` schedule words
- a rolling SHA-512 schedule
- explicitly unrolled 80 compression rounds
- a `sha512_compress_words()` fast entry point for already-parsed 64-bit words

Intent: reduce register/local-memory pressure and eliminate the 80-word schedule array.

### 2. PBKDF2 fixed-64-byte HMAC fast path

PBKDF2 rounds U2..U2048 always hash an exact 64-byte SHA-512 digest.

The original implementation converted every digest to 64 bytes, rebuilt a 128-byte padded block, then parsed the bytes back into SHA-512 words.

The optimized path keeps U and T as `uint64_t[8]` and uses `hmac_sha512_finish_fixed64_words()`.

U1 still uses the generic path because the salt length is variable.

### 3. Removed 24 KiB/block shared-memory BIP39 wordlist copy

The original kernel copied all 24 KiB of the packed BIP39 wordlist into shared memory for every block.

On SM86 that can cap residency even though the table is only read once per candidate before the long PBKDF2 loop.

The optimized branch reads `d_wordlist` directly and relies on the GPU caches.

### 4. Explicit SM86 NVRTC target

`CompileOptions` now uses:

```rust
arch: Some("compute_86")
```

The fork also defaults `SEEDPHRASE_BLOCK` to 128 rather than 64. This is only a starting point; real hardware must test 64/128/256/512.

### 5. Aligned u32 secp256k1 G table

Only on `opt/sm86-rtx3070-v1`.

The original host table was byte-packed and every scalar-multiplication window loaded 64 individual bytes then rebuilt 16 u32 values.

The optimized branch preconverts the table on the host to `Vec<u32>`, uploads `CudaSlice<u32>`, and loads aligned words directly in CUDA.

The `opt/sm86-rtx3070-core` branch stops before this change and should be used to measure whether this optimization helps or hurts.

## Validation status

The code has been reviewed structurally and the repository contains CI intended to run:

- `cargo check`
- CUDA PTX compilation for `compute_86`

No claim is made that the optimized kernel has already been validated on an RTX 3070. The local GPU self-test is mandatory before any recovery run.

## Required first test sequence

Build:

```bash
git checkout opt/sm86-rtx3070-v1
cargo build --release
```

Run the existing correctness self-test by starting the program normally. Do not continue if any BIP84 vector fails.

Then benchmark:

```bash
chmod +x scripts/bench_sm86.sh
./scripts/bench_sm86.sh
```

Record at least three runs per block size if results are noisy.

Also benchmark:

```bash
git checkout main
cargo build --release
./scripts/bench_sm86.sh

git checkout opt/sm86-rtx3070-core
cargo build --release
./scripts/bench_sm86.sh

git checkout opt/sm86-rtx3070-v1
cargo build --release
./scripts/bench_sm86.sh
```

Use the same GPU power limit, clocks, thermals, driver, CUDA toolkit, and machine state.

## Nsight Compute

Profile the winning block size:

```bash
chmod +x scripts/profile_sm86.sh
SEEDPHRASE_BLOCK=128 ./scripts/profile_sm86.sh
```

The most important things to extract are:

- registers per thread
- achieved occupancy / active warps
- local-memory load/store traffic
- instruction throughput / issue stalls
- L1/L2 hit behavior for the global wordlist
- branch efficiency
- kernel duration

If the `--set full` metric set is unsupported on the installed Nsight version, use `--set detailed` and query available metrics with `ncu --query-metrics`.

## Decision rules

### Keep the rolling SHA/PBKDF2 work if

- all self-tests pass, and
- end-to-end candidates/s improves reproducibly.

If performance regresses, inspect local-memory and register count first. Explicit unrolling can increase register pressure even while removing `W[80]`.

### Keep the global wordlist path if

- occupancy/residency improves enough to compensate for extra global loads.

If L1/L2 behavior is poor, test a hybrid alternative rather than blindly restoring the 24 KiB shared copy. Possible follow-ups:

- constant/read-only placement if practical
- a smaller cache for only words used by known positions
- warp/cooperative staging of only candidate-needed entries

### Keep the u32 G table if

`opt/sm86-rtx3070-v1` beats `opt/sm86-rtx3070-core`.

If it does not, revert only the G-table commits.

## High-value follow-up experiments

1. Test block sizes 64/128/256/512 after every major kernel change.
2. Inspect PTX/SASS register allocation around `sha512_compress_words`.
3. If register pressure remains high, test a partially unrolled SHA-512 implementation rather than fully explicit 80 rounds.
4. Test a low-register secp256k1 inversion addition chain only after PBKDF2 is no longer the dominant issue.
5. Consider a specialized BIP32 HMAC path for fixed 37-byte messages, but only after profiling shows BIP32 matters.
6. Consider keeping the PBKDF2 hot loop in 32-bit halves only if SM86 64-bit integer throughput proves limiting; this is a major rewrite and should not be attempted without measurements.
7. If power efficiency matters, benchmark RTX 3070 at multiple power limits (for example 150/170/190/220 W) and report candidates/s/W.

## Result template

Please return:

```text
GPU:
Driver:
CUDA toolkit:
Power limit:
Core clock behavior:
Memory clock:
Temperature:

main
  block 64:
  block 128:
  block 256:
  block 512:

opt/sm86-rtx3070-core
  block 64:
  block 128:
  block 256:
  block 512:

opt/sm86-rtx3070-v1
  block 64:
  block 128:
  block 256:
  block 512:

Best branch/block:
Best candidates/s:
Candidates/s/W:

NCU:
  registers/thread:
  achieved occupancy:
  local load/store:
  L1 hit:
  L2 hit:
  dominant stall reasons:

Correctness self-test:
  PASS/FAIL

Notes:
```

## Important

Do not use a real wallet search until the built-in correctness self-test passes on the exact optimized binary being benchmarked.
