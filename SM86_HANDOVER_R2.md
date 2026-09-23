# RTX 3070 / SM86 review handover (R2)

Branch: `opt/sm86-rtx3070-r2`  
Base measured branch from local agent: `opt/sm86-rtx3070-lb` @ `25489bc`

## What is already established on the reference RTX 3070

Hardware measurements reported by the local agent and encoded in the launch-bounds commit:

- Original SM86 core branch compiled around 254/255 registers per thread.
- `__launch_bounds__(256,2)` reduced the kernel to 128 registers/thread and was the only reproducibly positive tuning change in that round.
- Three alternating comparisons on block 256:
  - launch-bounds: 321575 / 316431 / 317846 c/s, mean 318617 c/s
  - no launch-bounds: 304929 / 303074 / 306695 c/s, mean 304899 c/s
  - measured gain: about +4.5%
- Reported baseline main throughput: about 278436 c/s, so the measured launch-bounds build was about +14% over baseline.
- Reported block-size samples after launch-bounds:
  - block 64: 313789 c/s
  - block 128: 318425 c/s
  - block 256: approximately 318617 c/s mean in the alternating test
- `__launch_bounds__(256,3)` forced about 80 registers and regressed to about 303k c/s.
- block 512 is not merely slow: it cannot launch with the high-register kernel.
- The aligned u32 G-table experiment did not establish a reproducible benefit and is intentionally not present in this branch.
- Nsight Compute hardware counters were unavailable in the prior environment because of `ERR_NVGPUCTRPERM`.

## Static ptxas evidence added in R2

GitHub Actions now compiles the kernel for SM86 with CUDA 12.4 and prints ptxas resource usage.

Baseline inline implementation:

| launch bounds | kernel registers | kernel stack | hmac spills | PBKDF2 spills |
|---|---:|---:|---:|---:|
| (256,1) | 255 | 1344 B | 0 / 0 B | 0 / 0 B |
| (256,2) | 128 | 1952 B | 512 / 512 B | 856 / 872 B |
| (256,3) | 80 | 2736 B | 1808 / 1896 B | 2760 / 3108 B |
| (256,4) | 64 | 2912 B | 2256 / 2540 B | 3612 / 4372 B |

Spill columns are stores / loads reported by ptxas.

This strongly supports the measured result: occupancy gains beyond min-blocks=2 are overwhelmed by spill traffic. The next optimization target should be reducing PBKDF2/SHA live ranges at 128 registers, not forcing the register cap lower.


### Inline/noinline ptxas probes at (256,2)

| mode | kernel stack/spills | relevant function spills | static read |
|---|---|---|---|
| A default inline | 1952 B, kernel 0/0 | HMAC 512/512; PBKDF2 856/872 | measured reference |
| B SHA words noinline | 1936 B, kernel 1092/1224 | SHA function 0/0 | spill moved across call boundary; hardware A/B required |
| C fixed64 HMAC noinline | 2416 B, kernel 0/0 | PBKDF2 1324/1360; fixed64 HMAC 0/0 | statically unattractive; lowest priority |
| D both noinline | 1968 B, kernel 1044/1188 | SHA + fixed64 HMAC 0/0 | spill moved across call boundary; hardware A/B required |

Spill pairs are stores/loads. These are allocation figures, not dynamic memory-transaction counts, so B/D cannot be accepted or rejected from ptxas alone. C, however, increases the PBKDF2 spill allocation substantially without reducing the 128-register cap and should only be tested after B/D, if at all.

## R2 code changes

R2 keeps the measured production default `__launch_bounds__(256,2)` and block 256, then adds only diagnostics / experiment infrastructure by default:

1. Removed the unused custom `int64_t` typedef that conflicts with host-side nvcc headers.
2. Invalid `SEEDPHRASE_BLOCK` values now fail loudly. Values must be warp multiples in 32..=256. There is no silent 512 -> 256 fallback.
3. Benchmark GPU-init failures now exit non-zero.
4. Benchmark prints both:
   - `weighted_all`
   - `weighted_steady` using chunks 1..4, excluding the cold first chunk
5. Benchmark prints the active experiment configuration.
6. Added `SEEDPHRASE_LB_MIN_BLOCKS=1..4` as a controlled NVRTC experiment knob. Default remains 2.
7. Added two opt-in inlining experiments; defaults preserve the measured code path:
   - `SEEDPHRASE_NOINLINE_SHA512_WORDS=1`
   - `SEEDPHRASE_NOINLINE_FIXED64_HMAC=1`
8. Added GitHub Actions ptxas resource checks.
9. Added `--self-test` as a non-interactive correctness gate for automation.
10. `--bench` and `--self-test` now print CUDA driver resource attributes directly: registers/thread, local bytes/thread, shared bytes/block, and max threads/block. No external probe script is required for these four metrics.
11. Added an opt-in `SEEDPHRASE_NOINLINE_PBKDF2=1` boundary probe, default OFF, to test whether SHA noinline can be combined with an explicit PBKDF2 call boundary.

## Important repository discrepancy

The previous local-agent report said that `SM86_RESULTS.md`, `results/`, `patch_launchbounds.py`, `patch_probe2.py`, and `recovery_smoke.py` were pushed. They are not present in the Git tree at `25489bc` or on `opt/sm86-rtx3070-lb`.

Do not upload the full multi-gigabyte scratch directory. Commit only the compact report, scripts, and selected raw text/CSV logs required to reproduce conclusions.

## Next experiment priority

The next round should test whether function-call boundaries reduce spills enough to outweigh CUDA device-call overhead.

At fixed block 256 and min-blocks=2, compare in this priority order:

- A: current/default inline path
- B: `SEEDPHRASE_NOINLINE_SHA512_WORDS=1`
- D: both flags = 1
- C: `SEEDPHRASE_NOINLINE_FIXED64_HMAC=1` only if time permits; static ptxas makes it the least promising

Use `weighted_steady` as the primary throughput metric.

Do not promote a variant on a single run. Use alternating order and at least three complete rounds. The target is a reproducible improvement above normal run-to-run noise; report all raw runs.

## Correctness gates

For every candidate that may be kept:

1. Build release.
2. Run the non-interactive gate with the exact experiment environment:
   ```bash
   ./target/release/seedphrase_recovery --self-test
   ```
   It must exit 0 and report 3/3 BIP84 vectors.
3. Public/redacted end-to-end recovery smoke test must pass.
4. `--bench` must return exit code 0.
5. No invalid block-size fallback is allowed.
6. If a candidate wins, repeat the default branch immediately before and after it (ABBA or equivalent) to control clock/thermal drift.

## If none of the noinline probes win

Do not lower launch-bounds below 128 registers again.

Next investigate partial SHA-512 unrolling / live-range reduction while keeping the 16-word rolling schedule. The goal is specifically:

- retain 128-register / two-block residency
- reduce PBKDF2 ptxas spill traffic below 856-store / 872-load bytes
- avoid moving the hot schedule back into dynamic local-memory arrays

A successful code change should be evaluated by both ptxas spill reduction and real end-to-end c/s. Lower spills alone are not sufficient.

## What to return

Commit a compact `SM86_RESULTS_R2.md` plus selected logs containing:

- exact commit SHA
- GPU / driver / CUDA toolkit
- power limit and observed clocks/temperature
- each experiment's environment variables
- self-test status
- recovery smoke status
- every `weighted_steady` result
- mean / median and relative change vs A
- the built-in `Kernel resources:` line (cuFuncGetAttribute via cudarc)
- ptxas registers / stack / spills
- any contaminated run explicitly marked and excluded
- final keep/reject decision for each experiment

Do not delete useful remote scratch data until review is complete, but do not commit gigabytes of profiler output.
