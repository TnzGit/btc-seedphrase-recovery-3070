# NEXT LOCAL AGENT PROMPT — standalone SM86 SHA-512 microbenchmark

You are the **local execution agent**. This is a standalone GPU microbenchmark only.

## Safety / scope boundary

This branch is intentionally isolated from the wallet-recovery pipeline.

Do **not**:
- modify or call the recovery enumeration kernel,
- enumerate BIP39 words,
- derive BIP32 keys,
- generate or match wallet addresses,
- copy any experimental SHA-512 implementation back into the recovery pipeline.

Only benchmark the standalone binary and synthetic SHA-512 workload in this branch.

## Repository / branch

- repo: `TnzGit/btc-seedphrase-recovery-3070`
- branch: `research/sm86-sha512-microbench`
- static screen baseline includes `9a8445daa49402b152cab559ef5c9a7e49b90806`

Start with:

```bash
git fetch origin
git checkout research/sm86-sha512-microbench
git pull --ff-only
git merge-base --is-ancestor 9a8445daa49402b152cab559ef5c9a7e49b90806 HEAD
cargo build --locked --release --bin sha512_sm86_microbench
```

Use the existing NVRTC 12.9 environment:

```bash
export NVRTC_DIR=/home/user/nvrtc/usr/local/cuda-12.9/targets/x86_64-linux/lib
export LD_LIBRARY_PATH="$NVRTC_DIR:/usr/lib/wsl/lib"
```

Verify `nvrtcVersion=12.9` using the existing `scripts/nvrtc_ver.py` if available.

## Static result already established

CUDA 12.9 / sm_86 CI found:

| kernel | regs/thread | stack | spill | SASS lines |
|---|---:|---:|---:|---:|
| u64 | 210 | 0 | 0/0 | 7,268 |
| pair32 | 96 | 0 | 0/0 | 8,148 |

Selected final-SASS counts:

- u64: IADD3 family 761, SHF family 1,728, LOP3 900
- pair32: IADD3 family 1,067, SHF family 1,668, LOP3 894

So pair32 trades more arithmetic instructions for much lower register pressure. Real RTX 3070 timing decides whether the occupancy gain wins.

## Mandatory correctness gate

Run:

```bash
./target/release/sha512_sm86_microbench --self-test
```

Require:
- exit 0
- `u64 self-test: PASS (SHA-512("abc"))`
- `pair32 self-test: PASS (SHA-512("abc"))`
- capture both runtime resource lines

If either implementation fails, stop benchmarking and report the failure. Do not patch the kernel locally.

## Benchmark calibration

The binary accepts:

```text
--bench <u64|pair32|both>
--threads N
--block N
--iters N
--samples N
```

Start with:

```bash
./target/release/sha512_sm86_microbench --bench u64   --threads 262144 --block 128 --iters 512 --samples 3
```

Adjust **only `--iters`** until each timed sample is roughly 2–6 seconds on the RTX 3070.

Once calibrated, freeze:
- threads
- iterations
- samples

and use the exact same workload for every candidate.

Do not include NVRTC compile time in throughput measurements; the binary times only synchronized kernel execution.

## Phase 1 — block-size screen

Screen both implementations at:

```text
block = 64, 96, 128, 224, 256
```

For each `variant × block`:
- run at least 3 samples
- use `weighted_steady` (samples after #0) as primary
- retain all raw output
- capture 1 Hz GPU telemetry
- summarize telemetry only for rows with GPU utilization >=99%
- record power, SM clock, temperature, and kernel resource lines

Use balanced ordering to reduce drift, for example:

```text
u64-64, pair-64, u64-96, pair-96, ...
... then reverse the order for a second pass if screening noise is >1%
```

Pick the fastest block size independently for:
- u64
- pair32

Do not assume both kernels want the same block size.

## Phase 2 — strict implementation comparison

Compare the fastest correct u64 configuration with the fastest correct pair32 configuration.

Run three complete alternating rounds:

```text
U P P U
U P P U
U P P U
```

where U and P may use different block sizes selected in Phase 1, but must use the same:
- thread count,
- iterations/thread,
- sample count,
- NVRTC/toolchain,
- GPU power limit.

For every run record:
- `weighted_steady`
- `weighted_all`
- individual sample timings
- runtime resources
- kernel-phase telemetry (util >=99%)
- exact command line
- exit code

No run may be silently discarded. If thermal or other regime changes occur, mark them explicitly and compare within the same regime.

## Decision rule

This is a research microbenchmark, not a production promotion.

Classify pair32 as:

- **strong win**: >=5% mean and median improvement, consistent across rounds
- **moderate win**: 2–5% consistent improvement
- **flat**: within ±2%
- **loss**: <=-2%

If pair32 wins, the next remote-agent step will remain inside this standalone microbenchmark and investigate why (occupancy / instruction scheduling / block geometry). Do not integrate it elsewhere.

## Deliverables

Create and commit:

- `research/sha512_sm86/HW_RESULTS.md`
- `research/sha512_sm86/results/logs/`
- `research/sha512_sm86/results/telemetry/`
- a compact helper script if useful

The report must include:

1. exact branch + commit
2. GPU/driver/NVRTC/Rust environment
3. both self-test results
4. runtime resources for both kernels
5. calibration workload
6. every Phase-1 `weighted_steady`
7. fastest block size for each implementation
8. all Phase-2 alternating data
9. mean/median/min/max and paired deltas
10. kernel-phase telemetry summaries
11. contaminated/excluded runs, if any
12. final pair32 classification: strong win / moderate win / flat / loss
13. exact result commit SHA
14. any anomaly/blocker

Push the branch.

End your reply with:
- fastest correct implementation
- fastest block size for each implementation
- exact pair32 delta vs u64
- final classification
- runtime regs/local for both
- result commit SHA
- report/log paths
- any anomaly/blocker
