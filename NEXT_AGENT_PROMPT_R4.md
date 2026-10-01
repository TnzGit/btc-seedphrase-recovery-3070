# NEXT LOCAL AGENT PROMPT — RTX 3070 / SM86 R4 NVRTC isolation

You are the **local execution agent**. The remote agent owns planning and code changes. In this round, do not modify CUDA/Rust hot-path code. Your job is to isolate the runtime NVRTC/toolkit effect that differs between R2 and R3.

## Repository / branch

- repo: `TnzGit/btc-seedphrase-recovery-3070`
- branch: `opt/sm86-rtx3070-r4-env`
- base result commit: `4ef1905a485a7f00eac2b1eda4c4164757c1b09a`
- do not modify `main`
- do not rewrite R2/R3 result branches

Start with:

```bash
git fetch origin
git checkout opt/sm86-rtx3070-r4-env
git pull --ff-only
git merge-base --is-ancestor 4ef1905a485a7f00eac2b1eda4c4164757c1b09a HEAD
cargo build --locked --release
```

Use one built binary for the whole comparison unless a dependency-loading issue forces otherwise.

## Why this round exists

R3 is internally valid: A and G128 were compared under the same NVRTC 12.9.86 environment, and G128 was only +0.326% mean, below the 2% promotion threshold.

However:
- R2 recorded NVRTC 12.4.127.
- R3 used NVRTC 12.9.86 from `/home/user/nvrtc/usr/local/cuda-12.9/targets/x86_64-linux/lib`.
- Absolute A throughput between rounds differs by several percent, but the old rounds also had GPU regime changes.

Before doing more code-level tuning, isolate whether NVRTC/toolkit version materially changes the generated kernel on this exact RTX 3070.

The remote static review also found that existing noinline/boundary probes do **not** become attractive at G128:
- G128 baseline on CUDA 12.9 ptxas: 168 regs, HMAC 340/340 B, PBKDF2 456/456 B.
- SHA noinline creates ~608/612 B kernel spill.
- HMAC noinline on CUDA 12.9 makes PBKDF2 spill ~1228/1252 B.
- PBKDF2-only noinline is effectively unchanged.
- scalar-U/T is effectively unchanged.
- shared-state only slightly reduces aggregate static spill while adding 16 KiB shared memory/block; do not benchmark it in this round.

So this round is strictly compiler/runtime isolation, not another probe sweep.

## Toolchain targets

Compare these exact NVRTC major/minor paths if available:

### N129
The known R3 environment:
- NVRTC reports 12.9.86
- current known library directory:
  `/home/user/nvrtc/usr/local/cuda-12.9/targets/x86_64-linux/lib`

### N124
Recover/reuse the exact R2-style NVRTC 12.4.x environment if it still exists.

Search the retained R2/R1 scratch locations and home directories first. The R2 report recorded NVRTC 12.4.127 as an unpacked toolkit, not a required system-wide install.

If the old files are gone, it is acceptable to download/extract the same CUDA 12.4 NVRTC userspace packages into a private scratch directory. Do **not** replace the system driver, do not install a different NVIDIA kernel driver, and do not mutate the machine-wide CUDA setup just for this experiment.

Before any benchmark, verify the active NVRTC version. A small Python ctypes check is acceptable, e.g. load the intended `libnvrtc.so` and call `nvrtcVersion`, or use an existing tool that reports the exact version. Save the command/output to the report.

Make sure `libnvrtc.so` and its matching `libnvrtc-builtins.so` come from the same toolkit directory. Include `/usr/lib/wsl/lib` as needed for the CUDA driver library, but do not accidentally mix NVRTC builtins from another toolkit.

## Fixed code / flags

All code is fixed at the R3 result source. Do not patch kernel.cu.

All experimental flags OFF:

```bash
export SEEDPHRASE_NOINLINE_SHA512_WORDS=0
export SEEDPHRASE_NOINLINE_FIXED64_HMAC=0
export SEEDPHRASE_NOINLINE_PBKDF2=0
export SEEDPHRASE_PBKDF2_SCALAR_UT=0
export SEEDPHRASE_PBKDF2_T_SHARED=0
export SEEDPHRASE_PBKDF2_STATE_SHARED=0
export SEEDPHRASE_SHA512_HALF_SHARED_SCHEDULE=0
```

### A geometry

```bash
export SEEDPHRASE_LB_MAX_THREADS=256
export SEEDPHRASE_LB_MIN_BLOCKS=2
export SEEDPHRASE_BLOCK=256
```

### G128 geometry

```bash
export SEEDPHRASE_LB_MAX_THREADS=128
export SEEDPHRASE_LB_MIN_BLOCKS=3
export SEEDPHRASE_BLOCK=128
```

## Mandatory correctness gate

For each of N124+A, N129+A, N124+G128, N129+G128:

1. Verify the active NVRTC version first.
2. Run:
   ```bash
   ./target/release/seedphrase_recovery --self-test
   ```
3. Require exit 0 and 3/3 BIP84 PASS.
4. Save the `Kernel resources:` line.

If 12.4 and 12.9 generate different runtime resource attributes, that is a major finding; report it before performance interpretation.

Do not benchmark a configuration that fails correctness.

## Phase 1 — sanity screen

Run two measurements each, interleaved so compiler versions and geometry do not correlate with thermal drift.

Suggested order:

```text
N124-A
N129-A
N124-G128
N129-G128
N129-G128
N124-G128
N129-A
N124-A
```

For every run:
- verify/log the NVRTC version used by that process
- run `--bench`
- record `weighted_steady` (primary)
- record `weighted_all` (secondary)
- record `Kernel resources:`
- capture 1 Hz util/power/SM clock/mem clock/temp telemetry
- save exit status and exact `LD_LIBRARY_PATH` / relevant runtime library variables

If a GPU regime transition occurs, mark the boundary and do not average across regimes.

## Phase 2A — strict NVRTC comparison on A

Unless Phase 1 finds a correctness problem, compare N124-A vs N129-A with three complete alternating rounds:

```text
N124-A  N129-A  N129-A  N124-A
N124-A  N129-A  N129-A  N124-A
N124-A  N129-A  N129-A  N124-A
```

This is the primary result of R4.

Report paired per-round deltas, mean/median/min/max, telemetry, and runtime resource attributes.

Interpretation:
- >=2% reproducible same-regime win: meaningful toolchain result worth pinning/recommending.
- <2%: no promotion; consider the NVRTC difference operational noise / low-value unless resource attributes change materially.
- inconsistent direction: no winner.

## Phase 2B — geometry interaction

Only if Phase 1 shows G128 remains within about 2% of A under both compilers, run a smaller controlled check to determine whether the G128 +0.3% direction depends on NVRTC version.

For each NVRTC version, do at least two complete A/G128 alternating rounds:

```text
A G128 G128 A
A G128 G128 A
```

Do not combine 12.4 and 12.9 results into one average. Treat compiler version as a separate regime.

This phase is secondary; do not extend it if G128 clearly loses under a compiler.

## CUDA cache / cold-start handling

The primary metric remains `weighted_steady`, so NVRTC compile/JIT startup time is not part of the throughput result.

Do not delete caches between every run unless needed to diagnose a problem. If you change cache behavior, apply it identically to both versions and report it.

## Recovery smoke

For any environment/toolchain that would be recommended or promoted, run the existing public/redacted recovery smoke test before final recommendation. Never use or commit real wallet seed material, private keys, passwords, or secrets.

## Deliverables

Create and commit:

- `SM86_RESULTS_R4.md`
- `results/r4/logs/`
- `results/r4/telemetry/`
- a compact text file recording exact NVRTC library paths, versions, and environment for N124/N129
- any small benchmark helper script you create

Do not commit toolkit binaries, CUDA packages, caches, or large scratch trees.

The report must include:

1. exact branch + commit SHA
2. exact NVRTC versions and library paths
3. proof that matching libnvrtc/libnvrtc-builtins were used
4. GPU/driver/Rust environment
5. every self-test result
6. every runtime `Kernel resources:` line
7. every raw `weighted_steady`
8. mean/median/min/max and paired deltas
9. telemetry summaries
10. regime boundaries or contaminated runs
11. whether NVRTC 12.4 or 12.9 is >=2% faster on A
12. whether compiler version changes the G128-vs-A direction
13. recovery smoke for any promoted environment
14. final keep/reject recommendation for N124/N129
15. anything the remote agent should change in setup/README next

Push the result branch.

End your reply with:
- fastest correct NVRTC environment on A
- exact paired delta vs the other NVRTC
- whether the >=2% promotion threshold was met
- whether runtime resources differ
- G128-vs-A result under each compiler (if Phase 2B ran)
- exact result commit SHA
- report/log paths
- any blocker/anomaly
