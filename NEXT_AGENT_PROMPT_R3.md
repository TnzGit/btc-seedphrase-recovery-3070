# NEXT LOCAL AGENT PROMPT — RTX 3070 / SM86 R3 hardware geometry

You are the **local execution agent**. The remote agent owns planning and code changes; do not modify CUDA/Rust hot-path code in this round. Your job is to run the prepared experiments on the reference RTX 3070, preserve raw evidence, and report results.

## Repository / branch

- repo: `TnzGit/btc-seedphrase-recovery-3070`
- branch: `opt/sm86-rtx3070-r3-hw`
- source-code baseline includes `dae5aaa295d4a5b123274c664d984f9bc78fb454`
- do **not** modify `main`
- do **not** rewrite `opt/sm86-rtx3070-r2`
- R2 remains the production/reference behavior

Start with:

```bash
git fetch origin
git checkout opt/sm86-rtx3070-r3-hw
git pull --ff-only
cargo build --locked --release
```

Confirm the checked-out history contains `dae5aaa`. This commit fixes the R3 runtime validator so all warp-multiple launch geometries in 64..=256 can actually be tested through `SEEDPHRASE_LB_MAX_THREADS`.

## Why this round exists

R2 established the correct/default reference:

- launch bounds: `(256,2)`
- block: 256
- 128 regs/thread
- kernel spill: 0/0
- HMAC spill: 512/512 B
- PBKDF2 spill: 856/872 B
- all noinline/boundary probes were rejected on hardware

R3 static ptxas screening found a new register point:

### G128 — launch bounds (128,3)

- 168 regs/thread
- kernel spill: 0/0
- HMAC spill: 340/340 B
- PBKDF2 spill: 456/456 B
- block 128 -> 3 resident blocks / 384 resident threads if resource limits behave as expected

### G192 — launch bounds (192,2)

Static allocation is the same as G128:

- 168 regs/thread
- kernel spill: 0/0
- HMAC spill: 340/340 B
- PBKDF2 spill: 456/456 B
- block 192 -> 2 resident blocks / 384 resident threads if resource limits behave as expected

Compared with A, these geometries substantially reduce static spill but also reduce resident warps from the A target of 16 warps (2 x 256) to 12 warps. Only real RTX 3070 measurements can decide which effect wins.

Other static geometry points (96x5, 160x3, 224x2) stayed at 128 regs with the old 856/872 PBKDF2 spill, so they are low priority.

Shared-memory probes at the 168-reg geometry are also secondary:
- `T_SHARED=1`: 168 regs, 8 KiB smem, kernel spill 432/432 B, HMAC 340/340 B
- half-shared SHA schedule: same headline resources
They only slightly change aggregate static spill while adding shared-memory traffic, so test them only after plain G128/G192 are understood.

Do not spend time on:
- scalar U/T probe: static allocation was unchanged vs A
- shared HMAC-state matrix at 128 regs: large kernel spill and 16–49 KiB shared memory
- rejected noinline probes from R2

## Mandatory correctness gate

For every configuration that will be benchmarked:

1. Run the exact same environment with:
   ```bash
   ./target/release/seedphrase_recovery --self-test
   ```
2. Require exit code 0 and all 3 BIP84 GPU vectors PASS.
3. Capture the full `Kernel resources:` line.
4. For any finalist, run the existing public/redacted recovery smoke test from `results/r1-recovered/recovery_smoke.py` (or the equivalent existing script) before promotion.

Do not benchmark/promote a configuration that fails correctness.

The known incorrect R2 noinline combination is already hard-blocked in Rust/CUDA; do not bypass the guard.

## Experiment flags

Unless a configuration explicitly says otherwise, force all experiment flags OFF:

```bash
export SEEDPHRASE_NOINLINE_SHA512_WORDS=0
export SEEDPHRASE_NOINLINE_FIXED64_HMAC=0
export SEEDPHRASE_NOINLINE_PBKDF2=0
export SEEDPHRASE_PBKDF2_SCALAR_UT=0
export SEEDPHRASE_PBKDF2_T_SHARED=0
export SEEDPHRASE_PBKDF2_STATE_SHARED=0
export SEEDPHRASE_SHA512_HALF_SHARED_SCHEDULE=0
```

### A — R2 reference

```bash
export SEEDPHRASE_LB_MAX_THREADS=256
export SEEDPHRASE_LB_MIN_BLOCKS=2
export SEEDPHRASE_BLOCK=256
```

### G128 — 168-reg geometry

```bash
export SEEDPHRASE_LB_MAX_THREADS=128
export SEEDPHRASE_LB_MIN_BLOCKS=3
export SEEDPHRASE_BLOCK=128
```

### G192 — 168-reg geometry

```bash
export SEEDPHRASE_LB_MAX_THREADS=192
export SEEDPHRASE_LB_MIN_BLOCKS=2
export SEEDPHRASE_BLOCK=192
```

First sanity-check that G192 is accepted by the runtime. If `LB_MAX_THREADS=192` is rejected, the checkout is stale or missing `dae5aaa`; do not work around it by patching locally.

## Phase 1 — hardware screening

Primary candidates: A, G128, G192.

- self-test all three first
- run at least two `--bench` measurements per candidate
- use `weighted_steady` as the primary number
- record `weighted_all` only as a cold-start diagnostic
- sample GPU utilization, power, SM clock, memory clock and temperature at 1 Hz
- use balanced/interleaved order rather than A-A-G-G sequences

Suggested screening order:

```text
A G128 G192
G192 G128 A
```

If GPU state changes materially during the sequence, mark the boundary and do not average across regimes.

Early reject a candidate if it is clearly >5% slower than A in comparable same-regime runs.

A candidate within roughly 2% of A should remain alive for rigorous testing because the screening noise observed in R2 was several percent.

## Phase 1b — optional shared probes

Only if plain G128/G192 are not catastrophically slower, screen these two at G128 geometry:

### G128-T

```bash
SEEDPHRASE_LB_MAX_THREADS=128
SEEDPHRASE_LB_MIN_BLOCKS=3
SEEDPHRASE_BLOCK=128
SEEDPHRASE_PBKDF2_T_SHARED=1
```

### G128-HS

```bash
SEEDPHRASE_LB_MAX_THREADS=128
SEEDPHRASE_LB_MIN_BLOCKS=3
SEEDPHRASE_BLOCK=128
SEEDPHRASE_SHA512_HALF_SHARED_SCHEDULE=1
```

All other flags remain 0.

Run self-test before benchmark. Two screening runs each are enough unless either is competitive.

Do not test shared HMAC state or combine shared probes in this round.

## Phase 2 — rigorous finalists

For every candidate that survives Phase 1, compare against A using three complete alternating rounds:

```text
A X X A
A X X A
A X X A
```

That yields six A and six X measurements.

For every run, save:
- full benchmark output
- `weighted_steady`
- `weighted_all`
- `Kernel resources:`
- 1 Hz GPU telemetry
- exit status
- exact environment variables

Promotion requires all of:
- all correctness gates pass
- recovery smoke passes
- consistent win across alternating rounds
- mean and median `weighted_steady` improvement approximately >=2% vs the paired A measurements
- no credible power/temperature/clock explanation for the apparent gain

Do not call a one-run spike a win.

If contamination occurs, retain the raw run, label it, explain why it is excluded, and do not silently replace/delete it.

## Power-limit sweep

Do **not** vary power limits while selecting code/geometry.

Only if a correct geometry/probe becomes a reproducible winner should you optionally run the final selected path at several power limits (for example 160/180/200/220/240 W) and report c/s/W separately.

## Deliverables

Do not change CUDA/Rust implementation code. You may add benchmark scripts, compact logs, telemetry and reports.

Create:

- `SM86_RESULTS_R3.md`
- `results/r3/logs/`
- `results/r3/telemetry/`
- `results/r3/ptxas/` only for compact static references actually used

The report must include:

1. exact branch and commit SHA
2. GPU, driver, NVRTC/CUDA environment
3. exact env vars for every candidate
4. self-test results
5. recovery smoke result for finalists
6. every raw `weighted_steady`
7. mean / median / min / max per candidate
8. paired delta vs A
9. runtime `Kernel resources:`
10. power / clock / temperature summary
11. all contaminated/excluded runs and reasons
12. keep/reject decision for every candidate
13. whether G128 or G192 established a real hardware win
14. whether any shared probe justified further work
15. any remaining blocker

Commit only compact evidence; do not commit multi-GB scratch/profiler data and never commit real wallet seed material, private keys, passwords, or other secrets.

Push the result branch and finish your reply with:

- fastest correct candidate
- exact paired delta vs A
- whether the >=2% promotion bar was met
- exact commit SHA
- report/log paths
- any anomaly or blocker that the remote agent should handle next
