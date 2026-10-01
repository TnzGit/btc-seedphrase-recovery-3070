# NEXT LOCAL AGENT PROMPT — RTX 3070 / SM86 R5 scalar-U/T hardware screen

You are the **local execution agent**. The remote agent owns planning and code changes. Do not modify CUDA/Rust hot-path code in this round. This is a deliberately small hardware screen of the last low-cost existing live-range variant that has static evidence but no real RTX 3070 throughput result.

## Repository / branch

- repo: `TnzGit/btc-seedphrase-recovery-3070`
- branch: `opt/sm86-rtx3070-r5-scalar-hw`
- base includes R4 result `a8d6482a21aef5751a04788535b8be96d876fc23`
- base also includes remote-review telemetry correction / harness fix through `6c9fb8e55f02e5841f15cd7410d9865c99edc085`
- do not modify `main`
- do not rewrite R2/R3/R4 result branches

Start with:

```bash
git fetch origin
git checkout opt/sm86-rtx3070-r5-scalar-hw
git pull --ff-only
git merge-base --is-ancestor 6c9fb8e55f02e5841f15cd7410d9865c99edc085 HEAD
cargo build --locked --release
```

## What the remote review has already closed

Do **not** spend hardware time on any of these:

- G192: rejected on hardware (~-5.2%).
- T_SHARED / half-shared: rejected on hardware.
- shared HMAC state: remote static screen shows only a tiny aggregate spill reduction while adding 16–24 KiB shared memory/block; reject without hardware.
- noinline/boundary variants: remote CUDA 12.9 ptxas screen rejects them; HMAC noinline makes G128 PBKDF2 spill much worse (~1228/1252 B).
- NVRTC 12.4 pinning: R4 shows N124 +0.542% mean vs N129 on A, below the 2% bar; do not pin.
- explicit fixed64 in-place HMAC: remote R5 experiment compiled ref/inplace to **byte-identical final SASS** at G128 (`SASS_IDENTICAL=1`); do not benchmark it.

The R4 remote review also corrected the telemetry interpretation: whole-run averages were contaminated by NVRTC compile time. For all R5 summaries use **kernel-phase telemetry only**, defined as samples with GPU utilization >=99%. The corrected `scripts/bench_sm86_r4.sh` already uses this rule.

## Why scalar-U/T is still worth one small hardware screen

`SEEDPHRASE_PBKDF2_SCALAR_UT=1` keeps PBKDF2 U/T as scalar u64 values and uses the existing scalar in-place HMAC macro.

Static ptxas at both A and G128 showed the same headline resource allocation as the corresponding reference:
- A: 128 regs, PBKDF2 spill 856/872 B
- G128: 168 regs, PBKDF2 spill 456/456 B

So this is **not** a spill-allocation bet. The only plausible win is different instruction scheduling / alias handling in the final generated code. If hardware does not show a clear signal quickly, reject it.

## Runtime / compiler environment

Use the existing default R3/R4 N129 environment only:

```text
NVRTC 12.9.86
/home/user/nvrtc/usr/local/cuda-12.9/targets/x86_64-linux/lib
```

Do not mix N124 into R5. Verify `nvrtcVersion=12.9` in logs.

All unrelated flags must be OFF:

```bash
SEEDPHRASE_NOINLINE_SHA512_WORDS=0
SEEDPHRASE_NOINLINE_FIXED64_HMAC=0
SEEDPHRASE_NOINLINE_PBKDF2=0
SEEDPHRASE_PBKDF2_T_SHARED=0
SEEDPHRASE_PBKDF2_STATE_SHARED=0
SEEDPHRASE_SHA512_HALF_SHARED_SCHEDULE=0
```

## Configurations

### A — production/reference geometry

```bash
SEEDPHRASE_LB_MAX_THREADS=256
SEEDPHRASE_LB_MIN_BLOCKS=2
SEEDPHRASE_BLOCK=256
SEEDPHRASE_PBKDF2_SCALAR_UT=0
```

### A-S — scalar-U/T at A geometry

Same as A except:

```bash
SEEDPHRASE_PBKDF2_SCALAR_UT=1
```

### G128 — known 168-reg geometry reference

```bash
SEEDPHRASE_LB_MAX_THREADS=128
SEEDPHRASE_LB_MIN_BLOCKS=3
SEEDPHRASE_BLOCK=128
SEEDPHRASE_PBKDF2_SCALAR_UT=0
```

### G128-S — scalar-U/T at G128

Same as G128 except:

```bash
SEEDPHRASE_PBKDF2_SCALAR_UT=1
```

## Correctness gate

Run `--self-test` for A, A-S, G128 and G128-S with their exact environments.

Require:
- exit 0
- 3/3 BIP84 PASS
- capture `Kernel resources:`
- confirm resource attributes for scalar vs reference are not unexpectedly different

Do not benchmark any failing configuration.

## Benchmark harness

Reuse the corrected R4 harness:

```text
scripts/bench_sm86_r4.sh
```

It accepts extra environment assignments after the fixed arguments, so pass `SEEDPHRASE_PBKDF2_SCALAR_UT=0|1` explicitly.

Important: its current summary semantics after remote review are:
- raw CSV remains complete
- `avg_power_W`, `avg_sm_clk`, `max_temp_C` summarize only rows where GPU util >=99%
- `kernel_rows` must be present and non-zero

If you need a small R5 wrapper for convenience, you may create/commit benchmark scripts and reports, but do not change CUDA/Rust implementation code.

## Phase 1 — small balanced screening

Run two balanced passes over all four configs. Suggested order:

```text
A  A-S  G128  G128-S
G128-S  G128  A-S  A
```

For every run save:
- `weighted_steady` primary
- `weighted_all` secondary
- runtime resources
- kernel-phase telemetry
- exit status
- exact environment
- NVRTC version

Interpret scalar effects within the **same geometry** first:
- A-S vs A
- G128-S vs G128

Also compare G128-S directly to A because promotion ultimately requires beating production A.

### Early-stop rule

Do not spend a full ABBA campaign on scalar-U/T unless there is a real screening signal.

Normally reject scalar-U/T after Phase 1 if:
- both same-geometry comparisons are flat/negative, or
- G128-S is <~1% above A and there is no consistent >=0.5% same-geometry scalar gain.

A one-run spike is not a signal.

If A-S unexpectedly shows a strong gain, keep it as a finalist independently of G128-S.

## Phase 2 — only for a surviving scalar finalist

For any scalar candidate that survives screening, compare it directly against A using:

```text
A X X A
A X X A
A X X A
```

Promotion still requires:
- all correctness gates pass
- recovery smoke passes
- all/most paired rounds agree in direction
- mean and median `weighted_steady` improvement approximately >=2% vs paired A
- no thermal/power/clock explanation

If the result is below 2%, reject it even if direction is consistently positive. We have enough sub-1% findings already.

## Recovery smoke

Run the existing public/redacted recovery smoke for every Phase-2 finalist before any promotion.

Never commit real wallet seed material, private keys, passwords or secrets.

## Deliverables

Create and commit:

- `SM86_RESULTS_R5.md`
- `results/r5/logs/`
- `results/r5/telemetry/`
- any compact helper script actually used

The report must include:

1. exact branch + commit
2. NVRTC version/path
3. correctness results for all four configs
4. runtime resource lines
5. all Phase-1 raw `weighted_steady`
6. same-geometry deltas A-S vs A and G128-S vs G128
7. G128-S vs production A
8. Phase-2 ABBA data only if a finalist survives
9. kernel-phase telemetry only (util >=99%), plus raw CSV retained
10. any excluded/contaminated runs
11. final keep/reject decision
12. explicit statement whether scalar-U/T is now closed as an optimization direction

Push the result branch.

End your reply with:
- fastest correct config
- scalar-U/T delta at A geometry
- scalar-U/T delta at G128 geometry
- best candidate delta vs production A
- whether the >=2% promotion bar was met
- exact result commit SHA
- report/log paths
- any anomaly/blocker

If scalar-U/T does not meet the bar, state clearly that the low-cost live-range family is exhausted; the remote agent will decide whether to proceed to a larger SHA-512 instruction-level redesign.
