# NEXT LOCAL AGENT PROMPT — RTX 3070 / SM86 R2

Continue the RTX 3070 optimization work in:

- repo: `TnzGit/btc-seedphrase-recovery-3070`
- branch to start from: `opt/sm86-rtx3070-r2`
- do NOT modify `main`
- read `SM86_HANDOVER_R2.md` before changing code

The previous hardware round established that `__launch_bounds__(256,2)` + block 256 is the current measured reference. Do not remove that reference path while experimenting.

## First: recover the missing prior deliverables

The previous report said these were pushed, but they are not present in Git at `opt/sm86-rtx3070-lb` / `25489bc`:

- `SM86_RESULTS.md`
- `results/`
- `patch_launchbounds.py`
- `patch_probe2.py`
- `recovery_smoke.py`

The remote scratch directory was reported as `/tmp/btc3070bench-j9PBaw`.

If it still exists, recover only the compact useful artifacts: report, scripts, selected text/CSV logs, and proof needed for the measured conclusions. Do NOT commit the full multi-GB scratch directory. Never commit real wallet seed material or secrets.

## Build and correctness gate

```bash
git fetch origin
git checkout opt/sm86-rtx3070-r2
git pull --ff-only
cargo build --release
```

For every experiment, use the exact same environment for self-test and benchmark.

The branch now has a non-interactive gate:

```bash
./target/release/seedphrase_recovery --self-test
```

It must:
- exit 0
- report PASS for all 3 BIP84 GPU vectors
- print a `Kernel resources:` line

Also run the existing public/redacted end-to-end recovery smoke test for any candidate that might be kept.

Do not benchmark or promote a candidate that fails correctness.

## Reference configuration A

Use:

```bash
export SEEDPHRASE_BLOCK=256
export SEEDPHRASE_LB_MIN_BLOCKS=2
export SEEDPHRASE_NOINLINE_SHA512_WORDS=0
export SEEDPHRASE_NOINLINE_FIXED64_HMAC=0
export SEEDPHRASE_NOINLINE_PBKDF2=0
```

This is the current measured reference path.

Previous hardware result to sanity-check, not to blindly force:
- around 318.6k c/s mean in the prior alternating test
- about +4.5% vs the same core code without launch bounds
- about +14% vs the reported original-main baseline

Use the new `weighted_steady` number from chunks 1..4 as the primary throughput metric. Keep `weighted_all` only as a secondary/cold-start diagnostic.

## Screening candidates

At fixed block=256 and lb_min_blocks=2, screen these configurations:

A — current inline reference:
```bash
SHA=0 HMAC=0 PBKDF2=0
```

B — SHA compression function boundary:
```bash
SEEDPHRASE_NOINLINE_SHA512_WORDS=1
SEEDPHRASE_NOINLINE_FIXED64_HMAC=0
SEEDPHRASE_NOINLINE_PBKDF2=0
```

D — SHA + fixed64 HMAC boundaries:
```bash
SEEDPHRASE_NOINLINE_SHA512_WORDS=1
SEEDPHRASE_NOINLINE_FIXED64_HMAC=1
SEEDPHRASE_NOINLINE_PBKDF2=0
```

E — SHA boundary + force PBKDF2 to remain a separate device function:
```bash
SEEDPHRASE_NOINLINE_SHA512_WORDS=1
SEEDPHRASE_NOINLINE_FIXED64_HMAC=0
SEEDPHRASE_NOINLINE_PBKDF2=1
```

F — all three boundaries:
```bash
SEEDPHRASE_NOINLINE_SHA512_WORDS=1
SEEDPHRASE_NOINLINE_FIXED64_HMAC=1
SEEDPHRASE_NOINLINE_PBKDF2=1
```

C — HMAC-only noinline:
```bash
SEEDPHRASE_NOINLINE_SHA512_WORDS=0
SEEDPHRASE_NOINLINE_FIXED64_HMAC=1
SEEDPHRASE_NOINLINE_PBKDF2=0
```

C is lowest priority. Static ptxas already shows it increases PBKDF2 spill allocation from roughly 856/872 B to 1324/1360 B at the same 128-register cap.

You can use `scripts/bench_sm86_r2.sh` for a first screening pass, but do NOT use its sequential order as final proof because of thermal/clock drift.

## Required benchmark method

Phase 1 — screening:
- run A, B, D, E, F at least twice each
- C only if time permits
- record every `weighted_steady`
- record the built-in `Kernel resources:` line
- sample GPU power, SM clock, memory clock, temperature during each run

Reject obviously slower candidates early.

Phase 2 — rigorous finalists:
For each candidate that survives screening, compare against A with alternating order:

```text
A X X A
A X X A
A X X A
```

That is three complete alternating rounds.

A candidate is not a winner merely because of one fast run. Prefer promotion only if:
- correctness gates all pass
- it wins consistently across alternating rounds
- mean/median `weighted_steady` improvement is at least about 2%
- there is no thermal/power/clock explanation for the apparent gain

If a run is contaminated by another GPU workload, mark it and exclude it explicitly. Do not silently replace data.

## Profiling / resource data

Nsight Compute previously failed with `ERR_NVGPUCTRPERM`. Do not waste time repeatedly retrying ncu unless permissions have actually changed.

The program now prints via cudarc / cuFuncGetAttribute:
- registers/thread
- local memory/thread
- static shared memory/block
- max threads/block

Use those values for every experiment.

For static compiler allocation, use the existing GitHub Actions workflow manually when useful, or locally:

```bash
nvcc -cubin -arch=sm_86 \
  -DRECOVERY_LAUNCH_MIN_BLOCKS=2 \
  -DRECOVERY_NOINLINE_SHA512_WORDS=<0|1> \
  -DRECOVERY_NOINLINE_FIXED64_HMAC=<0|1> \
  -DRECOVERY_NOINLINE_PBKDF2=<0|1> \
  -Xptxas=-v \
  src/gpu/cuda/kernel.cu \
  -o /tmp/kernel.cubin
```

Known static reference at inline A:
- LB1: 255 regs, kernel stack 1344 B, no reported function spills
- LB2: 128 regs, kernel stack 1952 B; generic HMAC spills 512/512 B; PBKDF2 spills 856/872 B
- LB3: 80 regs, PBKDF2 spills 2760/3108 B
- LB4: 64 regs, PBKDF2 spills 3612/4372 B

This is why simply increasing min_blocks is no longer a useful direction.

## Decision after noinline/boundary tests

If B/D/E/F produces a real win:
- create a child branch `opt/sm86-rtx3070-r3` from R2
- commit only the winning default behavior plus diagnostics
- retain a way to reproduce A for comparison
- rerun self-test, recovery smoke test, and full alternating benchmark after the final commit
- do not merge to main yet

If none wins:
- leave R2 default behavior unchanged
- proceed to ONE controlled partial-unroll/live-range experiment at a time

For a partial-unroll experiment:
- preserve the 16-word rolling SHA-512 schedule
- do NOT reintroduce dynamic `W[16]` or `W[80]` arrays
- keep the 128-register / two-block residency target
- goal: lower hot-path spill traffic without forcing 80 or 64 regs
- inspect ptxas before hardware benchmarking
- self-test before performance testing

Do not start secp256k1 inversion/addition-chain work yet unless PBKDF2/SHA experiments are exhausted. PBKDF2 is still the dominant hot path.

## Optional power-efficiency pass

After selecting the fastest correct code path, if permissions allow changing the 3070 power limit, measure the final code at several power limits such as:

```text
160 W
180 W
200 W
220 W
240 W
```

Record:
- weighted_steady c/s
- actual average power
- c/s/W
- SM clocks and temperature

Do this only after code-path selection so power-limit variation does not contaminate compiler/kernel comparisons.

## Required report and Git delivery

Create and commit `SM86_RESULTS_R2.md`.

Include:
- exact branch + commit SHAs
- GPU model, driver, CUDA toolkit
- power limit and observed clocks/temperature
- all experiment env vars
- self-test results
- recovery smoke result
- every raw `weighted_steady`
- mean and median per candidate
- relative delta vs A
- built-in kernel resource attributes
- ptxas register/stack/spill data where collected
- all excluded/contaminated runs with reason
- final keep/reject decision for every tested candidate
- any remaining limitations

Also commit a compact `results/r2/` set containing only useful logs/CSV, not gigabytes of profiler scratch.

Push the result branch to origin. Keep the remote scratch directory until review is complete.

At the end, report:
1. fastest correct candidate and exact delta vs A
2. best c/s/W if power testing was done
3. whether a new R3 branch was created
4. exact commit SHA(s)
5. paths to `SM86_RESULTS_R2.md` and compact raw logs
6. anything that still blocks further optimization
