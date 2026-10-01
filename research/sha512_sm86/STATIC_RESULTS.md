# SM86 standalone SHA-512 microbenchmark — static screen

This research target is intentionally standalone. It does **not** call the wallet-recovery pipeline,
does not enumerate BIP39 material, and does not perform BIP32/address matching.

## Implementations

- `sha512_u64_kernel`: ordinary `uint64_t` SHA-512 compression with a 16-word rolling schedule.
- `sha512_pair32_kernel`: the same compression represented as explicit `{lo:u32, hi:u32}` pairs,
  using 32-bit carry chains and funnel shifts.

Both kernels:
- consume the same synthetic 128-byte block,
- initialize the standard SHA-512 IV,
- write all 8 state words,
- support repeated compression for throughput measurement.

The standalone host binary performs a mandatory SHA-512("abc") one-block correctness test for
both kernels before benchmarking.

## CUDA 12.9 / sm_86 static result

CI run: `SM86 SHA512 microbench static screen` at commit `a329350`.

| kernel | regs/thread | stack | spill | SASS lines | SASS bytes |
|---|---:|---:|---:|---:|---:|
| u64 | **210** | 0 | 0/0 | 7,268 | 835,484 |
| pair32 | **96** | 0 | 0/0 | 8,148 | 936,684 |

Selected SASS counts:

| kernel | IADD3 family | SHF family | LOP3 |
|---|---:|---:|---:|
| u64 | 761 | 1,728 | 900 |
| pair32 | 1,067 | 1,668 | 894 |

Interpretation:

- pair32 cuts register allocation from 210 to 96 registers/thread, a major occupancy opportunity on
  SM86's 64K 32-bit register file.
- pair32 is not instruction-cheaper: SASS is ~12% larger and has substantially more integer adds.
- neither variant spills, so the hardware experiment is specifically an
  **instruction-count vs occupancy** trade-off.
- static results are not sufficient to choose a winner; real RTX 3070 timing is required.

No result from this research branch should be merged into or used to modify the wallet-recovery
pipeline as part of this experiment.
