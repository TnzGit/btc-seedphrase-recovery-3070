# Seedphrase Recovery Tool (BTC, CUDA)

Recover missing words of a BIP39 seed phrase for a Bitcoin native SegWit (`bc1q...`) wallet, fully on the GPU. No RPC, no network requests - matching is local against the address you give it.

## Requirements

- Linux (Ubuntu / Debian / Fedora / RHEL / Arch). On Windows use WSL with GPU passthrough; macOS is unsupported.
- NVIDIA GPU + driver. Confirm with `nvidia-smi`.

## Install

```bash
curl -O https://raw.githubusercontent.com/zunmax/btc-seedphrase-recovery/main/setup.sh && chmod +x setup.sh && ./setup.sh
```

The script installs build tools, the CUDA toolkit (if missing), and Rust; clones the repo if you ran it from outside; and builds the release binary. At the end it prints the exact path to run.

## Run

```bash
./target/release/seedphrase_recovery
```

The tool will:

1. Initialize CUDA (10-20 s on first run, NVRTC compiles the kernel).
2. Self-test the GPU pipeline against three BIP84 reference vectors. Aborts on mismatch.
3. Ask for: seed phrase length, number of missing words (1-3), the known words in order, whether you know the missing positions, and the target `bc1q...` address.
4. Scan. Progress bar shows rate in M c/s. Ctrl-C once to stop cleanly, twice to force-exit.
5. On success: print the recovered missing words and the full seed phrase.

## Benchmark your GPU

```bash
./target/release/seedphrase_recovery --bench
```
Runs 5 chunks of 4 M candidates and prints the rate per chunk. Useful to confirm your hardware is hitting expected throughput (~1.85 M c/s on RTX 5090).

## Derivation paths

Default `m/84'/0'/0'/0/0`. If the default does not match, the tool offers 22 alternative paths (receive indices 0/1 - 0/19, accounts 1'/0/0 and 2'/0/0, change `0'/1/0`).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA init failed` / `libcuda.so` missing | NVIDIA driver not installed. Install it via your distro's package manager. |
| Build fails on `secp256k1-sys` | Missing `build-essential` or `libssl-dev`. Re-run `./setup.sh`. |
| Self-test FAILS | Do not run a real recovery. Open an issue with your GPU/driver version. |
| `--bench` reports rate much lower than expected | Check `nvidia-smi` for thermal/power throttling or other CUDA processes holding the GPU. |

## Attribution

`src/gpu/cuda/kernel.cu` adapts secp256k1 field arithmetic, SHA-256, and RIPEMD-160 from [BitCrack](https://github.com/brichard19/BitCrack) (MIT). All BIP32 derivation, BIP39 candidate enumeration, windowed scalar mult, and pipeline integration are original.

## Disclaimer

For legal recovery of wallets you own. The author is not responsible for misuse.
