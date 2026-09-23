#!/usr/bin/env bash
set -euo pipefail

BIN="${BIN:-./target/release/seedphrase_recovery}"
BLOCKS="${BLOCKS:-64 128 256 512}"

echo "GPU:"
nvidia-smi --query-gpu=name,driver_version,power.limit,clocks.sm,clocks.mem,temperature.gpu --format=csv,noheader || true
echo

for b in $BLOCKS; do
  echo "===== SEEDPHRASE_BLOCK=$b ====="
  SEEDPHRASE_BLOCK="$b" "$BIN" --bench
  echo
done
