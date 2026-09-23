#!/usr/bin/env bash
set -euo pipefail

BIN="${BIN:-./target/release/seedphrase_recovery}"
BLOCK="${SEEDPHRASE_BLOCK:-128}"
OUT="${OUT:-ncu-sm86}"

echo "Profiling recovery_enumerate at block size $BLOCK"
echo "Skipping the 3 built-in self-test launches and capturing the first benchmark launch."
echo "If this produces too much output, switch --set full to --set detailed."
SEEDPHRASE_BLOCK="$BLOCK" ncu   --set full   --kernel-name regex:recovery_enumerate   --launch-count 1   --export "$OUT"   "$BIN" --bench
