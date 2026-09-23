#!/usr/bin/env bash
set -euo pipefail

BIN="${BIN:-./target/release/seedphrase_recovery}"
export SEEDPHRASE_BLOCK="${SEEDPHRASE_BLOCK:-256}"
export SEEDPHRASE_LB_MIN_BLOCKS="${SEEDPHRASE_LB_MIN_BLOCKS:-2}"

run_case() {
  local name="$1"
  local sha_noinline="$2"
  local hmac_noinline="$3"
  echo
  echo "===== $name ====="
  SEEDPHRASE_NOINLINE_SHA512_WORDS="$sha_noinline"   SEEDPHRASE_NOINLINE_FIXED64_HMAC="$hmac_noinline"     "$BIN" --bench
}

run_case A-inline 0 0
run_case B-sha-noinline 1 0
run_case C-hmac-noinline 0 1
run_case D-both-noinline 1 1
