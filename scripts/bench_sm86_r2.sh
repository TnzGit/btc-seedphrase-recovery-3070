#!/usr/bin/env bash
set -euo pipefail

BIN="${BIN:-./target/release/seedphrase_recovery}"
export SEEDPHRASE_BLOCK="${SEEDPHRASE_BLOCK:-256}"
export SEEDPHRASE_LB_MIN_BLOCKS="${SEEDPHRASE_LB_MIN_BLOCKS:-2}"

run_case() {
  local name="$1"
  local sha_noinline="$2"
  local hmac_noinline="$3"
  local pbkdf2_noinline="$4"
  echo
  echo "===== $name ====="
  SEEDPHRASE_NOINLINE_SHA512_WORDS="$sha_noinline" \
  SEEDPHRASE_NOINLINE_FIXED64_HMAC="$hmac_noinline" \
  SEEDPHRASE_NOINLINE_PBKDF2="$pbkdf2_noinline" \
    "$BIN" --bench
}

# Screening matrix only. Do not use this sequential order as final proof of a winner;
# finalists must be re-tested with alternating A/B/B/A ordering to control drift.
run_case A-inline                 0 0 0
run_case B-sha-noinline          1 0 0
run_case D-sha-hmac-noinline     1 1 0
run_case E-sha-pbkdf2-noinline   1 0 1
run_case F-all-boundaries        1 1 1
run_case C-hmac-noinline         0 1 0
