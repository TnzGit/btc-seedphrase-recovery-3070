#!/usr/bin/env bash
RD=/tmp/btc3070bench-j9PBaw
export LD_LIBRARY_PATH="$RD/nvrtc/usr/lib/x86_64-linux-gnu:/usr/lib/wsl/lib"
for tag in main opt_sm86-rtx3070-core opt_sm86-rtx3070-v1; do
  echo "########## SELFTEST $tag ##########"
  s=$(date +%s.%N)
  printf "9\n" | timeout 300 "$RD/bin/seed-$tag" > "$RD/selftest-$tag.log" 2>&1
  st=$?
  e=$(date +%s.%N)
  echo "exit=$st wall=$(echo "$e - $s" | bc)s"
  grep -a -E "Device:|self-test|PASS|FAIL|Aborting|error" "$RD/selftest-$tag.log" | head -10
done
echo "########## SELFTEST DONE ##########"
