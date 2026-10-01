#!/usr/bin/env bash
set -uo pipefail
REPO=/home/user/r3hw/btc-seedphrase-recovery-3070
BIN=$REPO/target/release/seedphrase_recovery
SMI=/usr/lib/wsl/lib/nvidia-smi
LOGD=$REPO/results/r4/logs
TELD=$REPO/results/r4/telemetry
mkdir -p "$LOGD" "$TELD"

run_case() {
  local runid="$1" nvrtc_dir="$2" lbmax="$3" lbmin="$4" block="$5"
  shift 5
  local envline="LD_LIBRARY_PATH=$nvrtc_dir:/usr/lib/wsl/lib SEEDPHRASE_LB_MAX_THREADS=$lbmax SEEDPHRASE_LB_MIN_BLOCKS=$lbmin SEEDPHRASE_BLOCK=$block"
  local kv
  for kv in "$@"; do envline="$envline $kv"; done
  local telfile="$TELD/tel-$runid.csv"
  local logfile="$LOGD/out-$runid.log"
  echo "util,power,sm_clk,mem_clk,temp" > "$telfile"
  "$SMI" --query-gpu=utilization.gpu,power.draw,clocks.sm,clocks.mem,temperature.gpu --format=csv,noheader,nounits >> "$telfile"
  ( while true; do "$SMI" --query-gpu=utilization.gpu,power.draw,clocks.sm,clocks.mem,temperature.gpu --format=csv,noheader,nounits >> "$telfile"; sleep 1; done ) &
  local sampid=$!
  local t0=$(date +%s)
  {
    echo "=== RUN $runid env: $envline ==="
    echo "=== NVRTC probe (same dir, same LD_LIBRARY_PATH): ==="
    LD_LIBRARY_PATH="$nvrtc_dir:/usr/lib/wsl/lib" python3 /home/user/r4env/nvrtc_ver.py "$nvrtc_dir" 2>&1
  } > "$logfile" 2>&1
  env LD_LIBRARY_PATH="$nvrtc_dir:/usr/lib/wsl/lib" SEEDPHRASE_LB_MAX_THREADS=$lbmax SEEDPHRASE_LB_MIN_BLOCKS=$lbmin SEEDPHRASE_BLOCK=$block "$BIN" --bench >> "$logfile" 2>&1
  local rc=$?
  local t1=$(date +%s)
  kill $sampid 2>/dev/null; wait $sampid 2>/dev/null
  local steady=$(grep -o "weighted_steady = [0-9]*" "$logfile" | tail -1 | grep -o "[0-9]*")
  local wallall=$(grep -o "weighted_all = [0-9]*" "$logfile" | tail -1 | grep -o "[0-9]*")
  local kres=$(grep "Kernel resources:" "$logfile")
  local nver=$(grep -o "nvrtcVersion=[0-9.]*" "$logfile" | head -1)
  local nsm=$(wc -l < "$telfile")
  local avgp=$(awk -F, "NR>1{s+=\$2;n++} END{if(n)printf \"%.1f\", s/n}" "$telfile")
  local maxp=$(awk -F, "NR>1{if(\$2>m)m=\$2} END{print m+0}" "$telfile")
  local avgs=$(awk -F, "NR>1{s+=\$3;n++} END{if(n)printf \"%.0f\", s/n}" "$telfile")
  local maxt=$(awk -F, "NR>1{if(\$5>m)m=\$5} END{print m+0}" "$telfile")
  echo "{\"runid\":\"$runid\",\"nvrtc_dir\":\"$nvrtc_dir\",\"nvrtc_ver\":\"$nver\",\"lb_max\":$lbmax,\"lb_min\":$lbmin,\"block\":$block,\"extra_env\":\"$envline\",\"exit\":$rc,\"wall_s\":$((t1-t0)),\"steady\":${steady:-0},\"all\":${wallall:-0},\"kernel_resources\":\"$kres\",\"telemetry_rows\":$nsm,\"avg_power_W\":\"$avgp\",\"max_power_W\":\"$maxp\",\"avg_sm_clk\":\"$avgs\",\"max_temp_C\":\"$maxt\"}" >> "$LOGD/summary.jsonl"
  echo "runid=$runid nvrtc=$nver exit=$rc steady=$steady wall=$((t1-t0))s"
}

run_case "$@"
