#!/usr/bin/env bash
RD=/tmp/btc3070bench-j9PBaw
export LD_LIBRARY_PATH="$RD/nvrtc/usr/lib/x86_64-linux-gnu:/usr/lib/wsl/lib"
SMI=/usr/lib/wsl/lib/nvidia-smi
mkdir -p "$RD/results/abba"
# ABBA-balanced order to cancel thermal/clock drift: core,v1,v1,core,core,v1,v1,core
ORDER="core v1 v1 core core v1 v1 core"
i=0
for short in $ORDER; do
  i=$((i+1))
  case "$short" in
    core) tag=opt_sm86-rtx3070-core ;;
    v1)   tag=opt_sm86-rtx3070-v1 ;;
  esac
  run="abba${i}-${short}"
  sleep 20
  TEL="$RD/results/abba/tel-$run.csv"
  ( while true; do $SMI --query-gpu=utilization.gpu,power.draw,clocks.sm,temperature.gpu --format=csv,noheader,nounits >> "$TEL" 2>/dev/null; sleep 1; done ) &
  MON=$!
  SEEDPHRASE_BLOCK=256 timeout 600 "$RD/bin/seed-$tag" --bench > "$RD/results/abba/$run.log" 2>&1
  kill $MON 2>/dev/null; wait $MON 2>/dev/null
  tot=$(grep -a -oE "elapsed = [0-9.]+" "$RD/results/abba/$run.log" | grep -oE "[0-9.]+" | awk "{s+=\$1} END {printf \"%.3f\", s}")
  nc=$(grep -ac "chunk #" "$RD/results/abba/$run.log")
  w=$(echo "scale=0; 20971520 / $tot" | bc 2>/dev/null)
  smclk=$(awk -F, "{s+=\$3; n++} END {if(n>0) printf \"%.0f\", s/n}" "$TEL" 2>/dev/null)
  pwr=$(awk -F, "{s+=\$2; n++} END {if(n>0) printf \"%.1f\", s/n}" "$TEL" 2>/dev/null)
  echo "$run chunks=$nc sum=$tot weighted=$w avg_smclk=$smclk avg_power=$pwr"
done
echo "ABBA DONE"
