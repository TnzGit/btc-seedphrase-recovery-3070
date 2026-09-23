#!/usr/bin/env bash
RD=/tmp/btc3070bench-j9PBaw
export LD_LIBRARY_PATH="$RD/nvrtc/usr/lib/x86_64-linux-gnu:/usr/lib/wsl/lib"
mkdir -p "$RD/results/rep"
# b256 is the winner on all three branches; 3 repeats each for significance
for r in 1 2 3; do
 for tag in main opt_sm86-rtx3070-core opt_sm86-rtx3070-v1; do
   run="r${r}-${tag}"
   sleep 20
   SEEDPHRASE_BLOCK=256 timeout 600 "$RD/bin/seed-$tag" --bench > "$RD/results/rep/$run.log" 2>&1
   tot=$(grep -a -oE "elapsed = [0-9.]+" "$RD/results/rep/$run.log" | grep -oE "[0-9.]+" | awk "{s+=\$1} END {printf \"%.3f\", s}")
   nc=$(grep -ac "chunk #" "$RD/results/rep/$run.log")
   w=$(echo "scale=0; 20971520 / $tot" | bc 2>/dev/null)
   echo "$run chunks=$nc sum=$tot weighted=$w"
 done
done
echo "REPEATS DONE"
