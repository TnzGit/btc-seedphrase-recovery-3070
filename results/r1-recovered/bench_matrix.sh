#!/usr/bin/env bash
RD=/tmp/btc3070bench-j9PBaw
export LD_LIBRARY_PATH="$RD/nvrtc/usr/lib/x86_64-linux-gnu:/usr/lib/wsl/lib"
SMI=/usr/lib/wsl/lib/nvidia-smi
mkdir -p "$RD/results"
echo "### GPU STATE BEFORE MATRIX"
$SMI --query-gpu=name,driver_version,power.limit,power.default_limit,clocks.max.sm,clocks.max.mem,temperature.gpu,pstate --format=csv,noheader
echo "### nvidia-smi -q power/clocks"
$SMI -q -d POWER,CLOCK 2>/dev/null | grep -a -E "Power Limit|Default Power Limit|Enforced Power Limit|Graphics *:|SM *:|Memory *:" | head -12

for tag in main opt_sm86-rtx3070-core opt_sm86-rtx3070-v1; do
  for b in 64 128 256 512; do
    run="${tag}__b${b}"
    echo "########## RUN $run ##########"
    # cooldown so each run starts from comparable thermal state
    sleep 20
    TEL="$RD/results/tel-$run.csv"
    echo "timestamp,util,mem_used,temp,power,sm_clk" > "$TEL"
    ( while true; do
        $SMI --query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm --format=csv,noheader,nounits >> "$TEL" 2>/dev/null
        sleep 1
      done ) &
    MON=$!
    s=$(date +%s.%N)
    SEEDPHRASE_BLOCK="$b" timeout 900 "$RD/bin/seed-$tag" --bench > "$RD/results/out-$run.log" 2>&1
    st=$?
    e=$(date +%s.%N)
    kill $MON 2>/dev/null; wait $MON 2>/dev/null
    echo "exit=$st wall=$(echo "$e - $s" | bc)"
    grep -a -E "Chunk|Total|rate|candidates" "$RD/results/out-$run.log" | head -20
  done
done
echo "########## MATRIX DONE ##########"
