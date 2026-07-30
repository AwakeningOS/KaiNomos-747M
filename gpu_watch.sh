#!/usr/bin/env bash
# Record GPU telemetry beside the training log.
#
# The 2026-07-29 run died mid-arm with no kernel panic -- the journal simply
# stopped, seconds after lact reported it had lost fan control. That is the
# signature of a hard reset, not a software fault, so the next failure needs
# evidence from the hardware side rather than from train.jsonl.
#
# This only observes. It does not change clocks, fan curves or power limits:
# those are the machine owner's to set.
set -u
OUT="${1:-runs/gpu_telemetry.csv}"
INTERVAL="${2:-15}"
[ -f "$OUT" ] || echo "timestamp,temp_c,fan_pct,power_w,util_pct,mem_mib,throttle" > "$OUT"
while true; do
  line=$(nvidia-smi --format=csv,noheader,nounits \
    --query-gpu=temperature.gpu,fan.speed,power.draw,utilization.gpu,memory.used 2>/dev/null)
  throttle=$(nvidia-smi -q -d PERFORMANCE 2>/dev/null \
    | grep -E "HW (Thermal|Power Brake) Slowdown|SW Power Cap" \
    | grep -c ": Active")
  printf '%s,%s,%s\n' "$(date '+%F %T')" "${line// /}" "${throttle:-0}" >> "$OUT"
  sleep "$INTERVAL"
done
