#!/usr/bin/env bash
# Record GPU telemetry beside the training log.
#
# Keep an independent time series across long runs. If the machine freezes,
# process logs alone may not identify whether the cause was software, thermal,
# power delivery, PCIe, or another hardware path.
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
