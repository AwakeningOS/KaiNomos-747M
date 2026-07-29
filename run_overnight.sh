#!/usr/bin/env bash
# One overnight segment: both arms, the same token count, a rest between them.
#
# The arms must see the *same* tokens for the comparison to mean anything, so the
# budget is set per arm rather than by wall clock -- Adaptive is slower, and
# stopping both at the same hour would hand Base more data.
#
#   Base     160M / 14,242 tok/s = 3.12 h
#   Adaptive 160M / 12,139 tok/s = 3.66 h
#   rest                          0.17 h
#                                 6.95 h
#
# Both arms resume from wherever they stopped, so tomorrow continues rather than
# restarts.
set -uo pipefail
cd "$(dirname "$0")"
PY="$HOME/デスクトップ/mini_kimi_organism/.venv/bin/python"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TOKENS="${TOKENS:-160000000}"
REST_MINUTES="${REST_MINUTES:-10}"
DATA="data/pool"
INIT="runs/kainomos_110m_init.pt"

for arm in base adaptive; do
  if [ "$arm" != "base" ]; then
    echo "=== resting the GPU for ${REST_MINUTES} min ==="
    nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used --format=csv,noheader || true
    sleep $((REST_MINUTES * 60))
  fi
  echo "=== ${arm}: ${TOKENS} tokens ($(date '+%F %T')) ==="
  "$PY" -m train \
    --arm "$arm" --seed 11 \
    --micro-batch 6 \
    --target-tokens "$TOKENS" \
    --init-from "$INIT" \
    --device cuda --allow-gpu \
    --data-dir "$DATA" \
    --run-dir "runs/${arm}_seed11"
  echo "=== ${arm} finished ($(date '+%F %T')) ==="
done
echo "=== overnight segment complete ($(date '+%F %T')) ==="
