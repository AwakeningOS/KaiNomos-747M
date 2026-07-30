#!/usr/bin/env bash
# Muon against AdamW, same init, same data order, same schedule, same tokens.
#
# The question is which reaches the lower NLL at *equal tokens*, not per hour.
# That matters for how this is read: Muon's throughput penalty grows sharply with
# width (22% at h896, 46% at h1408, 48% at h1536), so a per-hour verdict measured
# small would not transfer.  A per-token verdict does, because the penalty is a
# cost of running the comparison and not a term in it.  That is what makes it
# legitimate to compare at 258M and apply the answer at 722M.
#
# Both arms are seeded identically and read the pool from position 0 in the same
# order, so the optimizer is the only difference between them; no seed averaging
# is needed.
#
#   Muon    300M / 5,767 tok/s = 14.4 h
#   AdamW   300M / 7,413 tok/s = 11.2 h
#                                25.6 h
#
# Read the *trend*, not the endpoint.  Moonshot report plain Muon leading early
# and then losing over a long run, which is why the decoupled weight decay and
# update-RMS matching in muon.py exist.  Whether they worked is only visible after
# the transient, which is why this is 300M and not 100M.  Fit the slope over the
# last third of each curve:
#
#   Muon ahead and the gap still widening   -> adopt, and h1536/722.5M becomes the
#                                              final size (20.24 GB, Muon only:
#                                              AdamW needs 22.77 GB there)
#   Muon ahead but the gap narrowing        -> that is the known transient. Do not
#                                              adopt on the endpoint alone.
#   AdamW ahead                             -> keep AdamW, and h1408/611.0M is the
#                                              largest safe size (20.33 GB)
set -uo pipefail
cd "$(dirname "$0")"
PY="$HOME/デスクトップ/mini_kimi_organism/.venv/bin/python"
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TOKENS="${TOKENS:-300000000}"
MB="${MB:-4}"
REST_MINUTES="${REST_MINUTES:-10}"
DATA="data/pool"

# 3% rather than the 1% default: at 300M the default would spend only ~46 steps
# ramping and most of the run would sit near peak, which compares the ramp rather
# than the optimizers.
WARMUP="${WARMUP:-0.03}"

for opt in adamw muon; do
  if [ "$opt" != "adamw" ]; then
    echo "=== resting the GPU for ${REST_MINUTES} min ==="
    nvidia-smi --query-gpu=temperature.gpu,memory.used --format=csv,noheader || true
    sleep $((REST_MINUTES * 60))
  fi
  echo "=== ${opt}: ${TOKENS} tokens ($(date '+%F %T')) ==="
  "$PY" -m train \
    --arm dense --optimizer "$opt" --seed 11 \
    --micro-batch "$MB" \
    --target-tokens "$TOKENS" \
    --schedule-tokens "$TOKENS" \
    --warmup-fraction "$WARMUP" \
    --max-epochs 1 \
    --device cuda --allow-gpu \
    --data-dir "$DATA" \
    --run-dir "runs/cmp_${opt}"
  echo "=== ${opt} finished ($(date '+%F %T')) ==="
done

echo "=== comparison complete ($(date '+%F %T')) ==="
echo "compare with:  $PY compare_optimizers.py"
