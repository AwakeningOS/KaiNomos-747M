#!/usr/bin/env bash
# Acquire the whole 2.5B-token mix at low priority.
#
# GPU training keeps precedence: this runs under `nice` and `ionice -c3` so the
# download never competes with the training process for CPU or disk.  Every
# source is resumable -- stream offset, byte count, shard index, failed blob ids
# and provenance are written as it goes -- so an interrupted run continues where
# it stopped instead of restarting.
set -euo pipefail
cd "$(dirname "$0")"
# Override when the project uses a dedicated environment:
#   K3MINI_PYTHON=.venv/bin/python ./acquire_all.sh
PY="${K3MINI_PYTHON:-python3}"
export PYTHONPATH=.
RUN=(nice -n 19 ionice -c3)

echo "=== text sources ($(date '+%F %T')) ==="
"${RUN[@]}" "$PY" data_mix.py collect --root data/mix \
  --only ja_web,ja_paraphrase,ja_instruct,ja_wikipedia_reference,en_edu,math

echo "=== code sources via Software Heritage ($(date '+%F %T')) ==="
while read -r lang mb; do
  echo "--- $lang target ${mb}MB"
  "${RUN[@]}" "$PY" data_code.py --root data/mix --language "$lang" --target-mb "$mb" --workers 24
done <<'LANGS'
Python 420
Markdown 170
Shell 70
SQL 60
JavaScript 50
TypeScript 30
C 30
Cpp 20
LANGS

echo "=== manifest ==="
"${RUN[@]}" "$PY" data_mix.py manifest --root data/mix
echo "=== acquisition finished ($(date '+%F %T')) ==="
