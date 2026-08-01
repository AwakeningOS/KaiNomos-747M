# Train and talk with KaiNomos-750M

This guide takes a fresh clone through the complete path:

```text
raw documents -> included tokenizer -> packed shards -> GPU training
              -> exact resume -> interactive completion
```

KaiNomos is GPU-first. The tuned full-model configuration was measured on one
24 GB RTX 3090 and deliberately operates close to a 22 GiB reserved-VRAM
ceiling. A CUDA GPU with about 24 GB of VRAM is therefore the recommended
starting point.

## 1. Install the CUDA environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-cuda.txt
```

The exact 49,152-piece SentencePiece tokenizer is included under `tokenizer/`.

## 2. Supply training data

The packer accepts UTF-8 plain text, JSONL, and `.jsonl.gz` files. In plain
text, each non-empty line is treated as one document. A JSONL record must have
a string `text` field:

```json
{"text":"This is one complete training document."}
```

Dialogue data can use the same JSONL field:

```json
{"text":"User: Explain photosynthesis simply.\nAssistant: Plants use light to convert water and carbon dioxide into stored chemical energy."}
```

## 3. Pack the data

```bash
python tools/prepare_data.py \
  --input corpus/train.jsonl \
  --output data/myrun \
  --split train \
  --source-id mydata \
  --weight 1.0

python tools/prepare_data.py \
  --input corpus/validation.jsonl \
  --output data/myrun \
  --split validation \
  --source-id mydata
```

The command creates uint16 token shards, uint64 document offsets, and the
`manifest.json` consumed by the trainer. It appends the `<|eod|>` token (ID 4)
to each document so attention, recurrent state, convolution state, and loss are
all reset at the same boundary.

To mix another training source, run the first command again with a different
`--source-id` and weight. Add `--minor` for a source that must not drive the
main-source sampling constraint.

## 4. Start GPU training

First run one optimizer step as an end-to-end acceptance check:

```bash
python scripts/run_kainomos_runtime_tuned.py \
  --runtime-activation-checkpointing on \
  --runtime-micro-batch 16 \
  --runtime-checkpoint-every-steps 200 \
  --architecture kainomos_750m_v1 \
  --data-dir data/myrun \
  --run-dir runs/myrun \
  --device cuda \
  --allow-gpu \
  --optimizer muon \
  --depth-routing delta_block \
  --mtp off \
  --target-tokens 65536
```

For a longer run, change only `--target-tokens` to the intended total. The
selected runtime always rounds the requested ceiling up to a complete 65,536-
token optimizer step and records both values. Keep the same data, model, and
schedule options throughout one run.

The command starts from random initialization when `runs/myrun/latest.json`
does not exist. Re-running the same command later loads the latest checkpoint
and restores the model, optimizer, RNG states, source cursors, and unconsumed
read-ahead tokens. Periodic checkpoints default to every 200 steps in this
runtime wrapper, with an additional save at the final requested step.

The selected 22 GiB profile and a lower-memory fallback are documented in
[EFFICIENCY.md](EFFICIENCY.md).

## 5. Open an interactive session

Choose a saved checkpoint and run:

```bash
python examples/chat.py --checkpoint runs/myrun/step_00000001.pt
```

Use `/clear` to discard the transcript and `/exit` to quit.
Prompt prefill and temporal caches are enabled automatically; see
[INFERENCE.md](INFERENCE.md) for the measured path.

## What must remain compatible

- `tokenizer/kainomos-49152.model` and its SHA-256;
- vocabulary size 49,152 and EOD ID 4;
- model configuration stored in the checkpoint;
- data manifest and deterministic source order when resuming;
- optimizer and schedule settings for the same training run.

The trainer rejects several mismatches rather than silently performing an
approximate resume.
