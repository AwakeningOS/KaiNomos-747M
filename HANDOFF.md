# KaiNomos-747M handoff

Current production target: **747,368,168 training parameters** on one RTX 3090
24 GB GPU. The inference backbone is 702,134,416 parameters and the training-only
MTP head is 45,233,752 parameters.

## Fixed architecture

- 16 decoder layers, hidden size 1,536
- `KKKM` × 4: 12 KDA layers and 4 NoPE Gated MLA layers
- 24 attention heads × 64 dimensions
- dense SiTU-GLU FFN width 6,144 in every layer
- MUDD-QKV depth mixtures
- block-granularity Delta Attention Residual retrieval
- tied 49,152-piece embedding and LM head
- MTP-1 auxiliary loss, weight 0.30
- training context 1,024

The authoritative machine-readable specification is
`kainomos-747m-architecture.json`.

## Data contract

Training consumes the schema-v2 `DoubleDragon-DataMix-v2` packed pool:

- `manifest.json`
- the ordered `splits.train.shards` list
- `tokenizer.vocab_size = 49152`
- `eod_token_id = 4`
- uint16 token shards

`train.py` follows the manifest's declared shard order, validates every path and
token count, and checkpoints a global stream offset. Resume therefore remains
exact even when a sequence crosses a shard boundary. The pool is read once by
default; repeating data requires an explicit larger `--max-epochs` value.

Validation and test evaluation also read their shard lists from the same
manifest.

## Training and recovery

The production optimizer is Muon for eligible matrices with AdamW for the
remaining parameters. Checkpoints are written every 50 optimizer steps, with
the newest two rotating resume checkpoints retained. Milestone checkpoints and
the weights-only observation ladder are kept separately.

Typical launch:

```bash
PYTHONPATH=. python train.py \
  --data-dir /path/to/packed-pool \
  --run-dir /path/to/run \
  --device cuda \
  --allow-gpu \
  --optimizer muon \
  --schedule-tokens 17000000000
```

Re-running the same command resumes from the newest `step_*.pt`. Use
`--additional-tokens` for a new segment beyond an existing checkpoint.

## Hardware-failure investigation

For long runs, retain independent five-second telemetry for GPU core, hotspot
and VRAM temperature, power, clocks, PCIe replay counters and driver recovery
state. A manual reset cannot flush ordinary process logs, so checkpoint files
and telemetry must live on persistent storage and be inspected after reboot.

If continuous load reproduces the freeze, the planned fallback is a 30-minute
training / 5-minute idle cycle. Do not introduce that duty cycle until a
continuous run has provided a clean comparison.

## Release state

Implementation checks and the single-GPU smoke path are complete. Full
pre-training, held-out evaluation and trained weight publication remain pending.
Do not claim model quality before those artifacts exist.
