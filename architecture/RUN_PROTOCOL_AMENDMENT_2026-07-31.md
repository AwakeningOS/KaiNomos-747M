# KaiNomos-750M GPU run protocol amendment

This amendment replaces the earlier long smoke/burn-in plan with the active GPU
acceptance procedure.

## Active protocol

1. Confirm that no other training job is using the target GPU.
2. Run the FLA/reference and CUDA BF16 wiring checks.
3. Run KaiNomos-750M for three optimizer steps on the real data and save a
   resumable checkpoint.
4. If loss, gradients, data cursor, VRAM and hardware telemetry are clean,
   resume the same run for approximately 90 minutes.
5. Keep the existing 260 W power limit, active fan policy and five-second
   hardware telemetry during the run.
6. At the end, save a checkpoint and audit training/architecture/hardware logs.

The 200-step smoke and 2,000-step burn-in are no longer required.

## Commands

Startup segment:

```bash
PYTHONPATH=. python train.py \
  --architecture kainomos_750m_v1 \
  --data-dir /path/to/pool-v2-30b \
  --run-dir ../runs/kainomos_750m_delta_seed11 \
  --device cuda --allow-gpu --optimizer muon \
  --depth-routing delta_block --mtp off \
  --stop-after-steps 3
```

Resume the same run for the monitored segment:

```bash
PYTHONPATH=. python train.py \
  --architecture kainomos_750m_v1 \
  --data-dir /path/to/pool-v2-30b \
  --run-dir ../runs/kainomos_750m_delta_seed11 \
  --device cuda --allow-gpu --optimizer muon \
  --depth-routing delta_block --mtp off \
  --stop-after-minutes 90
```

The wall-clock limit is a segment boundary, not a new training schedule. Resume
uses the accumulated optimizer step, LR schedule, data cursor and RNG state.
