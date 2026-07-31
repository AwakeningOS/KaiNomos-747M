# KaiNomos-750M acceptance status

Updated 2026-07-31.

## Passed

- 24-layer `(KDA,KDA,KDA,MLA) × 6` structure and final MLA
- zero MuDD parameters and state keys
- 48 zero-initialized Delta queries and exact first-attention identity
- checkpoint-on/off forward and gradient equivalence on the CPU tiny model
- KDA initialization, causal behavior, cache and packed-document isolation
- strict-NoPE MLA latent-only cache and cached generation equivalence
- MTP OFF/ON parameter budgets and backbone initialization invariance
- shared-LR explicit Muon/AdamW classification and Per-Head reference update
- fixed-token source mixing, first-100-step major-source gate and exact resume
- atomic checkpoint save/load and stream/step/token alignment rejection
- final-full-validation-only A/B comparison contract
- JSON-safe Delta/KDA/MLA/optimizer/generation observations
- Python compilation, Ruff and the complete CPU test suite

## Pending GPU gates

- FLA KDA forward and gradient comparison against the reference
- CUDA BF16 three-step end-to-end check
- CUDA checkpoint/cache/compile integration check
- peak VRAM below 22.5 GiB
- short three-step startup check
- one monitored approximately 90-minute real-data training segment

## Pending scientific gates

- 67,108,864-token normal-residual versus Delta Block screen
- optional extension to 255,983,616 tokens according to the frozen rule
- MTP OFF versus ON screen after architecture selection
- promotion manifest before the 32.55B-token production run

The GPU gates require exclusive access to the target GPU and independent
hardware telemetry.

The earlier 200-step smoke and 2,000-step burn-in plan was superseded on
2026-07-31. See `RUN_PROTOCOL_AMENDMENT_2026-07-31.md`.
