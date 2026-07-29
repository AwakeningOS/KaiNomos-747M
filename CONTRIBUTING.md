# Contributing

KaiNomos welcomes reproducible architecture experiments, careful bug reports,
negative results, and documentation improvements.

## Before proposing a mechanism

1. Search existing issues, discussions, and relevant literature.
2. State the mechanism and the expected causal effect.
3. Define the smallest experiment that could disprove the claim.
4. Keep data, initialization, token count, and compute comparable to the control.
5. Register the experiment before looking at the held-out test result.

Use the Architecture Proposal issue form for an unimplemented idea and the
Experiment Report form after a run.

## Reproducibility requirements

An experiment report should include:

- experiment ID and commit SHA
- base checkpoint and SHA-256
- exact config and seed
- tokenizer and dataset-manifest hashes
- GPU, software versions, peak VRAM, and wall time
- training tokens and executed analytical compute
- validation and held-out metrics
- complete logs or a durable artifact link
- implementation bugs found during the run
- `adopted`, `rejected`, or `inconclusive` verdict

Do not delete or silently replace failed runs. If an implementation bug
invalidates a result, retain the record and mark it invalid.

## Pull requests

- Keep one causal claim per pull request when practical.
- Add tests for new mechanisms and identity/disabled modes.
- Run `pytest -q`.
- Do not commit datasets, secrets, or `.pt`/`.safetensors` checkpoints.
- Explain how the comparison controls parameter count, tokens, and compute.

By contributing, you agree that your contribution is licensed under Apache-2.0.
