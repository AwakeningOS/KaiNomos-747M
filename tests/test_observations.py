"""The observation ladder: snapshots for watching the model grow.

These are not resume points.  They exist so that at any later date the run can be
re-read at 50M, 100M, 200M ... tokens without retraining anything, which is the
only way to tell a genuine change of slope from a metric crossing a threshold.
"""

import torch

from config import K3MiniPlusPlusPlusConfig as Config
from model import K3MiniPlusPlusPlusForCausalLM as Model
from train import TrainConfig, save_observation


def crossed(tokens_done: int, per_step: int, ladder) -> list[int]:
    """The rungs a step landing on `tokens_done` just crossed."""
    return [m for m in ladder
            if tokens_done >= m and tokens_done - per_step < m]


def test_ladder_is_absolute_and_log_spaced():
    cfg = TrainConfig()
    ladder = cfg.observation_tokens
    assert ladder == tuple(sorted(ladder)), "rungs must increase"
    assert len(set(ladder)) == len(ladder)
    # Absolute token counts, not fractions: a segmented run changes its target
    # every night, and fractions would move the rungs between segments.
    assert all(isinstance(m, int) and m > 0 for m in ladder)
    assert ladder[0] == 50_000_000 and ladder[-1] == 20_000_000_000
    # Early growth is sampled at least as densely as the tail, in log terms.
    early = ladder[1] / ladder[0]
    late = ladder[-1] / ladder[-2]
    assert early >= late


def test_each_rung_fires_exactly_once_across_a_run():
    cfg = TrainConfig()
    per_step = 65_536
    fired = []
    tokens = 0
    # walk far enough to pass the first four rungs
    while tokens < 450_000_000:
        tokens += per_step
        fired.extend(crossed(tokens, per_step, cfg.observation_tokens))
    assert fired == [50_000_000, 100_000_000, 200_000_000, 400_000_000]
    assert len(fired) == len(set(fired)), "a rung fired twice"


def test_a_large_step_can_cross_several_rungs_at_once():
    """With a big batch one step may pass more than one early rung."""
    ladder = TrainConfig().observation_tokens
    assert crossed(120_000_000, 120_000_000, ladder) == [50_000_000, 100_000_000]


def test_snapshot_holds_weights_only_and_reloads():
    cfg = Config.tiny()
    cfg.joint_route.enabled = False
    torch.manual_seed(3)
    model = Model(cfg)

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory:
        path = save_observation(Path(directory), model, cfg, 7, 50_000_000, 50_000_000)
        assert path.name == "obs_000050M.pt"
        blob = torch.load(path, map_location="cpu", weights_only=False)

    # what it must carry
    assert blob["tokens_done"] == 50_000_000
    assert blob["observation_mark"] == 50_000_000
    assert blob["step"] == 7
    # what it must NOT carry: this is for reading, not resuming, and the optimizer
    # state would nearly triple the file
    for absent in ("optimizer", "python_rng", "numpy_rng", "torch_rng", "cuda_rng"):
        assert absent not in blob, absent

    # the weights round-trip exactly, in float32 regardless of training precision
    reloaded = Model(Config.from_dict(blob["config"]))
    reloaded.load_state_dict(blob["model"])
    for (name, saved), (_, live) in zip(
        reloaded.state_dict().items(), model.state_dict().items()
    ):
        assert saved.dtype == torch.float32, name
        assert torch.equal(saved, live.float()), name
