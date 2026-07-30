"""Muon for the hidden matrices, AdamW for everything else.

Muon orthogonalises the momentum before applying it, so every direction in a
weight matrix is updated at a comparable rate instead of at a rate set by that
direction's gradient magnitude.  That is something AdamW's per-coordinate
rescaling cannot express: it normalises each entry independently and has no view
of the matrix as a map.

This follows the Moonshot formulation (Muon is Scalable for LLM Training,
arXiv 2502.16982), not the original minimal one.  Two additions matter:

* **Decoupled weight decay.**  Without it the orthogonalised update keeps the
  weight norm growing -- the paper reports the plain version converging faster
  early and then losing to AdamW over a long run.
* **Update RMS matched to AdamW.**  The orthogonalised matrix has unit singular
  values, so its per-element RMS is `1/sqrt(max(A, B))` and therefore depends on
  the shape of whatever it is updating.  Rescaling by `0.2 * sqrt(max(A, B))`
  puts every matrix at the ~0.2 RMS AdamW would have produced, which is what
  makes a single learning rate meaningful across a model whose matrices range
  from 896x128 to 3584x896.

Parameters that are not matrices go to AdamW, because orthogonalisation is
meaningless for them:

    Muon    every 2-D weight that acts as a linear map -- KDA/MLA projections,
            the FFN, MUDD's coefficient MLP, the Delta key projections, MTP's fuse
    AdamW   token embedding and the tied LM head, RMSNorm gains, all biases,
            KDA's `A_log` and `dt_bias`, MUDD's `static_bias`, the Delta gate,
            the depthwise convolution filters, and anything 0- or 1-dimensional

The embedding is deliberately on the AdamW side even though it is 2-D: its rows
are looked up, not applied, and only the rows for tokens in the batch have any
gradient at all, so orthogonalising it would mix updates across tokens that never
appeared.
"""

from __future__ import annotations

import torch

# Quintic Newton-Schulz coefficients, tuned so the iteration pushes singular
# values towards 1 fastest in the first few steps rather than converging exactly.
_NS_COEFFS = (3.4445, -4.7750, 2.0315)


@torch.no_grad()
def orthogonalise(matrix: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate the orthogonal factor of `matrix` by Newton-Schulz iteration.

    Runs in bfloat16: this only has to produce a direction, and the iteration is
    contractive, so the reduced precision changes the step by far less than the
    learning rate does.
    """
    a, b, c = _NS_COEFFS
    x = matrix.bfloat16()
    x = x / (x.norm() + 1e-7)
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.mT
    for _ in range(steps):
        gram = x @ x.mT
        x = a * x + (b * gram + c * gram @ gram) @ x
    if transposed:
        x = x.mT
    return x


def _is_conv_filter(name: str) -> bool:
    """KDA's short depthwise filters, `[C, 1, K]`.

    They are initialised as a delta -- `weight[:, 0, -1] = 1` -- so the layer
    starts by passing its input through unchanged.  Weight decay pulls that
    towards zero and attenuates Q, K and V for no reason the loss asked for, the
    same failure mode as decaying MUDD's identity selector.
    """
    return name.endswith(("q_conv.weight", "k_conv.weight", "v_conv.weight"))


def is_muon_matrix(name: str, param: torch.Tensor) -> bool:
    """True for the 2-D linear maps, false for everything else.

    `static_bias` is the only 2-D tensor that is not a map: it is MUDD's identity
    selector, one row per stream, and weight decay or orthogonalisation would
    both dissolve the identity initialisation the mechanism starts from.
    """
    if param.ndim != 2:
        return False
    if name.endswith("static_bias"):
        return False
    return not name.startswith(("model.embed_tokens", "lm_head"))


class Muon(torch.optim.Optimizer):
    """Muon on groups flagged `use_muon`, AdamW on the rest, in one step().

    Both halves share `step()` so there is a single optimizer to checkpoint and a
    single place where the schedule is applied; the groups keep separate learning
    rates because a Muon learning rate and an AdamW learning rate are not the same
    quantity even after the RMS matching.
    """

    def __init__(
        self,
        param_groups,
        lr: float = 2e-2,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adamw_lr: float = 3e-4,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        weight_decay: float = 0.1,
        update_rms: float = 0.2,
    ):
        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
            adamw_lr=adamw_lr, adamw_betas=adamw_betas, adamw_eps=adamw_eps,
            weight_decay=weight_decay, update_rms=update_rms,
        )
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            if group.get("use_muon"):
                self._step_muon(group)
            else:
                self._step_adamw(group)
        return loss

    def _step_muon(self, group) -> None:
        for param in group["params"]:
            if param.grad is None:
                continue
            state = self.state[param]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(param)
            buffer = state["momentum_buffer"]
            buffer.mul_(group["momentum"]).add_(param.grad)
            # Nesterov: step along where the momentum is heading, not where it has
            # been, which is what the reference implementation uses.
            direction = (
                param.grad.add(buffer, alpha=group["momentum"])
                if group["nesterov"] else buffer
            )
            update = orthogonalise(direction, group["ns_steps"]).to(param.dtype)
            rows, cols = param.shape
            update.mul_(group["update_rms"] * max(rows, cols) ** 0.5)
            if group["weight_decay"]:
                param.mul_(1.0 - group["lr"] * group["weight_decay"])
            param.add_(update, alpha=-group["lr"])

    def _step_adamw(self, group) -> None:
        beta1, beta2 = group["adamw_betas"]
        for param in group["params"]:
            if param.grad is None:
                continue
            state = self.state[param]
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(param)
                state["exp_avg_sq"] = torch.zeros_like(param)
            state["step"] += 1
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            exp_avg.mul_(beta1).add_(param.grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(param.grad, param.grad, value=1 - beta2)
            bias1 = 1 - beta1 ** state["step"]
            bias2 = 1 - beta2 ** state["step"]
            denominator = (exp_avg_sq / bias2).sqrt_().add_(group["adamw_eps"])
            if group["weight_decay"]:
                param.mul_(1.0 - group["adamw_lr"] * group["weight_decay"])
            param.addcdiv_(exp_avg, denominator, value=-group["adamw_lr"] / bias1)


def muon_param_groups(
    model, weight_decay: float = 0.1, muon_lr: float = 2e-2, adamw_lr: float = 3e-4,
) -> list[dict]:
    """Split a model into the Muon half and the AdamW half.

    The `no_decay` split inside the AdamW half is the same one AdamW alone uses:
    norms, biases and the KDA decay parameters must not be shrunk towards zero.
    """
    matrices, decay, no_decay = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if is_muon_matrix(name, param):
            matrices.append(param)
        elif param.ndim <= 1 or name.endswith("static_bias") or _is_conv_filter(name):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": matrices, "use_muon": True,
         "lr": muon_lr, "weight_decay": weight_decay},
        {"params": decay, "use_muon": False,
         "adamw_lr": adamw_lr, "weight_decay": weight_decay},
        {"params": no_decay, "use_muon": False,
         "adamw_lr": adamw_lr, "weight_decay": 0.0},
    ]


__all__ = ["Muon", "muon_param_groups", "orthogonalise", "is_muon_matrix"]
