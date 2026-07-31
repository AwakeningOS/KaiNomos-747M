"""Shared-LR Per-Head Muon used by KaiNomos-750M.

The grouping contract is intentionally name based.  A new parameter is not
silently sent to an optimizer merely because it happens to be two-dimensional:
every trainable parameter must match exactly one of the four frozen classes.
This turns architecture changes into an explicit optimizer review.
"""

from __future__ import annotations

import math
from typing import Any

import torch

_NS_COEFFS = (3.4445, -4.7750, 2.0315)
_PER_HEAD_KDA = (".attn.q_proj.weight", ".attn.k_proj.weight", ".attn.v_proj.weight")
_PER_HEAD_MLA_Q = ".attn.q_b_proj.weight"
_PER_HEAD_MLA_KV = ".attn.kv_b_proj.weight"

_FULL_MATRIX_SUFFIXES = (
    # SiTU-GLU (backbone and MTP).
    ".ffn.gate_proj.weight", ".ffn.up_proj.weight", ".ffn.down_proj.weight",
    # KDA maps other than the head-expanded Q/K/V projections.
    ".attn.f_a_proj.weight", ".attn.f_b_proj.weight",
    ".attn.beta_proj.weight", ".attn.output_gate.weight",
    ".attn.output_proj.weight",
    # MLA latent maps.  q_b/kv_b are handled by the per-head rules above.
    ".attn.q_a_proj.weight", ".attn.kv_a_proj.weight",
    # MTP input fusion.
    "mtp.fuse.weight",
)

_CONV_SUFFIXES = (".attn.q_conv.weight", ".attn.k_conv.weight", ".attn.v_conv.weight")


@torch.no_grad()
def orthogonalise(matrix: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate the polar factor with the Muon Newton--Schulz iteration."""
    if matrix.ndim != 2:
        raise ValueError(f"Muon expects a matrix, got shape {tuple(matrix.shape)}")
    a, b, c = _NS_COEFFS
    # BF16 is the reference Muon direction precision.  Float64 is kept for
    # small CPU reference tests, where BF16 would discard most useful digits.
    work_dtype = torch.float64 if matrix.dtype == torch.float64 else torch.bfloat16
    x = matrix.to(work_dtype)
    x = x / (x.norm() + 1e-7)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.mT
    for _ in range(steps):
        gram = x @ x.mT
        x = a * x + (b * gram + c * gram @ gram) @ x
    return x.mT if transposed else x


@torch.no_grad()
def orthogonalise_blocks(blocks: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Vectorised independent polar iteration for equal-shaped head blocks."""
    if blocks.ndim != 3:
        raise ValueError(f"expected [heads, rows, cols], got {tuple(blocks.shape)}")
    a, b, c = _NS_COEFFS
    work_dtype = torch.float64 if blocks.dtype == torch.float64 else torch.bfloat16
    x = blocks.to(work_dtype)
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.mT
    for _ in range(steps):
        gram = x @ x.mT
        x = a * x + (b * gram + c * gram @ gram) @ x
    return x.mT if transposed else x


def validate_shared_lr(
    lr: float,
    *,
    muon_lr: float | None = None,
    adamw_lr: float | None = None,
) -> float:
    """Validate legacy CLI rates and return the one shared learning rate.

    Call this at the CLI/config boundary if compatibility flags still exist.
    A mismatch is rejected instead of recreating the old 0.02/0.0003 regime.
    """
    shared = float(lr)
    for label, candidate in (("muon_lr", muon_lr), ("adamw_lr", adamw_lr)):
        if candidate is not None and float(candidate) != shared:
            raise ValueError(
                "RMS-matched Muon and AdamW must share one LR "
                f"(lr={shared}, {label}={candidate})"
            )
    return shared


def _is_conv_filter(name: str) -> bool:
    return name.endswith(_CONV_SUFFIXES)


def _is_norm(name: str) -> bool:
    return name.endswith(".weight") and "norm" in name.rsplit(".weight", 1)[0].split(".")[-1]


def _classify_parameter(name: str, param: torch.Tensor) -> str:
    """Return one frozen optimizer class or raise for an unknown parameter."""
    matches: list[str] = []

    if name.endswith(_PER_HEAD_KDA + (_PER_HEAD_MLA_Q, _PER_HEAD_MLA_KV)):
        matches.append("muon_per_head")
    if name.endswith(_FULL_MATRIX_SUFFIXES):
        matches.append("muon_full")

    # The tied head is omitted by named_parameters(), but this also handles an
    # explicitly listed/shared parameter and makes the policy auditable.
    if name in {"model.embed_tokens.weight", "embed_tokens.weight", "lm_head.weight"}:
        matches.append("adamw_decay")

    no_decay = (
        param.ndim <= 1
        or name.endswith((".bias", ".A_log", ".dt_bias", ".query"))
        or _is_conv_filter(name)
        or _is_norm(name)
    )
    if no_decay:
        matches.append("adamw_no_decay")

    if len(matches) != 1:
        detail = "unclassified" if not matches else f"matched {matches}"
        raise ValueError(
            f"optimizer classification for {name!r} must be unique: {detail}; "
            f"shape={tuple(param.shape)}"
        )
    return matches[0]


def is_muon_matrix(name: str, param: torch.Tensor) -> bool:
    """Compatibility query backed by the explicit classification table."""
    return _classify_parameter(name, param).startswith("muon_")


def _row_block_metadata(name: str, param: torch.Tensor, config) -> dict[str, int | str]:
    if name.endswith(_PER_HEAD_KDA):
        num_blocks = config.kda.num_heads
        block_rows = config.kda.head_dim
    elif name.endswith(_PER_HEAD_MLA_Q):
        num_blocks = config.mla.num_heads
        block_rows = config.mla.q_head_dim
    elif name.endswith(_PER_HEAD_MLA_KV):
        num_blocks = config.mla.num_heads
        block_rows = config.mla.qk_nope_head_dim + config.mla.v_head_dim
    else:  # guarded by _classify_parameter
        raise ValueError(f"no per-head layout rule for {name!r}")
    expected_rows = num_blocks * block_rows
    if param.ndim != 2 or param.shape[0] != expected_rows:
        raise ValueError(
            f"invalid row-block layout for {name!r}: expected "
            f"[{num_blocks}*{block_rows}, cols], got {tuple(param.shape)}"
        )
    return {
        "muon_layout": "row_blocks",
        "num_blocks": num_blocks,
        "block_rows": block_rows,
    }


def muon_param_groups(
    model,
    weight_decay: float = 0.1,
    muon_lr: float | None = None,
    adamw_lr: float | None = None,
    *,
    lr: float | None = None,
) -> list[dict[str, Any]]:
    """Build explicit, complete and non-overlapping optimizer groups.

    ``muon_lr``/``adamw_lr`` are accepted only as a migration boundary.  If
    supplied they must equal ``lr`` (or each other when ``lr`` is omitted).
    """
    if lr is None:
        supplied = [float(x) for x in (muon_lr, adamw_lr) if x is not None]
        lr = supplied[0] if supplied else 3e-4
    shared_lr = validate_shared_lr(lr, muon_lr=muon_lr, adamw_lr=adamw_lr)
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("explicit Muon grouping requires model.config")

    groups: list[dict[str, Any]] = []
    full_params: list[torch.Tensor] = []
    full_names: list[str] = []
    decay_params: list[torch.Tensor] = []
    decay_names: list[str] = []
    no_decay_params: list[torch.Tensor] = []
    no_decay_names: list[str] = []
    seen: set[int] = set()
    trainable: dict[int, str] = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        identity = id(param)
        if identity in trainable:
            raise ValueError(
                f"trainable parameter is listed twice: {trainable[identity]!r} and {name!r}"
            )
        trainable[identity] = name
        category = _classify_parameter(name, param)
        if identity in seen:
            raise ValueError(f"parameter {name!r} was assigned more than once")
        seen.add(identity)

        if category == "muon_per_head":
            metadata = _row_block_metadata(name, param, config)
            groups.append({
                "params": [param], "param_names": [name],
                "group_name": "muon_per_head", "use_muon": True,
                "lr": shared_lr, "weight_decay": weight_decay, **metadata,
            })
        elif category == "muon_full":
            if param.ndim != 2:
                raise ValueError(f"full-matrix Muon requires 2-D {name!r}")
            full_params.append(param)
            full_names.append(name)
        elif category == "adamw_decay":
            decay_params.append(param)
            decay_names.append(name)
        else:
            no_decay_params.append(param)
            no_decay_names.append(name)

    if set(trainable) != seen:
        missing = [trainable[key] for key in set(trainable) - seen]
        raise ValueError(f"optimizer parameter coverage is incomplete: {missing}")

    if full_params:
        groups.append({
            "params": full_params, "param_names": full_names,
            "group_name": "muon_full", "use_muon": True,
            "muon_layout": "full_matrix", "lr": shared_lr,
            "weight_decay": weight_decay,
        })
    if decay_params:
        groups.append({
            "params": decay_params, "param_names": decay_names,
            "group_name": "adamw_decay", "use_muon": False,
            "lr": shared_lr, "weight_decay": weight_decay,
        })
    if no_decay_params:
        groups.append({
            "params": no_decay_params, "param_names": no_decay_names,
            "group_name": "adamw_no_decay", "use_muon": False,
            "lr": shared_lr, "weight_decay": 0.0,
        })
    return groups


def _rms(value: torch.Tensor) -> float:
    return float(value.float().square().mean().sqrt().detach())


class Muon(torch.optim.Optimizer):
    """Per-Head/full-matrix Muon plus AdamW under one shared LR."""

    def __init__(
        self,
        param_groups,
        lr: float = 3e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        weight_decay: float = 0.1,
        update_rms: float = 0.2,
    ):
        defaults = {
            "lr": float(lr), "momentum": momentum, "nesterov": nesterov,
            "ns_steps": ns_steps, "adamw_betas": adamw_betas,
            "adamw_eps": adamw_eps, "weight_decay": weight_decay,
            "update_rms": update_rms,
        }
        super().__init__(param_groups, defaults)
        self._validate_group_lrs()
        self.last_step_stats: list[dict[str, Any]] = []

    def _validate_group_lrs(self) -> float:
        rates = {float(group["lr"]) for group in self.param_groups}
        if len(rates) != 1:
            raise ValueError(f"RMS-matched Muon and AdamW must share one LR: {rates}")
        return next(iter(rates))

    @torch.no_grad()
    def step(self, closure=None, *, collect_stats: bool = False):
        loss = closure() if closure is not None else None
        self._validate_group_lrs()
        self.last_step_stats = []
        for group in self.param_groups:
            if group.get("use_muon"):
                self._step_muon(group, collect_stats)
            else:
                self._step_adamw(group, collect_stats)
        return loss

    def _muon_update(
        self, direction: torch.Tensor, group, collect_stats: bool
    ) -> tuple[torch.Tensor, list[float]]:
        layout = group.get("muon_layout")
        if layout == "row_blocks":
            count = int(group["num_blocks"])
            rows = int(group["block_rows"])
            blocks = direction.view(count, rows, direction.shape[1])
            updates = orthogonalise_blocks(blocks, group["ns_steps"]).to(
                direction.dtype
            )
            updates.mul_(group["update_rms"] * math.sqrt(max(blocks.shape[-2:])))
            block_rms = []
            if collect_stats:
                block_rms = (
                    updates.float().square().mean(dim=(-2, -1)).sqrt()
                    * float(group["lr"])
                ).cpu().tolist()
            return updates.view_as(direction), block_rms
        if layout != "full_matrix":
            raise ValueError(f"unknown Muon layout {layout!r}")
        update = orthogonalise(direction, group["ns_steps"]).to(direction.dtype)
        update.mul_(group["update_rms"] * math.sqrt(max(direction.shape)))
        return update, []

    def _step_muon(self, group, collect_stats: bool) -> None:
        names = group.get("param_names", [f"param_{i}" for i in range(len(group["params"]))])
        for name, param in zip(names, group["params"], strict=True):
            if param.grad is None:
                continue
            state = self.state[param]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(param)
            momentum = state["momentum_buffer"]
            momentum.mul_(group["momentum"]).add_(param.grad)
            direction = (
                param.grad.add(momentum, alpha=group["momentum"])
                if group["nesterov"] else momentum
            )
            update, head_update_rms = self._muon_update(
                direction, group, collect_stats
            )
            lr = float(group["lr"])
            if collect_stats:
                update_rms = _rms(update) * lr
                weight_rms = _rms(param)
                self.last_step_stats.append({
                    "group": group.get("group_name", "muon"), "parameter": name,
                    "shape": list(param.shape), "grad_rms": _rms(param.grad),
                    "update_rms": update_rms, "weight_rms": weight_rms,
                    "update_to_weight": update_rms / max(weight_rms, 1e-30),
                    "momentum_rms": _rms(momentum),
                    "head_update_rms": head_update_rms,
                    "head_update_rms_cv": (
                        float(torch.tensor(head_update_rms).std(unbiased=False)
                              / torch.tensor(head_update_rms).mean().clamp_min(1e-30))
                        if head_update_rms else None
                    ),
                })
            if group["weight_decay"]:
                param.mul_(1.0 - lr * group["weight_decay"])
            param.add_(update, alpha=-lr)

    def _step_adamw(self, group, collect_stats: bool) -> None:
        beta1, beta2 = group["adamw_betas"]
        names = group.get("param_names", [f"param_{i}" for i in range(len(group["params"]))])
        for name, param in zip(names, group["params"], strict=True):
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
            denominator = (exp_avg_sq / bias2).sqrt().add_(group["adamw_eps"])
            update = exp_avg / denominator / bias1
            lr = float(group["lr"])
            if collect_stats:
                update_rms = _rms(update) * lr
                weight_rms = _rms(param)
                self.last_step_stats.append({
                    "group": group.get("group_name", "adamw"), "parameter": name,
                    "shape": list(param.shape), "grad_rms": _rms(param.grad),
                    "update_rms": update_rms, "weight_rms": weight_rms,
                    "update_to_weight": update_rms / max(weight_rms, 1e-30),
                    "momentum_rms": _rms(exp_avg), "head_update_rms": [],
                    "head_update_rms_cv": None,
                })
            if group["weight_decay"]:
                param.mul_(1.0 - lr * group["weight_decay"])
            param.add_(update, alpha=-lr)


__all__ = [
    "Muon", "is_muon_matrix", "muon_param_groups", "orthogonalise",
    "orthogonalise_blocks",
    "validate_shared_lr",
]
