#!/usr/bin/env python
"""Migrate a trained 82M checkpoint into the 110M model.

Three things grow: the FFN (2432 -> 2816), the depth (13 -> 16 layers), and the
set of mechanisms (MUDD, Delta, MTP).  Each is handled so that the migrated
model *starts* as the 82M model it came from and has to earn any change:

* widened FFN channels are freshly initialised with their output projection
  scaled down, so the new band writes almost nothing at first
* new layers are cloned from existing layers of the same kind -- a KDA layer is
  never seeded from an MLA layer, which would be copying weights between two
  different operators -- and their output projections are likewise scaled down,
  so an added layer begins as a near-identity
* MUDD starts as the identity on the newest depth state, Delta gates start at
  zero, JointRoute starts on the Base policy, MTP is new

Usage:
    python migrate_82m_to_110m.py --source <82M checkpoint> --out init_110m.pt
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

from config import K3MiniPlusPlusPlusConfig
from model import K3MiniPlusPlusPlusForCausalLM

# axis of each FFN tensor that carries the intermediate dimension
_FFN_AXIS = {"ffn.gate_proj.weight": 0, "ffn.up_proj.weight": 0, "ffn.down_proj.weight": 1}
# tensors that are scaled down so a cloned layer starts as a weak correction
_DAMPED_SUFFIXES = ("attn.output_proj.weight", "ffn.down_proj.weight")

_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.*)$")

# Mechanisms that did not exist in the source, or whose shape depends on the
# layer index and so cannot be cloned across depths.  MUDD's coefficient table is
# sized by how many depth states a layer can see, so layer 4's table simply does
# not fit layer 12's.  All of these carry their own identity initialisation.
_NEW_MECHANISM = ("mudd.", "delta_attn.", "delta_ffn.")


def _ffn_axis(name: str) -> int | None:
    for suffix, axis in _FFN_AXIS.items():
        if name.endswith(suffix):
            return axis
    return None


def _copy_into(dest: torch.Tensor, src: torch.Tensor, name: str) -> str:
    """Copy `src` into `dest`, widening along the FFN axis when needed."""
    if dest.shape == src.shape:
        dest.copy_(src)
        return "whole"
    axis = _ffn_axis(name)
    if axis is None:
        raise ValueError(f"shape mismatch on {name}: {tuple(src.shape)} -> {tuple(dest.shape)}")
    index = [slice(None)] * dest.dim()
    index[axis] = slice(0, src.shape[axis])
    dest[tuple(index)].copy_(src)
    return "widened"


def build_source_layer_map(
    source_pattern: list[str], target_pattern: list[str]
) -> dict[int, int]:
    """Which source layer seeds each target layer.

    Existing depths map one-to-one.  Each new depth is cloned from an existing
    layer *of the same kind*, cycling through the later ones, because a KDA and
    an MLA layer do not share a parameter shape or a meaning.
    """
    mapping: dict[int, int] = {}
    by_kind: dict[str, list[int]] = {}
    for index, kind in enumerate(source_pattern):
        by_kind.setdefault(kind, []).append(index)

    cursor = {kind: 0 for kind in by_kind}
    for target_index, kind in enumerate(target_pattern):
        if target_index < len(source_pattern) and source_pattern[target_index] == kind:
            mapping[target_index] = target_index
            continue
        pool = by_kind.get(kind)
        if not pool:
            raise ValueError(f"no source layer of kind {kind} to clone")
        # prefer the later half: those are closest to the depths being added
        later = pool[len(pool) // 2:] or pool
        mapping[target_index] = later[cursor[kind] % len(later)]
        cursor[kind] += 1
    return mapping


def migrate(
    source_state: dict[str, torch.Tensor],
    source_pattern: list[str],
    config: K3MiniPlusPlusPlusConfig,
    clone_scale: float = 0.1,
    widened_scale: float = 0.1,
) -> tuple[K3MiniPlusPlusPlusForCausalLM, dict]:
    model = K3MiniPlusPlusPlusForCausalLM(config)
    target = dict(model.state_dict())
    layer_map = build_source_layer_map(source_pattern, list(config.layer_pattern))
    cloned = {t for t, s in layer_map.items() if t >= len(source_pattern) or t != s}

    stats = {"whole": 0, "widened": 0, "cloned_layers": sorted(cloned),
             "skipped": [], "new": []}

    with torch.no_grad():
        for name, dest in target.items():
            match = _LAYER_RE.match(name)
            if match is None:
                if name in source_state and source_state[name].shape == dest.shape:
                    dest.copy_(source_state[name])
                    stats["whole"] += 1
                else:
                    stats["new"].append(name)
                continue

            target_index, suffix = int(match.group(1)), match.group(2)
            if suffix.startswith(_NEW_MECHANISM):
                stats["new"].append(name)
                continue
            source_index = layer_map[target_index]
            source_name = f"model.layers.{source_index}.{suffix}"
            if source_name not in source_state:
                stats["new"].append(name)
                continue
            kind = _copy_into(dest, source_state[source_name], name)
            stats[kind] += 1

            if target_index in cloned and suffix.endswith(_DAMPED_SUFFIXES):
                dest *= clone_scale

        model.load_state_dict(target)

        # freshly widened FFN channels write weakly until they earn their FLOPs
        old_width = None
        for name, value in source_state.items():
            if name.endswith("ffn.down_proj.weight"):
                old_width = value.shape[1]
                break
        if old_width is not None and old_width < config.ffn_intermediate_size:
            for layer in model.model.layers:
                layer.ffn.down_proj.weight[:, old_width:] *= widened_scale
            stats["widened_from"] = old_width

    # the new mechanisms keep their own identity-preserving initialisation
    model.model.controller.reset_to_fixed_policy()
    for layer in model.model.layers:
        if layer.mudd is not None:
            layer.mudd.reset_to_identity()
        assert layer.delta_attn.gate.item() == 0.0
        assert layer.delta_ffn.gate.item() == 0.0

    return model, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, help="82M checkpoint (.pt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--clone-scale", type=float, default=0.1)
    ap.add_argument("--widened-scale", type=float, default=0.1)
    args = ap.parse_args()

    blob = torch.load(args.source, map_location="cpu", weights_only=False)
    state = blob.get("model", blob)
    state = {k: v for k, v in state.items() if "controller" not in k}
    source_cfg = blob.get("model_config") or blob.get("config") or {}
    source_pattern = list(source_cfg.get("layer_pattern", []))
    if not source_pattern:
        n = 1 + max(int(m.group(1)) for m in map(_LAYER_RE.match, state) if m)
        source_pattern = [("MLA" if (i + 1) % 4 == 0 else "KDA") for i in range(n)]

    config = K3MiniPlusPlusPlusConfig()
    model, stats = migrate(state, source_pattern, config,
                           args.clone_scale, args.widened_scale)

    report = model.parameter_report()
    out = Path(args.out)
    torch.save({"model": model.state_dict(), "config": config.to_dict(),
                "migration": stats, "parameters": report, "source": args.source}, out)
    print(f"source layers : {len(source_pattern)} -> {config.num_hidden_layers}")
    print(f"copied whole  : {stats['whole']}")
    print(f"widened       : {stats['widened']}  (from {stats.get('widened_from')})")
    print(f"cloned layers : {stats['cloned_layers']}")
    print(f"new tensors   : {len(stats['new'])}")
    print(f"total params  : {report['total_params']:,}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
