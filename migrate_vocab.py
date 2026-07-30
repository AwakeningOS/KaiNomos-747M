"""Carry an 82M checkpoint into KaiNomos-110M across a tokenizer change.

Tensors are copied when name and role match. Where the 110M tensor is a *widened*
version of the 82M one along the nested-FFN axis, the 82M weight is copied into
the leading slice instead of being discarded: the FFN is a nested supernet whose
prefix is the old network, so `W_110[:2432]` and `W_82` denote the same channels.
Requiring exact shape equality here threw away 45,416,448 trained parameters --
every FFN matrix in twelve layers plus the low-rank decay maps of nine KDA
layers -- and left a model described as "migrated from 82M" whose feed-forward
was entirely random.

The vocabulary is the hard case, because
the old 16,384-piece English BPE and the new 32,768-piece Unigram do not share an
identity: token id 500 means nothing in common between them.

Three rules, in order of how much they preserve:

* a piece that exists in both vocabularies keeps its old embedding row
* a new piece that the old tokenizer can still express keeps the **mean of the
  rows of its old pieces** -- a reasonable starting point, since the new piece
  denotes the same string
* anything else is initialised normally

The LM head is tied to the new embedding, so the old head is not carried across
at all: it was a separate 16,384-row tensor over a vocabulary that no longer
exists, and there is nothing in it to reuse that the embedding does not already
hold.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import torch

_LAYER = re.compile(r"^model\.layers\.(\d+)\.(.*)$")

# shapes depend on the layer index, or the mechanism is new in this model
_NEW_MECHANISM = ("mudd.", "delta_attn.", "delta_ffn.")
# These are 110M-only modules. JointRoute itself is not modified; its existing
# KaiNomos initialisation and implementation are simply left untouched.
_NEW_TOP_LEVEL = ("model.controller.", "mtp.")

# Tensors whose 110M shape is the 82M shape widened along the nested-FFN axis.
# The nested FFN uses a prefix of its channels, so channel i means the same thing
# in both models and the old weight belongs in the leading slice.  Nothing else
# may be prefix-copied: for any other tensor a shape change means the dimension
# was reinterpreted, not extended.
_NESTED_PREFIX = (
    "ffn.gate_proj.weight",
    "ffn.up_proj.weight",
    "ffn.down_proj.weight",
)


def prefix_copy(dest: torch.Tensor, source: torch.Tensor) -> bool:
    """Copy `source` into the leading slice of `dest`. False if it cannot fit."""
    if dest.ndim != source.ndim:
        return False
    if any(s > d for s, d in zip(source.shape, dest.shape)):
        return False
    dest[tuple(slice(0, size) for size in source.shape)].copy_(source)
    return True


def build_vocab_embedding(
    old_embedding: torch.Tensor,
    old_tokenizer: "object",
    new_tokenizer_path: Path,
    initializer_range: float = 0.02,
) -> tuple[torch.Tensor, dict]:
    """A new embedding table seeded from the old one wherever that is meaningful."""
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(new_tokenizer_path))
    hidden = old_embedding.shape[1]
    new_size = sp.get_piece_size()
    table = torch.empty(new_size, hidden)
    torch.nn.init.normal_(table, mean=0.0, std=initializer_range)

    old_vocab = old_tokenizer.get_vocab()
    stats = Counter()
    with torch.no_grad():
        for new_id in range(new_size):
            piece = sp.id_to_piece(new_id)
            # SentencePiece marks a leading space with U+2581; the old ByteLevel
            # BPE marks it with U+0120, so the same string has two spellings
            text = piece.replace("▁", " ")
            if piece in old_vocab:
                table[new_id] = old_embedding[old_vocab[piece]]
                stats["exact_piece"] += 1
                continue
            if not text or piece.startswith("<"):
                stats["special_or_empty"] += 1
                continue
            encoded = old_tokenizer.encode(text)
            pieces = encoded.ids
            round_trip = old_tokenizer.decode(pieces, skip_special_tokens=False)
            if pieces and round_trip == text:
                table[new_id] = old_embedding[torch.tensor(pieces)].mean(0)
                stats["decomposed_mean"] += 1
            else:
                stats["fresh"] += 1
    stats["total"] = new_size
    return table, dict(stats)


def source_layer_map(source_pattern: list[str], target_pattern: list[str]) -> dict[int, int]:
    """Map `target index -> source index` for depths with the same operator role.

    Added target depths are deliberately absent: they must remain freshly
    initialised rather than being clones of earlier layers.

    Beyond index alignment there is one structural correspondence worth keeping.
    The 82M stack is three `KDA,KDA,KDA,MLA` blocks *plus a final global-attention
    layer*, so its last layer is an MLA sitting at index 12 where the 110M stack
    has a KDA.  Index alignment therefore declines it and a fully trained
    attention layer is thrown away, even though the 110M stack ends in an MLA
    doing exactly that job.  When both stacks end in an unmapped layer of the same
    kind, the final layer keeps its role at the end rather than its index.
    """
    mapping = {
        index: index
        for index in range(min(len(source_pattern), len(target_pattern)))
        if source_pattern[index] == target_pattern[index]
    }
    if not source_pattern or not target_pattern:
        return mapping
    last_source = len(source_pattern) - 1
    last_target = len(target_pattern) - 1
    if (
        last_target not in mapping
        and last_source not in mapping.values()
        and source_pattern[last_source] == target_pattern[last_target]
    ):
        mapping[last_target] = last_source
    return mapping


def migrate(source_state: dict, source_pattern: list[str], config,
            new_tokenizer: Path | None = None, old_tokenizer=None,
            new_embedding: torch.Tensor | None = None,
            booster_scale: float = 0.1):
    from model import KaiNomosForCausalLM

    model = KaiNomosForCausalLM(config)
    target = dict(model.state_dict())
    layer_map = source_layer_map(source_pattern, list(config.layer_pattern))
    # Every target depth that received nothing, not just the depths past the end
    # of the source stack.  Reporting `range(len(source), num_layers)` understated
    # this: with the 82M stack ending in an MLA, layer 12 was also entirely fresh
    # and did not appear.
    fresh_layers = [i for i in range(config.num_hidden_layers) if i not in layer_map]
    stats = {
        "copied": 0,
        "prefix_copied": [],
        "shape_mismatch": [],
        "new": [],
        "layer_map": dict(layer_map),
        "fresh_layers": fresh_layers,
        "booster_scale": booster_scale,
        "damped": [],
    }

    with torch.no_grad():
        for name, dest in target.items():
            if name.startswith(("model.embed_tokens", "lm_head")):
                continue                      # handled by the vocabulary rules
            if name.startswith(_NEW_TOP_LEVEL):
                stats["new"].append(name)
                continue
            match = _LAYER.match(name)
            if match is None:
                if name in source_state and source_state[name].shape == dest.shape:
                    dest.copy_(source_state[name])
                    stats["copied"] += 1
                else:
                    stats["new"].append(name)
                continue

            target_index, suffix = int(match.group(1)), match.group(2)
            if suffix.startswith(_NEW_MECHANISM):
                stats["new"].append(name)
                continue
            if target_index not in layer_map:
                stats["new"].append(name)
                continue
            source_name = f"model.layers.{layer_map[target_index]}.{suffix}"
            value = source_state.get(source_name)
            if value is None:
                stats["new"].append(name)
                continue
            if value.shape == dest.shape:
                dest.copy_(value)
                stats["copied"] += 1
            elif suffix in _NESTED_PREFIX and prefix_copy(dest, value):
                stats["prefix_copied"].append({
                    "name": name,
                    "source_shape": list(value.shape),
                    "target_shape": list(dest.shape),
                })
            else:
                stats["shape_mismatch"].append({
                    "name": name,
                    "source_shape": list(value.shape),
                    "target_shape": list(dest.shape),
                })

        # Weak-residual start for everything the 82M model did not supply.  A
        # freshly initialised layer inserted into a trained stack otherwise adds
        # full-scale noise to a residual stream that already carries a useful
        # signal, and the first thing training has to do is undo the damage.
        # Scaling the two projections that *write* to the stream down by
        # `booster_scale` makes each new layer start close to a no-op and grow
        # into the stack.  `--booster-scale` used to be parsed and then ignored.
        if not 0.0 < booster_scale <= 1.0:
            raise ValueError("booster_scale must lie in (0, 1]")
        source_widths = {}
        for entry in stats["prefix_copied"]:
            match = _LAYER.match(entry["name"])
            if match and match.group(2) == "ffn.down_proj.weight":
                source_widths[int(match.group(1))] = entry["source_shape"][1]
        for index in range(config.num_hidden_layers):
            prefix = f"model.layers.{index}."
            if index in fresh_layers:
                for suffix in ("attn.output_proj.weight", "ffn.down_proj.weight"):
                    tensor = target.get(prefix + suffix)
                    if tensor is not None:
                        tensor.mul_(booster_scale)
                        stats["damped"].append(prefix + suffix)
            elif index in source_widths:
                # A migrated layer keeps its trained channels at full strength;
                # only the reinvestment band the 82M model never had is damped.
                width = source_widths[index]
                name = prefix + "ffn.down_proj.weight"
                target[name][:, width:].mul_(booster_scale)
                stats["damped"].append(f"{name}[:, {width}:]")

        # vocabulary
        old_embedding = source_state["model.embed_tokens.weight"]
        if new_embedding is None:
            if new_tokenizer is None or old_tokenizer is None:
                raise ValueError(
                    "new_tokenizer and old_tokenizer are required unless "
                    "new_embedding is supplied"
                )
            table, vocab_stats = build_vocab_embedding(
                old_embedding, old_tokenizer, new_tokenizer,
                config.initializer_range,
            )
        else:
            table = new_embedding
            vocab_stats = {"provided": table.shape[0], "total": table.shape[0]}
        expected = target["model.embed_tokens.weight"].shape
        if table.shape != expected:
            raise ValueError(
                f"new embedding shape {tuple(table.shape)} does not match "
                f"KaiNomos target {tuple(expected)}"
            )
        target["model.embed_tokens.weight"].copy_(table)
        target["lm_head.weight"].copy_(table)
        stats["vocabulary"] = vocab_stats

        model.load_state_dict(target)

    if config.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight
    if model.lm_head.weight is not model.model.embed_tokens.weight:
        raise RuntimeError("LM head must remain tied to the KaiNomos embedding")
    return model, stats


def main() -> int:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, help="82M checkpoint")
    ap.add_argument("--old-tokenizer-dir", required=True)
    ap.add_argument("--new-tokenizer", default="data/tokenizer/kainomos.model")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--booster-scale", type=float, default=0.1,
        help="output-projection scale for layers and FFN bands the 82M model did "
             "not supply, so they start as near no-ops on the residual stream",
    )
    args = ap.parse_args()

    from tokenizers import Tokenizer

    from config import KaiNomosConfig

    blob = torch.load(args.source, map_location="cpu", weights_only=False)
    state = {k: v for k, v in blob.get("model", blob).items() if "controller" not in k}
    raw_config = blob.get("model_config") or blob.get("config") or {}
    source_pattern = list(raw_config.get("layer_pattern", []))
    if not source_pattern:
        n = 1 + max(int(m.group(1)) for m in map(_LAYER.match, state) if m)
        source_pattern = ["MLA" if (i + 1) % 4 == 0 else "KDA" for i in range(n)]

    import sentencepiece as spm

    config = KaiNomosConfig()
    config.vocab_size = spm.SentencePieceProcessor(
        model_file=args.new_tokenizer).get_piece_size()
    config.tie_word_embeddings = True

    old_tokenizer = Tokenizer.from_file(
        str(Path(args.old_tokenizer_dir) / "tokenizer.json"))
    model, stats = migrate(state, source_pattern, config,
                           Path(args.new_tokenizer), old_tokenizer,
                           booster_scale=args.booster_scale)

    report = model.parameter_report()
    torch.save({"model": model.state_dict(), "config": config.to_dict(),
                "migration": stats, "parameters": report,
                "source": args.source}, args.out)

    recovered = sum(
        int(torch.Size(entry["source_shape"]).numel())
        for entry in stats["prefix_copied"]
    )
    print(f"source layers   : {len(source_pattern)} -> {config.num_hidden_layers}")
    print(f"layer map       : {stats['layer_map']}")
    print(f"tensors copied  : {stats['copied']}")
    print(f"prefix copied   : {len(stats['prefix_copied'])} "
          f"({recovered:,} parameters recovered)")
    print(f"shape mismatch  : {len(stats['shape_mismatch'])}")
    print(f"fresh layers    : {stats['fresh_layers']}")
    print(f"damped (x{args.booster_scale}) : {len(stats['damped'])} tensors")
    print(f"new tensors     : {len(stats['new'])}")
    print(f"vocabulary      : {json.dumps(stats['vocabulary'])}")
    print(f"total params    : {report['total_params']:,}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
