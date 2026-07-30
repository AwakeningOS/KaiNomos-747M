"""KaiNomos-747M.

Per-layer order (spec section 6), identical in every one of the 16 layers:

    1. Delta Block depth mixing before attention
    2. MUDD builds the attention input streams from every visible depth state
    3. KDA or MLA
    4. residual add onto the *main* stream h
    5. Delta Block depth mixing again before the FFN
    6. dense FFN
    7. residual add onto h, then h joins the depth bank
    8. at a block boundary, the block's delta is banked

The mixed inputs `h_attn_input` / `h_ffn_input` feed the sublayers but are never
themselves the residual base: sublayer outputs always land on `h`, so the update
is always `h = old_h + sublayer_output`.  Writing `h = h_attn_input + attn_out`
instead would add the mixed delta into the stream permanently and double-count it
on the next hop, which is a different mechanism from the published one.

Depth mixing is Delta Block (arXiv:2605.18855): the sources are the embedding,
the completed block deltas, and the delta the open block has made so far -- never
the accumulated hidden state.

No layer is ever skipped.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import KaiNomosConfig
from delta_block import DeltaBank, DeltaRouter
from kda import KDAttention
from layers import RMSNorm, SiTUMLP
from mla import GatedMLA as MLAttention
from mtp import MTPHead, mtp_loss, mtp_slices
from mudd_qkv import MuDDQKV
from segments import mask_targets_at_boundaries


@dataclass
class KaiNomosOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    ntp_loss: torch.Tensor | None = None
    mtp_logits: torch.Tensor | None = None
    mtp_loss: torch.Tensor | None = None


class DecoderLayer(nn.Module):
    def __init__(self, config: KaiNomosConfig, index: int):
        super().__init__()
        self.config = config
        self.index = index
        self.kind = config.layer_pattern[index]
        eps = config.rms_norm_eps
        h = config.hidden_size

        # depth sources visible to this layer: embedding + every earlier output
        self.num_sources = index + 1
        self.streams = config.mudd.streams_for(self.kind)
        self.mudd = MuDDQKV(h, self.num_sources, config.mudd.hidden, eps,
                            self.streams) if config.mudd.enabled else None

        self.attn = KDAttention(config) if self.kind == "KDA" else MLAttention(config)
        # The single pre-attention norm, carried over from the 82M model.  MUDD
        # mixes unnormalised depth states and this normalises the result, so at
        # identity initialisation the attention input is exactly `attn_norm(h)` --
        # the same tensor the 82M layer saw, with the same migrated weight.
        self.attn_norm = RMSNorm(h, eps)
        self.ffn_norm = RMSNorm(h, eps)
        self.ffn = SiTUMLP(h, config.ffn_intermediate_size)

        # One router per sublayer position, each with its own norm and query:
        # they ask different questions of the same sources.
        self.delta_attn = DeltaRouter(h, eps)
        self.delta_ffn = DeltaRouter(h, eps)


class KaiNomosModel(nn.Module):
    def __init__(self, config: KaiNomosConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.apply(self._init_weights)
        for layer in self.layers:
            if layer.mudd is not None:
                layer.mudd.reset_to_identity()

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        respect_documents: bool = True,
    ):
        from segments import cu_seqlens, document_starts, segment_ids

        cfg = self.config
        # Document boundaries, derived from the separator in the tokens.  Every
        # mechanism with a time axis has to be told about them: MLA must not attend
        # across one, and KDA must not carry its recurrent state or its short
        # convolution window across one.
        segments = offsets = None
        if respect_documents and input_ids.shape[1] > 1:
            segments = segment_ids(input_ids, cfg.eod_token_id)
            offsets = cu_seqlens(document_starts(input_ids, cfg.eod_token_id))
        h = self.embed_tokens(input_ids)
        depth_states = [h]
        bank = DeltaBank(cfg.delta.granularity)
        bank.start(h)      # the embedding is the first source, permanently

        for index, layer in enumerate(self.layers):
            # 1. Delta Block depth mixing before attention. `h_attn_input` feeds the
            # sublayer; the residual stream itself is untouched here.
            h_attn_input = layer.delta_attn(h, bank.sources(h))

            # 2. MUDD over the depth states this layer can see, then the layer's
            # single pre-attention norm.  KDA mixes q/k/v; MLA mixes q/kv, since
            # its K and V come from one latent.
            sources = depth_states[:-1] + [h_attn_input]
            if layer.mudd is not None:
                mixed = layer.mudd(sources)
                streams = {n: layer.attn_norm(t) for n, t in mixed.items()}
            else:
                normed = layer.attn_norm(h_attn_input)
                streams = {name: normed for name in layer.streams}
            if layer.kind == "KDA":
                q_in, k_in, v_in = streams["q"], streams["k"], streams["v"]
            else:
                q_in = streams["q"]
                k_in, v_in = streams["kv"], None

            # 3. attention; 4. residual onto the main stream, never onto the input
            kwargs = {"q_input": q_in, "k_input": k_in, "v_input": v_in,
                      "segments": segments}
            if layer.kind == "KDA":
                kwargs["seq_offsets"] = offsets
            attn_out, _ = layer.attn(**kwargs)
            bank.record_sublayer(attn_out)
            h = h + attn_out

            # 5. depth mixing again before the FFN, from the updated stream: the
            # partial delta has grown by the attention output.
            h_ffn_input = layer.delta_ffn(h, bank.sources(h))

            # 6-7. dense FFN, residual onto h
            ffn_out = layer.ffn(layer.ffn_norm(h_ffn_input))
            bank.record_sublayer(ffn_out)
            h = h + ffn_out
            depth_states.append(h)

            # 8. bank the block delta at a block boundary
            if (index + 1) % cfg.block_size == 0:
                bank.close_block(h)

        hidden = self.final_norm(h)
        # `segments` always travels back: the MTP head and the loss both need the
        # same boundaries the body just used, and recomputing them there would let
        # the two drift apart.
        stats = {"segments": segments, "seq_offsets": offsets}
        return hidden, stats


class KaiNomosForCausalLM(nn.Module):
    def __init__(self, config: KaiNomosConfig):
        super().__init__()
        self.config = config
        self.model = KaiNomosModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.mtp = MTPHead(config) if config.mtp.enabled else None

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        use_mtp: bool = True,
    ) -> KaiNomosOutput:
        hidden, stats = self.model(input_ids)
        logits = self.lm_head(hidden)
        segments = stats.get("segments")
        eod = self.config.eod_token_id

        loss = ntp = mtp_l = None
        mtp_logits = None
        if labels is not None:
            targets = labels[:, 1:]
            if segments is not None:
                # Position i predicts token i+1, so the only pair that straddles a
                # boundary is the one whose input is the separator itself.
                targets = mask_targets_at_boundaries(
                    targets, input_ids[:, :-1], eod
                )
            ntp = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                targets.reshape(-1), ignore_index=-100,
            )
            loss = ntp
            if self.mtp is not None and use_mtp and input_ids.shape[1] >= 3:
                h_sl, e_sl, t_sl = mtp_slices(input_ids)
                fused = self.mtp(hidden[:, h_sl],
                                 self.model.embed_tokens(input_ids[:, e_sl]),
                                 segments=None if segments is None else segments[:, h_sl])
                mtp_logits = self.lm_head(self.model.final_norm(fused))
                mtp_targets = labels[:, t_sl]
                if segments is not None:
                    # MTP reads position i and the token at i+1 to predict i+2, so
                    # a separator at *either* of those two inputs makes the triple
                    # cross a boundary.
                    crosses = ((input_ids[:, h_sl] == eod)
                               | (input_ids[:, e_sl] == eod))
                    mtp_targets = mtp_targets.masked_fill(crosses, -100)
                mtp_l = mtp_loss(mtp_logits, mtp_targets)
                loss = loss + self.config.mtp.loss_weight * mtp_l

        return KaiNomosOutput(
            logits=logits, loss=loss, ntp_loss=ntp,
            mtp_logits=mtp_logits, mtp_loss=mtp_l,
        )

    def parameter_report(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        mtp_only = sum(p.numel() for n, p in self.named_parameters() if n.startswith("mtp."))
        mudd = sum(p.numel() for n, p in self.named_parameters() if ".mudd." in n)
        delta = sum(p.numel() for n, p in self.named_parameters()
                    if "delta_" in n)
        return {
            "total_params": total,
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "inference_backbone_params": total - mtp_only,
            "mtp_only_params": mtp_only,
            "mudd_params": mudd,
            "delta_params": delta,
        }


__all__ = [
    "KaiNomosForCausalLM", "KaiNomosModel", "KaiNomosOutput",
]
