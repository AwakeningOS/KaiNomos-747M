"""Cache-stable generation API for KaiNomos-750M."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from model import KaiNomosForCausalLM, ModelCache


@dataclass
class GenerationState:
    cache: ModelCache | None = None
    previous_was_eod: bool = False


@torch.no_grad()
def cached_forward(
    model: KaiNomosForCausalLM,
    input_ids: torch.Tensor,
    state: GenerationState | None = None,
):
    """Run a prompt/chunk and return logits plus a resumable cache.

    Packed-training document masks are intentionally not reused here.  At
    generation time an EOD starts a new stream, so all temporal caches are
    dropped immediately before processing the first token after EOD.
    """
    state = state or GenerationState()
    outputs = []
    cache = state.cache
    previous_was_eod = state.previous_was_eod
    for position in range(input_ids.shape[1]):
        if previous_was_eod:
            cache = None
        token = input_ids[:, position:position + 1]
        result = model(
            token,
            respect_documents=False,
            cache=cache,
            use_cache=True,
        )
        outputs.append(result.logits)
        cache = result.cache
        previous_was_eod = bool((token == model.config.eod_token_id).all())
    return torch.cat(outputs, dim=1), GenerationState(cache, previous_was_eod)


@torch.no_grad()
def generate(
    model: KaiNomosForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    *,
    temperature: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    logits, state = cached_forward(model, input_ids)
    output = input_ids
    for _ in range(max_new_tokens):
        next_logits = logits[:, -1]
        if temperature <= 0:
            token = next_logits.argmax(-1, keepdim=True)
        else:
            probability = (next_logits / temperature).softmax(-1)
            token = torch.multinomial(probability, 1, generator=generator)
        output = torch.cat((output, token), dim=1)
        logits, state = cached_forward(model, token, state)
    return output


__all__ = ["GenerationState", "cached_forward", "generate"]
