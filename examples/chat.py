"""Interactive User/Assistant loop for a trained KaiNomos checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import sys
from contextlib import nullcontext
from pathlib import Path

import sentencepiece as spm
import torch

ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURE = ROOT / "architecture"
sys.path.insert(0, str(ARCHITECTURE))

from config import KaiNomosConfig
from generation import generate
from model import KaiNomosForCausalLM


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=ROOT / "tokenizer" / "kainomos-49152.model",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens < 1_024:
        raise SystemExit("--max-new-tokens must be between 1 and 1023")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")

    blob = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    config = KaiNomosConfig.from_dict(blob["model_config"])
    model = KaiNomosForCausalLM(config)
    model.load_state_dict(blob["model"], strict=True)
    model = model.to(args.device).eval()
    tokenizer = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    if tokenizer.get_piece_size() != config.vocab_size:
        raise RuntimeError("tokenizer and checkpoint vocabulary sizes differ")
    expected_tokenizer = blob.get("metadata", {}).get("tokenizer_sha256")
    if expected_tokenizer and sha256(args.tokenizer) != expected_tokenizer:
        raise RuntimeError("tokenizer SHA-256 differs from the training checkpoint")

    history = ""
    print("KaiNomos interactive completion. Type /clear or /exit.")
    while True:
        try:
            user = input("User> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user == "/exit":
            break
        if user == "/clear":
            history = ""
            print("History cleared.")
            continue
        prompt = f"{history}User: {user}\nAssistant:"
        ids = tokenizer.encode(prompt, out_type=int)
        if len(ids) > config.context_length_train - args.max_new_tokens:
            ids = ids[-(config.context_length_train - args.max_new_tokens) :]
        input_ids = torch.tensor([ids], dtype=torch.long, device=args.device)
        precision = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if args.device.startswith("cuda")
            else nullcontext()
        )
        with precision:
            output = generate(
                model,
                input_ids,
                args.max_new_tokens,
                temperature=args.temperature,
            )
        generated = output[0, input_ids.shape[1] :].tolist()
        for stop_id in (3, 4):
            if stop_id in generated:
                generated = generated[: generated.index(stop_id)]
        answer = tokenizer.decode(generated).strip()
        print(f"Assistant> {answer}")
        history = f"{prompt} {answer}\n"


if __name__ == "__main__":
    main()
