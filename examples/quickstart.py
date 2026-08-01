"""Instantiate the full KaiNomos backbone and run one CUDA forward pass.

The public quick start is GPU-first. ``--tiny`` is retained only as a fast
implementation check for contributors.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURE = ROOT / "architecture"
sys.path.insert(0, str(ARCHITECTURE))

from config import KaiNomosConfig
from model import KaiNomosForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="use the small reference configuration instead of the full model",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--length", type=int, default=16)
    args = parser.parse_args()
    if args.length < 2:
        raise SystemExit("--length must be at least 2")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")

    config = KaiNomosConfig.tiny() if args.tiny else KaiNomosConfig()
    model = KaiNomosForCausalLM(config).to(args.device).eval()
    input_ids = torch.randint(
        5,
        config.vocab_size,
        (1, args.length),
        device=args.device,
    )
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if args.device.startswith("cuda")
        else nullcontext()
    )
    with torch.no_grad(), autocast:
        output = model(input_ids, return_route_stats=True)

    print(
        json.dumps(
            {
                "configuration": "tiny" if args.tiny else "full",
                "device": args.device,
                "input_shape": list(input_ids.shape),
                "logits_shape": list(output.logits.shape),
                "route_stat_records": len(output.route_stats),
                "parameters": model.parameter_report(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
