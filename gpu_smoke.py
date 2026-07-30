"""GPU smoke test: forward, backward, step, eval, checkpoint save and reload."""
import json, time, torch
from config import KaiNomosConfig as Config
from model import KaiNomosForCausalLM as Model
from train import TrainConfig, build_optimizer

cfg = Config()
torch.manual_seed(11)
m = Model(cfg).cuda()
report = m.parameter_report()
train_cfg = TrainConfig()
train_cfg.optimizer = "muon"
opt = build_optimizer(m, train_cfg, "muon")

import os
mb, T = int(os.environ.get("MB", 8)), 1024
accum = 65536 // (mb * T)
torch.cuda.reset_peak_memory_stats()
for it in range(3):
    if it == 1:
        torch.cuda.synchronize(); t0 = time.time()
    opt.zero_grad(set_to_none=True)
    for _ in range(accum):
        ids = torch.randint(0, cfg.vocab_size, (mb, T), device='cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = m(ids, labels=ids)
        (out.loss / accum).backward()
    gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
torch.cuda.synchronize()
dt = (time.time() - t0) / 2
peak = torch.cuda.max_memory_allocated() / 1e9

m.eval()
with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
    ev = m(torch.randint(0, cfg.vocab_size, (2, T), device='cuda'),
           labels=torch.randint(0, cfg.vocab_size, (2, T), device='cuda'))

torch.save({"model": m.state_dict(), "config": cfg.to_dict()}, "runs/smoke.pt")
blob = torch.load("runs/smoke.pt", map_location="cpu", weights_only=False)
m2 = Model(Config.from_dict(blob["config"])); m2.load_state_dict(blob["model"])

result = {
    **report,
    "optimizer": "muon",
    "micro_batch": mb, "seq_len": T, "grad_accum": accum,
    "step_seconds": round(dt, 3), "tokens_per_sec": round(65536 / dt),
    "peak_vram_gb": round(peak, 2),
    "initial_ntp_loss": round(float(out.ntp_loss.detach()), 4),
    "initial_mtp_loss": round(float(out.mtp_loss.detach()), 4),
    "grad_norm": round(float(gn.detach()), 3),
    "checkpoint_reload_ok": True,
}
print(json.dumps(result, indent=2))
open("runs/gpu_smoke.json", "w").write(json.dumps(result, indent=2))
