"""
Training, checkpointing and inference of the RateGNN (see ml/model.py).

Protocol
  - leave-one-SAMPLE-out: the model that plans a sample never saw any scan of
    that sample (at any resolution);
  - inside the training samples, spatial blocks (1 in 5) are held out to pick
    the best step - never the test sample;
  - the reference to beat is printed next to every number: the same network at
    initialisation IS the bilinear baseline, so `val_nll - base_nll < 0` means
    the learning helped.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import make_example, SCANS, N_FEATURES, BLOCK, val_split
from .model import RateGNN, poisson_nll

def _tensors(ex, device="cpu"):
    t = {k: torch.from_numpy(ex[k]) for k in ("x", "erel", "qrel", "base")}
    t.update({k: torch.from_numpy(ex[k]).long() for k in ("src", "dst", "nbr")})
    if "y" in ex:
        t["y"] = torch.from_numpy(ex["y"])
    return {k: v.to(device) for k, v in t.items()}


def fold_examples(test_sample, target, scans=SCANS):
    """Training examples of a fold: every scan of the OTHER samples."""
    return [make_example(s, target, px)[0] for s, px in scans if s != test_sample]


def _nll(model, ex, t, idx, h=None):
    if h is None:
        h = model.encode(t["x"], t["src"], t["dst"], t["erel"])
    base = t["base"][idx]
    rel = model.decode(h, t["nbr"][idx], t["qrel"][idx], base, ex["rho"],
                       float(ex["base"].mean()))
    return poisson_nll(float(ex["scale"]) * rel, t["y"][idx], float(ex["t"]))


def pick_device(name="auto"):
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def train(examples, steps=1500, batch=16384, hidden=32, layers=3, lr=2e-3,
          seed=0, eval_every=100, log=print, device="auto", block=BLOCK):
    """Train on `examples`; returns (model with the best validation weights,
    history dict)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = pick_device(device)
    model = RateGNN(N_FEATURES, hidden, layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    data = []
    for ex in examples:
        tr, va = val_split(ex, block)
        data.append((ex, _tensors(ex, device), np.flatnonzero(tr), np.flatnonzero(va)))

    def validate():
        model.eval()
        tot = 0.0
        with torch.no_grad():
            for ex, t, _, va in data:
                idx = torch.from_numpy(va[:: max(1, len(va) // 60000)]).to(device)
                tot += float(_nll(model, ex, t, idx))
        model.train()
        return tot / len(data)

    base_nll = validate()                 # untrained = bilinear baseline
    # the step-0 weights are the fallback: a run that never beats the baseline
    # returns the baseline, not its last (possibly diverged) step
    best = base_nll
    best_state = {k: p.clone() for k, p in model.state_dict().items()}
    hist = {"step": [0], "val": [base_nll]}
    log(f"  step    0  val NLL {base_nll:.5f}  (= bilinear baseline)")
    t0 = time.time()
    for step in range(1, steps + 1):
        ex, t, tr, _ = data[rng.integers(len(data))]
        idx = torch.from_numpy(rng.choice(tr, size=min(batch, len(tr)),
                                          replace=False)).to(device)
        loss = _nll(model, ex, t, idx)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        if step % eval_every == 0 or step == steps:
            v = validate()
            hist["step"].append(step); hist["val"].append(v)
            if v < best:
                best, best_state = v, {k: p.clone() for k, p in model.state_dict().items()}
            log(f"  step {step:4d}  val NLL {v:.5f}  ({v - base_nll:+.5f} vs baseline)"
                f"  {time.time() - t0:5.0f}s")
    model.load_state_dict(best_state)
    hist.update(base_nll=base_nll, best_nll=best,
                improved=bool(best < base_nll))
    return model.to("cpu"), hist


@torch.no_grad()
def predict(model, ex, chunk=200000):
    """Relative fine rate map (in coarse-count units; multiply by the
    coarse->fine count scale and divide by the dwell to get counts/ms)."""
    model.eval()
    t = _tensors(ex)
    h = model.encode(t["x"], t["src"], t["dst"], t["erel"])
    out = np.empty(len(ex["base"]), dtype=np.float32)
    bm = float(ex["base"].mean())
    for a in range(0, len(out), chunk):
        s = slice(a, a + chunk)
        out[s] = model.decode(h, t["nbr"][s], t["qrel"][s], t["base"][s],
                              ex["rho"], bm).numpy()
    return out.reshape(ex["shape"])


def save(model, path, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "meta": meta}, path)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    m = ck["meta"]
    model = RateGNN(N_FEATURES, m["hidden"], m["layers"])
    model.load_state_dict(ck["state"])
    return model, m
