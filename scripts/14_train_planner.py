"""
Train the learned planner (RateGNN, scripts/ml) - leave-one-sample-out.

One model per (held-out sample, target element): it is trained on every scan
of the OTHER samples and will only ever plan the held-out one.

    python scripts/14_train_planner.py                          # all folds, Fe Zn Mn Cu
    python scripts/14_train_planner.py --samples UB1_P1 --targets Cu.Ka --steps 300

Needs PyTorch (a separate environment from the smak one; see phase2_ml/README.md).
Runs on a GPU when one is visible (--device auto), e.g. on Stanford FarmShare:
    sbatch phase2_ml/farmshare_train.sbatch
Checkpoints: phase2_ml/models/<target>/heldout_<sample>.pt (+ .json with the
training curve). Evaluate them with scripts/15_evaluate_planner.py.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.paths import PROJECT_ROOT
from utils.bench import SAMPLES


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--samples", nargs="+", default=list(SAMPLES),
                   help="held-out samples (one model each)")
    p.add_argument("--targets", nargs="+", default=["Fe.Ka", "Zn.Ka", "Mn.Ka", "Cu.Ka"])
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=16384, help="fine pixels per step")
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda")
    p.add_argument("--out", default=str(PROJECT_ROOT / "phase2_ml" / "models"))
    p.add_argument("--force", action="store_true", help="retrain existing checkpoints")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from ml.train import fold_examples, train, save, pick_device
    print(f"device: {pick_device(args.device)}")
    for target in args.targets:
        for sample in args.samples:
            path = Path(args.out) / target / f"heldout_{sample}.pt"
            if path.exists() and not args.force:
                print(f"{path} exists - skipped (--force to retrain)")
                continue
            print(f"\n=== target {target}, held-out {sample} ===", flush=True)
            t0 = time.time()
            examples = fold_examples(sample, target)
            print("  trained on: " + ", ".join(f"{e['sample']}@{e['fine_px']}um"
                                               for e in examples), flush=True)
            model, hist = train(examples, steps=args.steps, batch=args.batch,
                                hidden=args.hidden, layers=args.layers, lr=args.lr,
                                seed=args.seed, device=args.device,
                                eval_every=max(1, args.steps // 30))
            meta = {"target": target, "heldout": sample, "hidden": args.hidden,
                    "layers": args.layers, "steps": args.steps, "seed": args.seed,
                    "trained_on": [f"{e['sample']}@{e['fine_px']}" for e in examples],
                    "history": hist, "minutes": (time.time() - t0) / 60}
            save(model, path, meta)
            print(f"  saved {path}  (val NLL {hist['best_nll']:.5f}, baseline "
                  f"{hist['base_nll']:.5f}, {meta['minutes']:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
