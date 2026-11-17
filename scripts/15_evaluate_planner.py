"""
Benchmark the learned planner against the rule-based pipeline and against
non-learned controls, with the reference protocol (utils/bench.py: equal real
time, independent truth, MSE + detection + quantification).

For each held-out sample x target, at the SAME scanned area as the rule:
    rule k=0.5 / k=1   : the current pipeline (MAD ROIs + harmonic fill)
    bilinear, smoothed : non-learned rate maps through the same planner
    learned            : the RateGNN trained on the other samples
each with the rate map itself or the harmonic interpolation as the fill.
The learned model must beat the SMOOTHED control to be worth anything:
beating the raw threshold only shows that denoising helps.

    python scripts/15_evaluate_planner.py                 # every checkpoint found
    python scripts/15_evaluate_planner.py --samples UB1_P1 --targets Cu.Ka --draws 3
    python scripts/15_evaluate_planner.py --controls-only  # no PyTorch needed

Output: outputs/evaluation_planner.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils.paths import PROJECT_ROOT
from utils.bench import Case, SAMPLES
from ml.plan import plan_from_rate, rate_bilinear, rate_smoothed


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--samples", nargs="+", default=list(SAMPLES))
    p.add_argument("--targets", nargs="+", default=["Fe.Ka", "Zn.Ka", "Mn.Ka", "Cu.Ka"])
    p.add_argument("--models", default=str(PROJECT_ROOT / "phase2_ml" / "models"))
    p.add_argument("--controls-only", action="store_true",
                   help="rule + non-learned controls only (no checkpoint needed)")
    p.add_argument("--draws", type=int, default=10)
    p.add_argument("--boot", type=int, default=300)
    p.add_argument("--out", default=str(PROJECT_ROOT / "outputs" / "evaluation_planner.csv"))
    return p.parse_args(argv)


def evaluate(sample, target, args):
    case = Case(sample, target)
    kw = dict(draws=args.draws, boot=args.boot)
    rows, areas = [], {}
    for k in (0.5, 1.0):
        mask, weight, pred = case.rule_plan(k)
        areas[k] = float(mask[case.edge_ok & case.full].mean())
        rows += case.score(mask, weight, pred, name=f"rule_k{k:g}", **kw)

    rates = {"bilinear": rate_bilinear(case), "smoothed": rate_smoothed(case)}
    ckpt = Path(args.models) / target / f"heldout_{sample}.pt"
    if not args.controls_only:
        if ckpt.exists():
            from ml.data import make_example
            from ml.plan import rate_learned
            from ml.train import load
            model, meta = load(ckpt)
            assert meta["heldout"] == sample and meta["target"] == target
            ex, _ = make_example(sample, target, labels=False)
            rates["learned"] = rate_learned(case, ex, model)
        else:
            print(f"    no checkpoint {ckpt} - controls only")
    for name, rate in rates.items():
        for k, area in areas.items():
            for fill in ("rate", "harmonic"):
                plan = plan_from_rate(case, rate, area, fill)
                rows += case.score(*plan, name=f"{name}@k{k:g}+{fill}", **kw)
    return rows


def main(argv=None):
    args = parse_args(argv)
    rows = []
    for s in args.samples:
        for t in args.targets:
            print(f"  {s} {t} ...", flush=True)
            rows += evaluate(s, t, args)
    hdr = (f"{'sample':7} {'target':6} {'tr':2} {'strategy':24} {'area':>5} "
           f"{'MSE ratio [95% CI]':>21} {'sqrt':>5} {'quant':>6} {'recall':>6} {'prec':>5}")
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for r in rows:
        print(f"{r['sample']:7} {r['target']:6} {r['truth']:2} {r['strategy']:24} "
              f"{r['area']:5.0%} {r['mse_ratio']:6.2f} [{r['ci_lo']:5.2f},{r['ci_hi']:5.2f}] "
              f"{r['sqrt_ratio']:5.2f} {r['quant_err']:6.1%} {r['recall']:6.0%} "
              f"{r['precision']:5.0%}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  saved -> {out}")


if __name__ == "__main__":
    main()
