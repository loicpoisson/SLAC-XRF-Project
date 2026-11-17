"""
Training / inference examples for the learned planner (numpy only).

One example = one sample x target element x fine resolution:
  - a GRAPH of the 100 um detection scan: one node per coarse pixel, features
    = every element channel (the detector records them all), edges to the 8
    neighbours with their relative offsets;
  - QUERIES = the fine pixels, each attached to the 4 coarse nodes around it;
  - the fine scan's counts as the label (training samples only).

Nothing of the fine scan enters the inputs. The coarse fly-scan edge columns
are neutralised like in utils.bench.
"""

import numpy as np

from utils.bench import Case, without_edges, SAMPLES, T_REF
from utils.hdf5_reader import get_channel
from utils.reconstruction import fill_bilinear

ELEMENTS = ["Fe.Ka", "Au.La", "Al.Ka", "Si.Ka", "P.Ka", "S.Ka", "Cl.Ka", "K.Ka",
            "Ca.Ka", "Ti.Ka", "V.Ka", "Cr.Ka", "Mn.Ka", "Cu.Ka", "Zn.Ka"]
N_FEATURES = len(ELEMENTS) + 3     # + target (absolute), target @250, local contrast

# every (sample, fine pixel) with a full fine scan: UA1/UB1 also have 50 um
SCANS = [("UA1_P1", 25), ("UB1_P1", 25), ("UA1_P1", 50), ("UB1_P1", 50), ("FP1_P1", 50)]


BLOCK = 16          # coarse pixels (1.6 mm): spatial validation blocks


def val_split(ex, block=BLOCK):
    """Boolean (Q,) arrays: train / validation queries (1 block in 5).

    Blocks are defined on the COARSE grid (physical position), so the 25 um
    and the 50 um scans of one sample hold out the same area: a validation
    pixel is never trained on through the sample's other resolution."""
    nx = ex["cshape"][1]
    node = ex["nbr"][:, 0]                       # lower-left coarse node of the pixel
    by, bx = (node // nx) // block, (node % nx) // block
    held = (by + 2 * bx) % 5 == 0
    return ex["valid"] & ~held, ex["valid"] & held


def grid_edges(ny, nx):
    """Directed edges (src, dst) between 8-neighbours of an ny x nx grid and
    their relative offsets (dy, dx) in pixels."""
    idx = np.arange(ny * nx).reshape(ny, nx)
    src, dst, rel = [], [], []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ys = slice(max(0, -dy), ny - max(0, dy))
            xs = slice(max(0, -dx), nx - max(0, dx))
            yd = slice(max(0, dy), ny - max(0, -dy))
            xd = slice(max(0, dx), nx - max(0, -dx))
            s, d = idx[ys, xs].ravel(), idx[yd, xd].ravel()
            src.append(s); dst.append(d)
            rel.append(np.tile([dy, dx], (len(s), 1)))
    return (np.concatenate(src), np.concatenate(dst),
            np.concatenate(rel).astype(np.float32))


def node_features(case):
    """(N, N_FEATURES) float32 features of the 100 um nodes.

    Element channels are divided by (their median + 1) and log1p'ed, so the
    features do not depend on the sample's overall count level; the target
    channel is also given in absolute log counts (its Poisson noise level
    matters), together with the 250 um scan of the target and the target's
    contrast to its 3x3 neighbourhood."""
    from scipy import ndimage
    d100 = case.data["c100"]
    chans = [without_edges(get_channel(d100, c)) for c in ELEMENTS]
    x = [np.log1p(np.maximum(c, 0) / (np.median(c) + 1.0)) for c in chans]
    tgt = case.coarse(100)
    x.append(np.log1p(np.maximum(tgt, 0)))
    t250 = fill_bilinear(case.coarse(250), case.data["c250"], d100)
    x.append(np.log1p(np.maximum(t250, 0)))
    x.append(np.log1p(np.maximum(tgt, 0))
             - np.log1p(ndimage.uniform_filter(np.maximum(tgt, 0), 3)))
    return np.stack(x, axis=-1).reshape(-1, len(x)).astype(np.float32)


def queries(case):
    """Per fine pixel: indices of the 4 surrounding coarse nodes, the offsets
    to them in coarse-pixel units, and the bilinear baseline (target counts of
    the coarse scan interpolated to the pixel)."""
    d100, F = case.data["c100"], case.F
    xc, yc = np.asarray(d100["xdata"]), np.asarray(d100["ydata"])
    ny, nx = len(yc), len(xc)
    u = (np.asarray(F["xdata"]) - xc[0]) / (xc[1] - xc[0])       # coarse col coords
    v = (np.asarray(F["ydata"]) - yc[0]) / (yc[1] - yc[0])
    u0 = np.clip(np.floor(u).astype(int), 0, nx - 2)
    v0 = np.clip(np.floor(v).astype(int), 0, ny - 2)
    VV, UU = np.meshgrid(v0, u0, indexing="ij")
    fv, fu = np.meshgrid(v - v0, u - u0, indexing="ij")
    nbr = np.stack([VV * nx + UU, VV * nx + UU + 1,
                    (VV + 1) * nx + UU, (VV + 1) * nx + UU + 1], axis=-1)
    rel = np.stack([np.stack([fv, fu], -1), np.stack([fv, fu - 1], -1),
                    np.stack([fv - 1, fu], -1), np.stack([fv - 1, fu - 1], -1)], axis=-2)
    base = fill_bilinear(case.coarse(100), d100, F)
    return (nbr.reshape(-1, 4), rel.reshape(-1, 4, 2).astype(np.float32),
            base.reshape(-1).astype(np.float32))


def make_example(sample, target, fine_px=None, labels=True):
    """Dict of numpy arrays for one case. `scale` = coarse->fine count scale:
    measured on the example itself when it is a training example (labels=True),
    left to the caller otherwise."""
    case = Case(sample, target, fine_px or SAMPLES[sample])
    ny, nx = case.maps["c100"].shape
    src, dst, rel = grid_edges(ny, nx)
    nbr, qrel, base = queries(case)
    ex = {"sample": sample, "target": target, "fine_px": case.fine_px,
          "x": node_features(case), "src": src, "dst": dst, "erel": rel,
          "nbr": nbr, "qrel": qrel, "base": base,
          "rho": np.float32(case.fine_px / 100.0), "shape": case.shape,
          "cshape": (ny, nx), "coarse_dwell": np.float32(case.dwell["c100"])}
    if labels:
        y = case.maps["A"].reshape(-1).astype(np.float32)          # counts at 10 ms
        valid = ((case.maps["A"].sum(axis=1) > 0)[:, None] & case.edge_ok
                 & case.full).reshape(-1)
        ex.update(y=y, valid=valid, t=np.float32(case.dwell["A"] / T_REF),
                  scale=np.float32(y[valid].mean() / max(base[valid].mean(), 1e-9)))
    return ex, case
