"""
Multi-resolution refinement cascade — single source of truth.

Generalized from script 05. The cascade progressively refines the sample
footprint: at each resolution level we scan only inside the mask carried over
from the coarser level, then re-detect the footprint on the newly acquired data.

The atomic operation is `refine_step` (one acquired level -> refined mask). It is
the production primitive: at the beamline each level is acquired AFTER analysing
the previous one, so a single `refine_step` is exactly what one beamline step
needs. `refine_cascade` simply loops it over a list of already-loaded scans
(used for offline validation / simulation, and by script 05).
"""

import re
from pathlib import Path

import numpy as np

from .roi_utils import sample_mask
from .validation_utils import project_mask
from .paths import find_first, DATA_DIR


# ── File discovery ────────────────────────────────────────────────────────────

def find_level_file(sample, px_um, data_dir=None):
    """
    Locate the HDF5 for a given sample stem and resolution.

    Pattern anchors the resolution token with underscores (`_<px>um_`) so that
    e.g. px=50 does NOT accidentally match a `250um` filename.
    """
    return find_first(f"*{sample}*_{px_um}um_*.hdf5", base_dir=data_dir)


def available_levels(sample, data_dir=None):
    """
    Discover the resolutions present for a sample, sorted coarse -> fine.
    Returns a list of ints (um), e.g. [250, 100, 50, 25].
    """
    base = Path(data_dir) if data_dir else DATA_DIR
    if not base.exists():
        return []
    found = set()
    for p in base.rglob(f"*{sample}*um*.hdf5"):
        m = re.search(r"_(\d+)um_", p.name)
        if m:
            found.add(int(m.group(1)))
    return sorted(found, reverse=True)


def dwell_ms_from_path(path, default=10.0):
    """
    Parse the per-pixel dwell [ms] from a SMAK filename like
    'SMW_UA1_P1_250um_10ms_...' (-> 10.0). Falls back to `default` if the
    '<N>ms' token is absent.
    """
    m = re.search(r"_(\d+(?:\.\d+)?)ms_", Path(path).name)
    return float(m.group(1)) if m else float(default)


# ── Refinement primitives ─────────────────────────────────────────────────────

def refine_step(comp, data, prev_mask=None, prev_data=None,
                kernel_px=2, mode="otsu_lower", level=50.0):
    """
    Refine the sample footprint for ONE acquired level.

    Parameters
    ----------
    comp       : 2D composite of the level just acquired
    data       : the level's data dict (needs 'xdata', 'ydata')
    prev_mask  : mask from the coarser level (None at the first level)
    prev_data  : data dict of the coarser level (needed to project prev_mask)
    kernel_px, mode, level : sample_mask parameters

    Returns
    -------
    (mask, thresh, stats)
        mask   : bool 2D array on this level's grid
        thresh : threshold used by sample_mask
        stats  : dict {n_in, n_scanned, n_tot, area_frac, signal_in_frac,
                       threshold}

    Note on the two pixel counts: `n_in` is the size of the REFINED mask
    (what survives re-detection at this level), while `n_scanned` is the
    number of pixels actually ACQUIRED at this level — the full frame at the
    first level (you must scan everything to detect anything), the projection
    of the previous level's mask afterwards. Time accounting must use
    `n_scanned`; using n_in undercounts the cascade cost.
    """
    if prev_mask is None:
        mask, thresh = sample_mask(comp, kernel_px=kernel_px, mode=mode, level=level)
        n_scanned = int(mask.size)
    else:
        mask_here = project_mask(prev_mask, prev_data["xdata"], prev_data["ydata"],
                                 data["xdata"], data["ydata"])
        mask, thresh = sample_mask(comp, kernel_px=kernel_px, mode=mode,
                                   level=level, restrict_to=mask_here)
        n_scanned = int(mask_here.sum())

    n_in, n_tot = int(mask.sum()), int(mask.size)
    total = float(comp.sum())
    stats = {
        "n_in":           n_in,
        "n_scanned":      n_scanned,
        "n_tot":          n_tot,
        "area_frac":      n_in / n_tot if n_tot else 0.0,
        "signal_in_frac": float(comp[mask].sum()) / total if total > 0 else 0.0,
        "threshold":      thresh,
    }
    return mask, thresh, stats


def refine_cascade(scans, kernel_px=2, mode="otsu_lower", level=50.0,
                   detect_final=False, verbose=False):
    """
    Run the full refinement cascade over an ordered list of loaded scans.

    Parameters
    ----------
    scans        : list (coarse -> fine) of dicts {"data", "comp", "px"}
    detect_final : if False (script-05 behavior), the finest level only PROJECTS
                   the previous mask without re-running sample_mask (because 05
                   does ROI detection there instead). If True, the finest level
                   is also re-detected.

    Returns
    -------
    (final_mask, per_level_stats)
        per_level_stats : list of dicts with keys {px, shape, n_in, n_scanned,
                          n_tot, area_frac, signal_in_frac, threshold}
    """
    mask = None
    prev_data = None
    stats_all = []
    n = len(scans)

    for i, sc in enumerate(scans):
        comp, data = sc["comp"], sc["data"]
        px = sc.get("px")
        is_final = (i == n - 1)

        if is_final and not detect_final and mask is not None:
            # 05 behavior: carry footprint onto the fine grid (no re-detection).
            m = project_mask(mask, prev_data["xdata"], prev_data["ydata"],
                             data["xdata"], data["ydata"])
            total = float(comp.sum())
            n_in, n_tot = int(m.sum()), int(m.size)
            st = {
                "n_in":           n_in,
                "n_scanned":      n_in,   # the projected mask IS what's scanned
                "n_tot":          n_tot,
                "area_frac":      n_in / n_tot if n_tot else 0.0,
                "signal_in_frac": float(comp[m].sum()) / total if total > 0 else 0.0,
                "threshold":      None,
            }
        else:
            m, thresh, st = refine_step(comp, data, prev_mask=mask,
                                        prev_data=prev_data, kernel_px=kernel_px,
                                        mode=mode, level=level)

        st["px"] = px
        st["shape"] = data["mapdata"].shape if "mapdata" in data else comp.shape
        stats_all.append(st)

        if verbose:
            tag = f"{px}um" if px is not None else f"level {i}"
            print(f"  [{tag}] in-mask {st['n_in']:,}/{st['n_tot']:,} "
                  f"({st['area_frac']*100:.1f}%)  signal {st['signal_in_frac']*100:.1f}%")

        mask = m
        prev_data = data

    return mask, stats_all
