"""
Per-line variable-dwell scan plans (step 3).

Boltz, Ratner & Webb (J. Synchrotron Rad. 2026) ran their adaptive scans on
SSRL 7-2 as unidirectional line scans in which only the per-pixel dwell
varies; a hardware check inserted ramps where the stage acceleration limit
forbids the requested dwell change, then renormalised the total time. This
module does the same for a dwell map on the fine grid:

- each row is flown left to right; pixel i is crossed at v_i = px / t_i;
- skipped pixels (dwell 0) are crossed as fast as the kinematics allow;
- the speed is planned at pixel BOUNDARIES with constant acceleration inside
  each pixel, |u_(j+1)^2 - u_j^2| <= 2 a px, so every ramp is paid in full and
  a scanned pixel is entered and left no faster than px / dwell: it gets at
  least its requested dwell (a ramp can only add dwell);
- each row starts at the first and ends at the last scanned pixel; between
  rows the stage moves (trapezoidal profile, utils.roi_utils.travel_time) plus
  a per-line setup.

The raster baseline is timed with exactly the same model (uniform dwell,
full rows), so speedups compare like with like. Units: mm, ms (velocities in
mm/s for readability).
"""

import numpy as np

from .roi_utils import V_MAX_MM_S, A_MAX_MM_S2, travel_time

LINE_SETUP_MS = 0.0   # per-line trigger / settle overhead (unknown for 7-2)


def _boundary_speeds(cap, px_mm, a_max):
    """Stage speed at every pixel BOUNDARY of each row [mm/s].

    cap: (nrow, n) speed cap per pixel (px / requested dwell on scanned
    pixels, v_max across gaps, +inf past the end of a short row). Boundary j
    (0..n) sits between pixel j-1 and j; both neighbours' caps apply, so a
    scanned pixel is entered and left no faster than px / dwell and gets at
    least its requested dwell. With constant acceleration inside a pixel,
    |u_(j+1)^2 - u_j^2| <= 2 a px is exactly the kinematic bound, and the
    pixel takes 2 px / (u_j + u_(j+1)). Row ends are capped by the first /
    last pixel's own speed (the run-up and stop happen outside the row, for
    the raster and the adaptive plan alike).
    """
    nrow, n = cap.shape
    big = np.full((nrow, 1), np.inf)
    u = np.minimum(np.concatenate([cap, big], 1), np.concatenate([big, cap], 1))
    dv2 = 2.0 * a_max * px_mm
    for j in range(1, n + 1):                       # forward: accelerate
        u[:, j] = np.minimum(u[:, j], np.sqrt(u[:, j - 1] ** 2 + dv2))
    for j in range(n - 1, -1, -1):                  # backward: brake in time
        u[:, j] = np.minimum(u[:, j], np.sqrt(u[:, j + 1] ** 2 + dv2))
    return u


def plan_lines(dwell, xdata, ydata, v_max=V_MAX_MM_S, a_max=A_MAX_MM_S2,
               line_setup_ms=LINE_SETUP_MS):
    """Time a per-pixel dwell map [ms] (0 = not scanned) as line scans.

    Returns a dict:
      velocity_mm_s : (ny, nx) mean stage speed over each flown pixel (0 elsewhere)
      crossing_ms   : (ny, nx) time the stage spends over each flown pixel,
                      gaps included (what a per-pixel time table needs)
      dwell_ms      : (ny, nx) dwell obtained on SCANNED pixels (>= requested)
      line_ms       : (ny,) time spent flying each row
      move_ms       : total time of the moves between rows + setups
      total_ms      : everything
      extra_dwell   : fraction of extra dwell forced by the ramps
      x0, x1        : (ny,) first / last pixel index flown per row (-1 if none)
    """
    dwell = np.asarray(dwell, dtype=float)
    ny, nx = dwell.shape
    px = abs(float(np.median(np.diff(xdata)))) if nx > 1 else 1.0
    scanned = dwell > 0
    has = scanned.any(axis=1)
    x0 = np.where(has, scanned.argmax(axis=1), -1)
    x1 = np.where(has, nx - 1 - scanned[:, ::-1].argmax(axis=1), -1)
    cap = np.where(scanned, px / np.maximum(dwell, 1e-12) * 1000.0, v_max)
    cap = np.minimum(cap, v_max)

    cross = np.zeros_like(dwell)
    vel = np.zeros_like(dwell)
    line_ms = np.zeros(ny)
    rows = np.nonzero(has)[0]
    if len(rows):
        n_row = (x1 - x0 + 1)[rows]
        width = int(n_row.max())
        C = np.full((len(rows), width), np.inf)      # rows shifted to start at 0
        for k, r in enumerate(rows):
            C[k, :n_row[k]] = cap[r, x0[r]:x1[r] + 1]
        U = _boundary_speeds(C, px, a_max)
        T = 2.0 * px / np.maximum(U[:, :-1] + U[:, 1:], 1e-12) * 1000.0   # ms
        for k, r in enumerate(rows):
            m = n_row[k]
            cross[r, x0[r]:x1[r] + 1] = T[k, :m]
            vel[r, x0[r]:x1[r] + 1] = 0.5 * (U[k, :m] + U[k, 1:m + 1])
            line_ms[r] = float(T[k, :m].sum())

    got = np.where(scanned, cross, 0.0)
    move = 0.0
    for r0, r1 in zip(rows[:-1], rows[1:]):
        d = np.hypot(xdata[x0[r1]] - xdata[x1[r0]], ydata[r1] - ydata[r0])
        move += float(travel_time(d, v_max, a_max)) * 1000.0
    move += line_setup_ms * len(rows)
    req = dwell[scanned].sum()
    return {"velocity_mm_s": vel, "crossing_ms": cross, "dwell_ms": got,
            "line_ms": line_ms, "move_ms": move,
            "total_ms": float(line_ms.sum() + move),
            "extra_dwell": float(got[scanned].sum() / req - 1.0) if req else 0.0,
            "x0": x0, "x1": x1}


def raster_line_time(shape, xdata, ydata, dwell_ms, **kw):
    """Total time of a uniform raster under the same line-scan model."""
    return plan_lines(np.full(shape, float(dwell_ms)), xdata, ydata, **kw)["total_ms"]


def fit_to_budget(dwell, budget_ms, xdata, ydata, lo=1.0, hi=50.0, n_iter=30, **kw):
    """Scale a dwell map so the line-scan TOTAL (ramps and moves included)
    equals `budget_ms` - the renormalisation step of Boltz et al.
    Requested dwells stay within [lo, hi] where scanned. Returns (dwell, plan);
    plan["converged"] is False when the bounds make the budget unreachable
    (e.g. a small ROI already at `hi`), plan["scale"] is the factor applied."""
    dwell = np.asarray(dwell, dtype=float)
    scanned = dwell > 0
    s = 1.0
    d = dwell
    plan = None
    for _ in range(n_iter):
        d = np.where(scanned, np.clip(dwell * s, lo, hi), 0.0)
        plan = plan_lines(d, xdata, ydata, **kw)
        t = plan["total_ms"]
        if abs(t - budget_ms) <= 1e-4 * budget_ms:
            break
        fixed = t - d[scanned].sum()                 # ramps + gaps + moves
        target = max(budget_ms - fixed, lo * scanned.sum())
        s *= target / max(d[scanned].sum(), 1e-12)
    plan["converged"] = bool(abs(plan["total_ms"] - budget_ms) <= 1e-3 * budget_ms)
    plan["scale"] = float(s)
    return d, plan


def export_line_plan(path, plan, xdata, ydata, requested_dwell, meta=None):
    """Save the plan as .npz: per-row start/stop index, x / y axes [mm], the
    requested and the obtained per-pixel dwell [ms] (0 on gaps), the time over
    every flown pixel gaps included (crossing_ms) and the mean stage speed
    [mm/s]. `meta` (dict of scalars/strings) is stored alongside."""
    meta = meta or {}
    np.savez_compressed(
        path, xdata_mm=np.asarray(xdata), ydata_mm=np.asarray(ydata),
        row_start=plan["x0"], row_stop=plan["x1"],
        dwell_requested_ms=requested_dwell.astype(np.float32),
        dwell_ms=plan["dwell_ms"].astype(np.float32),
        crossing_ms=plan["crossing_ms"].astype(np.float32),
        velocity_mm_s=plan["velocity_mm_s"].astype(np.float32),
        line_ms=plan["line_ms"], total_ms=plan["total_ms"],
        **{f"meta_{k}": np.asarray(v) for k, v in meta.items()})
