"""
Scanner parameters (kinematics, dwell bounds, default ladder) - one place.

Read once at import from, in order:
  1. the file named by the XRF_BEAMLINE environment variable,
  2. <project root>/beamline.json,
  3. the built-in defaults below (Betterton et al. 2020 kinematics).
A partial file is fine: missing keys fall back to the defaults.

Usage:  from utils.beamline import BEAMLINE;  BEAMLINE["v_max_mm_s"]
"""

import json
import math
import os
from pathlib import Path

DEFAULTS = {
    "name": "built-in defaults (Betterton 2020 kinematics)",
    "v_max_mm_s": 200.0,        # stage max velocity
    "a_max_mm_s2": 500.0,       # stage max acceleration
    "region_setup_ms": 500.0,   # per-rectangle overhead (rectangle plans)
    "line_setup_ms": 0.0,       # per-row overhead (line plans)
    "fine_dwell_ms": 10.0,      # reference dwell of the raster baseline
    "dwell_low_ms": 1.0,        # shortest dwell the scanner should be asked for
    "dwell_high_ms": 50.0,      # longest
    "target_ladder_um": [250, 100, 25],   # --target ladder when no files tell
}

_NUMERIC = [k for k, v in DEFAULTS.items() if isinstance(v, float)]


def _number(path, key, value, allow_zero):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path}: {key} must be a number, got {value!r}")
    v = float(value)
    if not math.isfinite(v) or v < 0 or (v == 0 and not allow_zero):
        raise ValueError(f"{path}: {key} must be a finite "
                         f"{'non-negative' if allow_zero else 'positive'} number, "
                         f"got {value!r}")
    return v


def load_beamline(path=None):
    """Return the parameter dict (defaults overlaid with `path` if given)."""
    cfg = dict(DEFAULTS)
    if path is None:
        return cfg
    try:
        with open(path, encoding="utf-8") as f:
            user = json.load(f)
    except OSError as e:
        raise ValueError(f"beamline file not readable: {path} ({e}). Check the "
                         f"XRF_BEAMLINE environment variable.") from e
    if not isinstance(user, dict):
        raise ValueError(f"{path}: expected a JSON object")
    unknown = sorted(set(user) - set(DEFAULTS) - {"_comment"})
    if unknown:
        raise ValueError(f"{path}: unknown beamline key(s) {unknown}; "
                         f"known: {sorted(DEFAULTS)}")
    cfg.update({k: v for k, v in user.items() if k != "_comment"})
    for k in _NUMERIC:
        cfg[k] = _number(path, k, cfg[k], allow_zero=k in ("line_setup_ms",
                                                            "region_setup_ms"))
    if not cfg["dwell_low_ms"] <= cfg["fine_dwell_ms"] <= cfg["dwell_high_ms"]:
        raise ValueError(f"{path}: need dwell_low_ms <= fine_dwell_ms <= dwell_high_ms")
    lad = cfg["target_ladder_um"]
    if (not isinstance(lad, list) or len(lad) < 2
            or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in lad)
            or any(a_ <= b_ for a_, b_ in zip(lad, lad[1:]))):
        raise ValueError(f"{path}: target_ladder_um must list >= 2 positive integer "
                         f"pixel sizes in um, coarse to fine, got {lad!r}")
    cfg["source"] = str(path)
    return cfg


def _default_path():
    env = os.environ.get("XRF_BEAMLINE")
    if env:
        return Path(env)
    p = Path(__file__).parent.parent.parent / "beamline.json"
    return p if p.exists() else None


BEAMLINE = load_beamline(_default_path())
