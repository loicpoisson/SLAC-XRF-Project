"""
Default-file discovery helpers.

Every script in scripts/ uses these to auto-detect a coarse or fine HDF5
inside the project's data/ folder, regardless of the current working
directory. Callers can always override the autodetection by passing an
explicit path.
"""

import re
from pathlib import Path

# This file lives at scripts/utils/paths.py -> project root is two levels up.
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR     = PROJECT_ROOT / "data"


def sample_hint_from_path(path):
    """Extract a sample stem like 'UA1_P1' from a SMAK filename, else None."""
    if path is None:
        return None
    m = re.search(r"SMW_(.+?)_\d+um_", Path(path).name)
    return m.group(1) if m else None


def find_first(pattern, base_dir=None):
    """
    Return the first file matching the glob `pattern` under `base_dir`
    (or the project's data/ if base_dir is None), sorted alphabetically.
    Returns None when nothing matches.
    """
    base = Path(base_dir) if base_dir else DATA_DIR
    if not base.exists():
        return None
    matches = sorted(base.rglob(pattern))
    return matches[0] if matches else None


def find_coarse(sample_hint=None, res_um=250):
    """
    Find a default coarse HDF5 (default: 250 um resolution).
    If `sample_hint` is given (e.g. 'UA1_P1'), prefer that sample.

    The resolution token is anchored as `_<res>um_` so that e.g. res_um=50
    never matches a '250um' filename ('50um' is a substring of '250um').
    """
    if sample_hint:
        m = find_first(f"*{sample_hint}*_{res_um}um_*.hdf5")
        if m is not None:
            return m
    return find_first(f"*_{res_um}um_*.hdf5")


def find_fine(sample_hint=None, res_um=25):
    """
    Find a default fine HDF5 (default: 25 um resolution).
    If `sample_hint` is given, prefer that sample.

    The resolution token is anchored as `_<res>um_` so that e.g. res_um=50
    never matches a '250um' filename.
    """
    if sample_hint:
        m = find_first(f"*{sample_hint}*_{res_um}um_*.hdf5")
        if m is not None:
            return m
    return find_first(f"*_{res_um}um_*.hdf5")


def require(path, what="file"):
    """
    Raise a friendly error if `path` is None or does not exist on disk.
    Use to validate auto-detected defaults before opening the file.
    """
    if path is None:
        raise SystemExit(
            f"No {what} found by auto-detection under {DATA_DIR}.\n"
            f"Pass --coarse / --fine explicitly, or place an HDF5 file in data/."
        )
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"{what.capitalize()} not found: {p}")
    return p
