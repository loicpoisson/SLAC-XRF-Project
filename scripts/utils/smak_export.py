"""
Write / read a scan plan in the region-queue file format SMAK produces.

SMAK's "File > Export scan area to dataqueue" (smak.py, Main.exportScanAreaDQ,
lines 11585-11652; identical copy in smak/__main__.py 10317-10384) is the only
code in SMAK that touches this format. It writes a `.dpm` file that is

    b'DATASERVER REGIONS\\n'                         # header line (11649)
    pickle.dumps({1: ['image', region],              # queue dict (11633-11648)
                  2: ['image', region], ...})

with 1-based int keys, and `region` a dict with exactly these keys, in this
insertion order (11607-11632):

    comment   '\\n'                 (11608)
    ooptics   'No'                  (11609)
    scandir   'Horizontal'          (11610)
    scantype  'Continuous'          (11611)
    slave     'Enabled'             (11612)
    dwell     float, milliseconds   (11614: prompt "dwell time ... (ms)")
    energy    [float], eV           (11617: "MONO energy ... (eV)"; a 1-item list)
    wheel     int, sample wheel     (11620, default 0)
    file      str, region file name (11623, default '')
    yrange    [x_start, x_end, +dx]   map X (horizontal), in map units = mm
    zrange    [y_start, y_end, -dx]   map Y (vertical), in map units = mm

The axis naming is the stage's: SMAK puts the map's X axis (Display.xsc =
mapdata.xvals = HDF5 main/xdata) in `yrange` and the map's Y axis (Display.ysc
= mapdata.yvals[::-1], Display.py:1342-1343) in `zrange` (11625-11632).
For the zoom box, zmxyi[0] < zmxyi[2] and zmxyi[1] < zmxyi[3] (Display.py
1002-1006); because ysc is reversed, zrange[0] = ysc[zmxyi[3]] is the SMALLER
y and zrange[1] the LARGER, i.e. both ranges run low -> high in mm (for the
ascending xdata/ydata in our HDF5 files). The zrange step is
`xsc[0] - xsc[1]`, i.e. MINUS the X pixel size (SMAK assumes square pixels and
reuses dx; 11631). So zrange has start < stop with a negative step. We
reproduce that convention exactly; how the dataserver interprets it is not
in SMAK (see "uncertain" notes in write_smak_regions).

When the file already exists SMAK validates the header by opening it in TEXT
mode and comparing readline() to 'DATASERVER REGIONS\\n' (11597-11605), then
re-opens it binary, skips one line, pickle.loads() the rest and appends the new
region at key len(queue)+1 (11637-11644). The text-mode readline decodes the
whole first chunk with the locale codec (cp1252 on the beamline Windows PCs),
so a binary pickle (protocol >= 1) makes it raise UnicodeDecodeError for ~43%
of plans. We therefore write pickle protocol 0, which is pure ASCII, loads in
any Python 2 or 3, and keeps the file appendable from SMAK's own menu.
"""

import math
import pickle
from pathlib import Path

HEADER = b"DATASERVER REGIONS\n"
HEADER_TEXT = "DATASERVER REGIONS\n"   # what SMAK compares against (text mode)
SCAN_KIND = "image"                    # first element of each queue entry

# Key order exactly as SMAK's writer inserts them (smak.py 11608-11632).
SMAK_REGION_KEYS = ("comment", "ooptics", "scandir", "scantype", "slave",
                    "dwell", "energy", "wheel", "file", "yrange", "zrange")

# SMAK's hard-coded defaults for the fields it does not ask the user for
# (smak.py 11608-11612) plus the dialog defaults for wheel/file (11620, 11623).
SMAK_DEFAULTS = {
    "comment":  "\n",
    "ooptics":  "No",
    "scandir":  "Horizontal",
    "scantype": "Continuous",
    "slave":    "Enabled",
    "wheel":    0,
}

_REQUIRED = ("x_start_mm", "x_end_mm", "y_start_mm", "y_end_mm",
             "fine_px_mm", "dwell_uniform_ms")


def _finite(name, v):
    v = float(v)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {v!r}")
    return v


def smak_region(r, energy_ev, file_name, *, comment=SMAK_DEFAULTS["comment"],
                ooptics=SMAK_DEFAULTS["ooptics"],
                scandir=SMAK_DEFAULTS["scandir"],
                scantype=SMAK_DEFAULTS["scantype"],
                slave=SMAK_DEFAULTS["slave"], wheel=SMAK_DEFAULTS["wheel"]):
    """Convert one 10_run_pipeline `_region_dict` into a SMAK region dict."""
    missing = [k for k in _REQUIRED if k not in r]
    if missing:
        raise KeyError(f"region {r.get('region_id', '?')} lacks {missing}")
    x0, x1 = _finite("x_start_mm", r["x_start_mm"]), _finite("x_end_mm", r["x_end_mm"])
    y0, y1 = _finite("y_start_mm", r["y_start_mm"]), _finite("y_end_mm", r["y_end_mm"])
    px = _finite("fine_px_mm", r["fine_px_mm"])
    dwell = _finite("dwell_uniform_ms", r["dwell_uniform_ms"])
    if px <= 0:
        raise ValueError(f"fine_px_mm must be > 0, got {px}")
    if dwell <= 0:
        raise ValueError(f"dwell_uniform_ms must be > 0, got {dwell}")
    # SMAK always writes low -> high (sorted zoom box); do the same.
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    return {
        "comment":  str(comment),
        "ooptics":  str(ooptics),
        "scandir":  str(scandir),
        "scantype": str(scantype),
        "slave":    str(slave),
        "dwell":    dwell,                      # ms
        "energy":   ([float(energy_ev)] if isinstance(energy_ev, (int, float))
                     else [float(e) for e in energy_ev]),   # eV, list like SMAK
        "wheel":    int(wheel),
        "file":     str(file_name),
        "yrange":   [x0, x1, px],               # map X, +step
        "zrange":   [y0, y1, -px],              # map Y, -step (SMAK: xsc[0]-xsc[1])
    }


def write_smak_regions(path, regions, energy_ev, stem, *, append=False,
                       protocol=0, **fields):
    """
    Write `regions` (list of 10_run_pipeline `_region_dict` dicts) as a SMAK
    dataqueue region file (.dpm). Returns the queue dict that was written.

    energy_ev : mono energy in eV (scalar, or a sequence for several energies;
                SMAK itself always writes a 1-item list).
    stem      : used for each region's `file` field, "<stem>_r<region_id>".
    append    : like SMAK when the file exists: keep its entries and add ours at
                keys len(queue)+1, ... (smak.py 11635-11644). Default overwrites.
    protocol  : pickle protocol. 0 (ASCII) keeps SMAK's text-mode header check
                from crashing; see module docstring.
    fields    : override SMAK's defaults for comment / ooptics / scandir /
                scantype / slave / wheel (all regions).

    Values only seen as SMAK literals and to be confirmed with the beamline
    scientist: ooptics/scandir/scantype/slave alternatives, wheel numbering,
    whether `file` should carry an extension or the dataserver adds the
    _NNN.hdf5 suffix, and the dataserver's reading of zrange's negative step.
    """
    unknown = set(fields) - set(SMAK_DEFAULTS)
    if unknown:
        raise TypeError(f"unknown SMAK field override(s): {sorted(unknown)}")
    if isinstance(energy_ev, (int, float)):
        energy = [_finite("energy_ev", energy_ev)]
    else:
        energy = [_finite("energy_ev", e) for e in energy_ev]
    if not energy or any(e <= 0 for e in energy):
        raise ValueError(f"energy_ev must be > 0 eV, got {energy_ev!r}")
    stem = str(stem)
    if any(c in stem for c in "/\\"):
        raise ValueError(f"stem must be a bare name, got {stem!r}")

    path = Path(path)
    queue = read_smak_regions(path) if (append and path.exists()) else {}
    start = len(queue) + 1                     # SMAK: indexqueue=len(newqueue)+1
    for i, r in enumerate(regions):
        rid = r.get("region_id", i)
        queue[start + i] = [SCAN_KIND,
                            smak_region(r, energy, f"{stem}_r{rid}", **fields)]

    payload = pickle.dumps(queue, protocol=protocol)
    with open(path, "wb") as fid:              # SMAK: open(fn,'wb') 11647-11651
        fid.write(HEADER)
        fid.write(payload)
    return queue


def read_smak_regions(path):
    """
    Parse a SMAK dataqueue region file the way smak.py 11597-11643 does:
    check the first line is 'DATASERVER REGIONS\\n', then re-open binary, skip
    one line and pickle.loads() the rest. Returns {int: ['image', region]}.

    The header check is done on bytes (accepting the \\r\\n form SMAK's
    universal-newline text read would also accept) so that files written by
    SMAK itself with a binary pickle do not hit its cp1252 decode bug.
    Like SMAK this uses plain pickle: only read files you trust.
    """
    with open(path, "rb") as fid:
        first = fid.readline()
        if first not in (HEADER, b"DATASERVER REGIONS\r\n"):
            raise ValueError(f"{path}: invalid parameter file (first line "
                             f"{first[:40]!r}, expected {HEADER!r})")
        queue = pickle.loads(fid.read())
    if not isinstance(queue, dict):
        raise ValueError(f"{path}: payload is {type(queue).__name__}, not a dict")
    return queue
