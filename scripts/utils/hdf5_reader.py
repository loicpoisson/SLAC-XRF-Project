"""
Utility for loading SSRL XRF HDF5 files produced by SMAK / USDC.

HDF5 structure:
    main/mapdata  : (ny, nx, 2 + n_labels) float64
                    planes 0/1 = stage X/Y positions, plane i+2 = labels[i]
                    (SMAK itself reads channels as labels.index(ch) + 2)
    main/xdata    : (nx,)                 float64  [mm]
    main/ydata    : (ny,)                 float64  [mm]
    main/attrs    : energy, labels, origin, ...

The TIME channel is the real per-pixel integration time. In these fly scans
it is ~65x nominal in column 0 and 1.2-4x along the first/last columns
(acceleration ramps), and every count channel scales with it (OCR tracks
TIME), so load_xrf rescales count channels to the median dwell by default.
"""

import h5py
import numpy as np
from pathlib import Path


# Channels that are not photon counts: never time-normalised.
_NO_TIME_NORM = {"I0", "I1", "TIME", "DTF", "DTPCT"}


def load_xrf(filepath, normalize_time=True):
    """
    Load an SSRL XRF HDF5 file.

    normalize_time : rescale count channels by median(TIME) / TIME so every
                     pixel is expressed at the nominal dwell (see module doc).

    Returns a dict with:
        mapdata : np.ndarray (ny, nx, n_labels)  plane i = labels[i]
        xdata   : np.ndarray (nx,)  real X coordinates [mm]
        ydata   : np.ndarray (ny,)  real Y coordinates [mm]
        labels  : list[str]         channel names (e.g. 'Fe.Ka', 'Au.La')
        energy  : float             beam energy [eV]
        dx      : float             pixel size X [mm]
        dy      : float             pixel size Y [mm]
        filepath: Path
    """
    filepath = Path(filepath)
    with h5py.File(filepath, "r") as f:
        mapdata = f["main/mapdata"][:]
        xdata   = f["main/xdata"][:]
        ydata   = f["main/ydata"][:]
        attrs   = dict(f["main"].attrs)

    labels = [s.decode() if isinstance(s, bytes) else s
              for s in attrs.get("labels", [])]
    energy = float(attrs.get("energy", 0.0))

    if mapdata.ndim != 3 or not labels:
        raise ValueError(f"{filepath.name}: not a processed SMAK map "
                         f"(mapdata shape {mapdata.shape}, {len(labels)} labels)")
    if mapdata.shape[:2] != (len(ydata), len(xdata)):
        raise ValueError(f"{filepath.name}: mapdata {mapdata.shape[:2]} does not "
                         f"match (ny, nx) = ({len(ydata)}, {len(xdata)})")
    offset = mapdata.shape[2] - len(labels)
    if offset not in (0, 2):
        raise ValueError(f"{filepath.name}: {mapdata.shape[2]} planes for "
                         f"{len(labels)} labels")
    mapdata = mapdata[:, :, offset:].astype(float)

    if normalize_time and "TIME" in labels:
        t = mapdata[:, :, labels.index("TIME")]
        t_ref = np.median(t[t > 0]) if np.any(t > 0) else 0.0
        if t_ref > 0:
            scale = np.where(t > 0, t_ref / np.where(t > 0, t, 1.0), 0.0)
            for i, lab in enumerate(labels):
                if lab not in _NO_TIME_NORM:
                    mapdata[:, :, i] *= scale

    dx = float(xdata[1] - xdata[0]) if len(xdata) > 1 else 1.0
    dy = float(ydata[1] - ydata[0]) if len(ydata) > 1 else 1.0

    return {
        "mapdata":  mapdata,
        "xdata":    xdata,
        "ydata":    ydata,
        "labels":   labels,
        "energy":   energy,
        "dx":       abs(dx),
        "dy":       abs(dy),
        "filepath": filepath,
    }


def get_channel(data, name):
    """
    Extract one element map by channel name (e.g. 'Fe.Ka').
    Returns a 2D array (ny, nx).
    """
    try:
        idx = data["labels"].index(name)
    except ValueError:
        raise ValueError(f"Channel '{name}' not found. Available: {data['labels']}")
    return data["mapdata"][:, :, idx]


def list_element_channels(data):
    """Return only the XRF element channels (exclude monitors, diagnostics)."""
    skip = {"I0", "I1", "I0ZEBRA", "I1ZEBRA", "ICR", "OCR",
            "TIME", "DTF", "DTPCT", "LASER"}
    return [l for l in data["labels"] if l not in skip]


def get_composite_map(data, channels=None):
    """
    Build a 2D intensity map by summing over element channels.

    Channels that are constant across the image (floor-value, no signal)
    are automatically excluded — they carry no spatial information.

    Parameters
    ----------
    channels : list[str] or None
        Element names to include. None = all element channels.

    Returns
    -------
    composite : 2D array (ny, nx)
    used      : list[str]  channels actually summed (after excluding constants)
    excluded  : list[str]  channels dropped because they were constant
    """
    if channels is None:
        candidates = list_element_channels(data)
    else:
        available = list_element_channels(data)
        for ch in channels:
            if ch not in data["labels"]:
                raise ValueError(f"Channel '{ch}' not found. Available: {available}")
        candidates = channels

    used, excluded = [], []
    for ch in candidates:
        arr = data["mapdata"][:, :, data["labels"].index(ch)]
        if arr.min() == arr.max():          # constant = floor value, no information
            excluded.append(ch)
        else:
            used.append(ch)

    if not used:
        raise ValueError("All requested channels are constant (floor value). "
                         "No signal to work with.")

    indices = [data["labels"].index(ch) for ch in used]
    composite = data["mapdata"][:, :, indices].sum(axis=2)
    return composite, used, excluded
