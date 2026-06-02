"""
Utility for loading SSRL XRF HDF5 files produced by SMAK / USDC.

HDF5 structure:
    main/mapdata  : (ny, nx, n_channels)  float64
    main/xdata    : (nx,)                 float64  [mm]
    main/ydata    : (ny,)                 float64  [mm]
    main/attrs    : energy, labels, origin, ...
"""

import h5py
import numpy as np
from pathlib import Path


def load_xrf(filepath):
    """
    Load an SSRL XRF HDF5 file.

    Returns a dict with:
        mapdata : np.ndarray (ny, nx, n_channels)
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
