"""Tests for ROI utilities (utils.roi_utils), incl. the vectorized label_rois."""

import numpy as np

from utils.roi_utils import label_rois, threshold_map, sample_mask, _thresh_otsu


def test_label_rois_removes_small_regions():
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:4, 1:4] = True        # 9-pixel blob (kept)
    mask[7, 7] = True            # 1-pixel blob (dropped at min_pixels=2)
    labeled, n = label_rois(mask, min_pixels=2)
    assert n == 1
    assert labeled[7, 7] == 0
    assert (labeled[1:4, 1:4] > 0).all()


def test_label_rois_min1_keeps_all():
    mask = np.zeros((10, 10), dtype=bool)
    mask[0, 0] = True
    mask[5, 5] = True
    _, n = label_rois(mask, min_pixels=1)
    assert n == 2


def test_label_rois_matches_reference_loop():
    # The vectorized bincount path must match a brute-force size filter.
    rng = np.random.default_rng(3)
    mask = rng.random((30, 30)) > 0.5
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    keep = np.array([(lab == i).sum() >= 3 for i in range(1, n + 1)])
    ref_kept = sum(keep)
    _, n_vec = label_rois(mask, min_pixels=3)
    assert n_vec == ref_kept


def test_threshold_map_flags_bright_roi():
    # threshold_map finds bright outliers ABOVE a low background (ROI detection),
    # not a 50/50 split. A small bright block on a low floor should be flagged.
    m = np.full((10, 10), 5.0)
    m[2:4, 2:4] = 500.0
    mask, thr, _ = threshold_map(m, method="auto", k=1.0)
    assert mask[2:4, 2:4].all()
    assert mask.sum() == 4


def test_sample_mask_restrict_confines():
    comp = np.full((20, 20), 100.0)
    comp[:5, :] = 0.0
    restrict = np.zeros((20, 20), dtype=bool)
    restrict[:, :10] = True
    m, _ = sample_mask(comp, kernel_px=1, restrict_to=restrict)
    assert not m[:, 10:].any()              # nothing selected outside the restriction


def test_otsu_threshold_between_modes():
    x = np.concatenate([np.zeros(100), np.full(100, 100.0)])
    t = _thresh_otsu(x)
    assert 0 < t < 100
