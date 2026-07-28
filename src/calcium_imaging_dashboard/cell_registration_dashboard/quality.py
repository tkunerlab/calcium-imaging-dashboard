"""Cell-quality metrics used by automatic cleaning and its distributions."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def cell_quality_metrics(
    spatial: np.ndarray,
    temporal: np.ndarray,
    *,
    footprint_peak_fraction: float = 0.20,
    baseline_percentile: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return footprint area and normalized positive temporal AUC per cell.

    Footprint area is the number of pixels above ``footprint_peak_fraction`` of
    that cell's peak. Temporal activity is positive area above the cell's
    ``baseline_percentile`` baseline, divided by trace length so sessions with
    different frame counts remain comparable.
    """
    spatial = np.asarray(spatial, dtype=np.float64)
    temporal = np.asarray(temporal, dtype=np.float64)
    if spatial.ndim != 3:
        raise ValueError("Spatial footprints must have shape (cells, height, width).")
    if temporal.ndim != 2:
        raise ValueError("Temporal footprints must have shape (cells, frames).")
    if spatial.shape[0] != temporal.shape[0]:
        raise ValueError("Spatial and temporal footprints must have the same cell count.")
    if not 0.0 < footprint_peak_fraction <= 1.0:
        raise ValueError("footprint_peak_fraction must be in (0, 1].")
    if not 0.0 <= baseline_percentile <= 100.0:
        raise ValueError("baseline_percentile must be in [0, 100].")

    clean_spatial = np.nan_to_num(spatial, nan=0.0, posinf=0.0, neginf=0.0)
    peaks = np.max(clean_spatial, axis=(1, 2), initial=0.0)
    cutoffs = peaks[:, None, None] * footprint_peak_fraction
    areas = np.count_nonzero(
        (clean_spatial > cutoffs) & (peaks[:, None, None] > 0.0),
        axis=(1, 2),
    ).astype(np.float64)

    if temporal.shape[1] == 0:
        activity = np.zeros(temporal.shape[0], dtype=np.float64)
    else:
        clean_temporal = np.nan_to_num(temporal, nan=0.0, posinf=0.0, neginf=0.0)
        baselines = np.percentile(
            clean_temporal, baseline_percentile, axis=1, keepdims=True
        )
        activity = np.sum(
            np.maximum(clean_temporal - baselines, 0.0), axis=1
        ) / temporal.shape[1]
    return areas, activity


def histogram(values: np.ndarray, *, max_bins: int = 40) -> Tuple[list, list]:
    """Build compact adaptive histogram counts and bin centers."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [], []
    if np.all(values == values[0]):
        return [int(values.size)], [float(values[0])]
    bins = min(max_bins, max(5, int(np.ceil(np.sqrt(values.size)))))
    counts, edges = np.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return counts.astype(int).tolist(), centers.astype(float).tolist()
