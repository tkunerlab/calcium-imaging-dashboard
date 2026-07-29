"""Cross-pipeline cell-quality metrics and compact distributions."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def cell_quality_metrics(
    spatial: np.ndarray,
    temporal: np.ndarray,
    *,
    footprint_peak_fraction: float = 0.20,
) -> Dict[str, np.ndarray]:
    """Return common area, eccentricity, and temporal-contrast arrays.

    Footprint support contains finite positive pixels above 20% (configurable)
    of the cell's maximum positive value. Eccentricity is derived from the
    spatial covariance of that binary support. Temporal Contrast is the trace's
    99th-percentile amplitude above its median divided by the trace's standard
    deviation. It measures how strongly a trace contains distinct high
    excursions without pretending to estimate acquisition noise.

    Degenerate inputs deliberately receive finite zero values so they remain
    usable by histograms and threshold controls.
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
    clean_spatial = np.where(np.isfinite(spatial), spatial, 0.0)
    peaks = np.max(clean_spatial, axis=(1, 2), initial=0.0)
    cutoffs = peaks[:, None, None] * footprint_peak_fraction
    supports = (clean_spatial > cutoffs) & (peaks[:, None, None] > 0.0)
    areas = np.count_nonzero(supports, axis=(1, 2)).astype(np.float64)

    eccentricity = np.zeros(spatial.shape[0], dtype=np.float64)
    for index, support in enumerate(supports):
        coordinates = np.argwhere(support)
        if coordinates.shape[0] < 2:
            continue
        covariance = np.cov(coordinates, rowvar=False, bias=True)
        eigenvalues = np.linalg.eigvalsh(covariance)
        major = float(max(eigenvalues[-1], 0.0))
        minor = float(max(eigenvalues[0], 0.0))
        if major > 0.0:
            eccentricity[index] = np.sqrt(max(0.0, 1.0 - minor / major))

    temporal_contrast = np.zeros(temporal.shape[0], dtype=np.float64)
    for index, trace in enumerate(temporal):
        finite_trace = trace[np.isfinite(trace)]
        if finite_trace.size < 2:
            continue
        median = float(np.median(finite_trace))
        amplitude = max(float(np.percentile(finite_trace, 99.0)) - median, 0.0)
        scale = float(np.std(finite_trace))
        if np.isfinite(scale) and scale > 0.0:
            temporal_contrast[index] = amplitude / scale

    return {
        "FootprintArea": areas,
        "FootprintEccentricity": eccentricity,
        "TemporalContrast": temporal_contrast,
    }


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
