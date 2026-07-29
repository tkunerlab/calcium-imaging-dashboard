import unittest

import numpy as np


from calcium_imaging_dashboard.cell_registration_dashboard.quality import cell_quality_metrics, histogram


class CellQualityMetricTests(unittest.TestCase):
    def test_footprint_area_uses_relative_peak_threshold(self):
        spatial = np.array(
            [
                [[0.0, 1.0], [0.19, 0.21]],
                [[0.0, 10.0], [2.0, 2.01]],
            ]
        )
        temporal = np.ones((2, 3))
        metrics = cell_quality_metrics(spatial, temporal)
        np.testing.assert_equal(metrics["FootprintArea"], np.array([2.0, 2.0]))

    def test_temporal_contrast_uses_trace_standard_deviation(self):
        spatial = np.ones((2, 1, 1))
        temporal = np.array(
            [
                [0.0, 1.0, 0.0, 1.0, 0.0],
                [5.0, 5.0, 5.0, 5.0, 5.0],
            ]
        )
        metrics = cell_quality_metrics(spatial, temporal)
        np.testing.assert_allclose(
            metrics["TemporalContrast"],
            np.array([
                (np.percentile(temporal[0], 99.0) - np.median(temporal[0]))
                / np.std(temporal[0]),
                0.0,
            ]),
        )

    def test_eccentricity_uses_the_thresholded_support(self):
        spatial = np.zeros((2, 5, 5))
        spatial[0, 1:4, 1:4] = 1.0
        spatial[1, 2, 1:4] = 1.0
        metrics = cell_quality_metrics(spatial, np.ones((2, 3)))
        np.testing.assert_allclose(metrics["FootprintEccentricity"], [0.0, 1.0])

    def test_non_finite_and_degenerate_cells_receive_finite_values(self):
        spatial = np.array([[[np.nan]], [[np.inf]]])
        temporal = np.array([[np.nan, np.inf], [1.0, 1.0]])
        metrics = cell_quality_metrics(spatial, temporal)
        for values in metrics.values():
            self.assertTrue(np.all(np.isfinite(values)))
            np.testing.assert_equal(values, np.zeros(2))

    def test_histogram_handles_constant_values(self):
        counts, bins = histogram(np.array([3.0, 3.0, 3.0]))
        self.assertEqual(counts, [3])
        self.assertEqual(bins, [3.0])


if __name__ == "__main__":
    unittest.main()
