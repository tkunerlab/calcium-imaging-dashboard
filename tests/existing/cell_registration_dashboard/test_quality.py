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
        areas, _ = cell_quality_metrics(spatial, temporal)
        np.testing.assert_equal(areas, np.array([2.0, 2.0]))

    def test_temporal_activity_is_positive_baseline_corrected_auc_per_frame(self):
        spatial = np.ones((2, 1, 1))
        temporal = np.array(
            [
                [0.0, 0.0, 2.0, 2.0],
                [5.0, 5.0, 5.0, 5.0],
            ]
        )
        _, activity = cell_quality_metrics(spatial, temporal)
        np.testing.assert_allclose(activity, np.array([1.0, 0.0]))

    def test_histogram_handles_constant_values(self):
        counts, bins = histogram(np.array([3.0, 3.0, 3.0]))
        self.assertEqual(counts, [3])
        self.assertEqual(bins, [3.0])


if __name__ == "__main__":
    unittest.main()
