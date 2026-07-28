import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np


from calcium_imaging_dashboard.cell_registration_dashboard.alignment import (
    compose_displacement_fields,
    compose_warp_matrix_from_params,
    compute_alignment_nccs,
    compute_centroids,
)
from calcium_imaging_dashboard.cell_registration_dashboard.database import CalciumImagingDatabase


class AlignmentRegressionTests(unittest.TestCase):
    def test_sequential_dual_ncc_scores_immediate_neighbor_not_root(self):
        root = np.zeros((20, 20), dtype=np.float32)
        root[2:5, 2:5] = 1.0
        neighbor = np.zeros_like(root)
        neighbor[8:12, 8:12] = 1.0
        target = neighbor.copy()
        identity = np.eye(2, 3, dtype=np.float32)
        scores = compute_alignment_nccs(
            [root, neighbor, target],
            [root, neighbor, target],
            active_index=2,
            reference_index=1,
            mode="translation",
            active_transform=identity,
            reference_transform=identity,
            downsample=False,
        )
        self.assertAlmostEqual(scores["mip_ncc"], 1.0, places=6)
        self.assertAlmostEqual(scores["footprints_ncc"], 1.0, places=6)
        root_correlation = float(np.corrcoef(root.ravel(), target.ravel())[0, 1])
        self.assertLess(abs(root_correlation), 0.2)

    def test_similarity_matrix_uses_supplied_fov_center(self):
        matrix = compose_warp_matrix_from_params(
            7.0, -4.0, 0.0, 1.0, cx=20.0, cy=10.0
        )
        np.testing.assert_allclose(matrix[:, 2], [7.0, -4.0], atol=1e-6)

    def test_constant_backward_displacements_compose(self):
        first = np.zeros((12, 15, 2), dtype=np.float32)
        second = np.zeros_like(first)
        first[..., 0] = 2.0
        first[..., 1] = -1.0
        second[..., 0] = 3.0
        second[..., 1] = 4.0
        composed = compose_displacement_fields(first, second)
        # Interior pixels do not touch the remap border.
        np.testing.assert_allclose(composed[4:-4, 4:-4, 0], 5.0, atol=1e-6)
        np.testing.assert_allclose(composed[4:-4, 4:-4, 1], 3.0, atol=1e-6)

    def test_saved_translation_keeps_same_forward_sign_after_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shift.h5"
            spatial = np.zeros((1, 20, 20), dtype=np.float64)
            spatial[0, 8, 8] = 1.0
            with h5py.File(path, "w") as handle:
                group = handle.require_group(
                    "Database/Cohort/Animal/Type/Session/CalciumData"
                )
                group.create_dataset(
                    "SpatialFootprints", data=np.transpose(spatial, (0, 2, 1))
                )
                group.create_dataset("TemporalFootprints", data=np.ones((1, 5)))
                group.create_dataset("MaxProjection", data=spatial[0].T)
                group.create_dataset("AlignmentShift", data=np.array([[3.0], [-2.0]]))

            database = CalciumImagingDatabase(str(path))
            unwarped = database.load_session_calcium_data(
                "Cohort", "Animal", "Type", "Session", warp_cached=False
            )["spatial_footprints"]
            warped = database.load_session_calcium_data(
                "Cohort", "Animal", "Type", "Session", warp_cached=True
            )["spatial_footprints"]
            before = compute_centroids(unwarped)[0]
            after = compute_centroids(warped)[0]
            np.testing.assert_allclose(after - before, [3.0, -2.0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
