import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np


from calcium_imaging_dashboard.db_builder.builder import (
    _publish_session,
    _validated_arrays,
    mapped_sessions,
)


class BuilderTests(unittest.TestCase):
    def test_duplicate_mappings_are_reported_before_build(self):
        sessions = [
            {"rel_parts": ["one"], "analysis_dir": "/source/one"},
            {"rel_parts": ["two"], "analysis_dir": "/source/two"},
        ]
        globals_ = {
            "CohortName": "Cohort",
            "MouseName": "Animal",
            "SessionType": "Type",
            "SessionNumber": "Session",
        }
        _, collisions = mapped_sessions(sessions, [], globals_)
        self.assertEqual(len(collisions), 1)
        self.assertEqual(collisions[0]["destination"], "Cohort/Animal/Type/Session")

    def test_shape_validation_rejects_cell_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, "Cell count mismatch"):
            _validated_arrays(
                {
                    "spatial": np.zeros((2, 10, 10)),
                    "temporal": np.zeros((3, 20)),
                    "max_proj": np.zeros((10, 10)),
                }
            )

    def test_staged_publication_replaces_complete_session(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "builder.h5"
            with h5py.File(path, "w") as handle:
                parent = handle.require_group("Database/Cohort/Animal/Type")
                old = parent.require_group("Session/CalciumData")
                old.create_dataset("Sentinel", data=np.array([1]))

                spatial = np.ones((2, 8, 6), dtype=np.float64)
                temporal = np.ones((2, 12), dtype=np.float64)
                mip = np.ones((8, 6), dtype=np.float64)
                _publish_session(
                    handle,
                    parent,
                    "Database/Cohort/Animal/Type",
                    "Session",
                    {"max_projection_source": "test:mip"},
                    spatial,
                    temporal,
                    mip,
                    "Animal/Type/Session/caiman-analysis",
                    "caiman-analysis",
                    "float32",
                    "gzip",
                    1,
                )

                self.assertFalse(any(name.startswith(".__") for name in parent))
                session = parent["Session"]
                calcium = session["CalciumData"]
                self.assertEqual(calcium["SpatialFootprints"].dtype, np.dtype("float32"))
                self.assertEqual(calcium["SpatialFootprints"].compression, "gzip")
                self.assertEqual(session.attrs["MaxProjectionSource"], "test:mip")
                self.assertNotIn("Sentinel", calcium)


if __name__ == "__main__":
    unittest.main()
