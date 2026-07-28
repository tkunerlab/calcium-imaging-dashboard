import unittest
import hashlib
import tempfile
from pathlib import Path

import h5py
import numpy as np

from calcium_imaging_dashboard.cell_registration_dashboard.alignment_models import resolve_alignment_reference
from calcium_imaging_dashboard.cell_registration_dashboard.matching import delete_master_cell_groups, summarize_matching
from calcium_imaging_dashboard.cell_registration_dashboard.database import CalciumImagingDatabase
from calcium_imaging_dashboard.cell_registration_dashboard.save_coordinator import SaveCoordinator
from calcium_imaging_dashboard.cell_registration_dashboard.workspace import EditWorkspace


class AlignmentReferenceTests(unittest.TestCase):
    def test_direct_always_uses_selected_root(self):
        sessions = list(range(7))
        for active in range(len(sessions)):
            reference = resolve_alignment_reference(sessions, active, 4, "Direct")
            self.assertEqual(reference.reference_index, 4)

    def test_sequential_crawls_toward_root_on_both_sides(self):
        sessions = list(range(7))
        expected = [1, 2, 3, 4, 4, 4, 5]
        actual = [
            resolve_alignment_reference(sessions, active, 4, "Sequential").reference_index
            for active in range(len(sessions))
        ]
        self.assertEqual(actual, expected)


class MatchingSummaryTests(unittest.TestCase):
    def test_coverage_is_fraction_of_tracks_seen_in_multiple_sessions(self):
        matrix = np.array(
            [
                [0.0, 1.0, np.nan],
                [1.0, np.nan, np.nan],
                [2.0, 3.0, 2.0],
                [np.nan, 4.0, np.nan],
            ]
        )
        summary = summarize_matching(matrix)
        self.assertEqual(summary["n_master_cells"], 4)
        self.assertEqual(summary["n_matched_cells"], 2)
        self.assertEqual(summary["n_unmatched_cells"], 2)
        self.assertEqual(summary["coverage_pct"], 50.0)

    def test_delete_master_group_remaps_remaining_session_indices(self):
        matrix = np.array(
            [
                [0.0, 0.0, np.nan],
                [1.0, np.nan, 0.0],
                [2.0, 1.0, 1.0],
            ]
        )
        centroids = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        reduced, reduced_centroids, footprints, deleted = delete_master_cell_groups(
            matrix, centroids, ["a", "b", "c"], [0]
        )
        np.testing.assert_equal(
            reduced,
            np.array([[0.0, np.nan, 0.0], [1.0, 0.0, 1.0]]),
        )
        np.testing.assert_equal(reduced_centroids, centroids[1:])
        self.assertEqual(footprints, ["b", "c"])
        self.assertEqual(deleted, [[0], [0], []])


class WorkspaceDomainHistoryTests(unittest.TestCase):
    def setUp(self):
        self.workspace = EditWorkspace()

    def test_alignment_change_invalidates_and_undo_restores_matching(self):
        self.workspace.update_alignments(
            "Animal",
            "Cohort",
            {"Type_Session01": {"dx": 1.0, "dy": 0.0}},
        )
        self.workspace.replace_matching(
            "Animal",
            {"matching_matrix": np.array([[0.0, 0.0]])},
        )
        self.workspace.mark_saved()

        self.workspace.update_alignments(
            "Animal",
            "Cohort",
            {"Type_Session01": {"dx": 4.0, "dy": -2.0}},
        )
        self.assertNotIn("Animal", self.workspace.matching_state)
        self.assertEqual(
            self.workspace.alignment_state["Animal"]["Type_Session01"]["dx"],
            4.0,
        )

        self.assertEqual(self.workspace.undo(), "Adjust alignment")
        self.assertIn("Animal", self.workspace.matching_state)
        self.assertEqual(
            self.workspace.alignment_state["Animal"]["Type_Session01"]["dx"],
            1.0,
        )
        self.assertEqual(self.workspace.redo(), "Adjust alignment")
        self.assertNotIn("Animal", self.workspace.matching_state)

    def test_matching_replacement_is_undoable(self):
        first = {"matching_matrix": np.array([[0.0, 0.0]])}
        second = {"matching_matrix": np.array([[0.0, np.nan], [np.nan, 1.0]])}
        self.workspace.replace_matching("Animal", first, label="First matching")
        self.workspace.mark_saved()
        self.workspace.replace_matching("Animal", second, label="Second matching")

        self.assertEqual(self.workspace.undo(), "Second matching")
        np.testing.assert_equal(
            self.workspace.matching_state["Animal"]["matching_matrix"],
            first["matching_matrix"],
        )
        self.assertEqual(self.workspace.redo(), "Second matching")
        np.testing.assert_equal(
            self.workspace.matching_state["Animal"]["matching_matrix"],
            second["matching_matrix"],
        )


class SaveCoordinatorTests(unittest.TestCase):
    def test_session_summary_uses_unsaved_workspace_cell_count(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "calcium.h5"
            with h5py.File(raw, "w") as handle:
                group = handle.require_group(
                    "Database/Cohort/Animal/Training/Session01/CalciumData"
                )
                spatial = np.arange(36.0).reshape(3, 3, 4)
                group.create_dataset("SpatialFootprints", data=spatial)
                group.create_dataset("TemporalFootprints", data=np.ones((3, 5)))
                group.create_dataset("MaxProjection", data=np.max(spatial, axis=0))

            database = CalciumImagingDatabase(str(raw))
            workspace = EditWorkspace()
            database.workspace = workspace
            key = ("Cohort", "Animal", "Training", "Session01")

            def loader():
                data = database.load_session_calcium_data(
                    *key, warp_cached=False, include_workspace=False
                )
                return data["spatial_footprints"], data["temporal_footprints"]

            workspace.discard_indices({key: [2]}, {key: loader})
            self.assertEqual(database.load_session_summary(*key)["n_cells"], 2)
            workspace.merge_indices(key, [0, 1], loader)
            self.assertEqual(database.load_session_summary(*key)["n_cells"], 1)

    def test_cell_checkpoint_preserves_raw_database(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "calcium.h5"
            with h5py.File(raw, "w") as handle:
                group = handle.require_group(
                    "Database/Cohort/Animal/Training/Session01/CalciumData"
                )
                spatial = np.arange(24.0).reshape(2, 3, 4)
                group.create_dataset("SpatialFootprints", data=spatial)
                group.create_dataset("TemporalFootprints", data=np.ones((2, 5)))
                group.create_dataset("MaxProjection", data=np.max(spatial, axis=0))
            raw_digest = hashlib.sha256(raw.read_bytes()).hexdigest()

            database = CalciumImagingDatabase(str(raw))
            workspace = EditWorkspace()
            database.workspace = workspace
            key = ("Cohort", "Animal", "Training", "Session01")

            def loader():
                data = database.load_session_calcium_data(
                    *key, warp_cached=False, include_workspace=False
                )
                return data["spatial_footprints"], data["temporal_footprints"]

            workspace.discard_indices({key: [1]}, {key: loader})
            saved = SaveCoordinator(database, workspace).save(workspace.save_snapshot())

            self.assertEqual(hashlib.sha256(raw.read_bytes()).hexdigest(), raw_digest)
            self.assertEqual(Path(saved["path"]), Path(database.processed_db_path))
            with h5py.File(saved["path"], "r") as handle:
                dataset = handle[
                    "Database/Cohort/Animal/Training/Session01/CalciumData/SpatialFootprints"
                ]
                self.assertEqual(dataset.shape[0], 1)
            self.assertFalse(workspace.status()["dirty"])


if __name__ == "__main__":
    unittest.main()
