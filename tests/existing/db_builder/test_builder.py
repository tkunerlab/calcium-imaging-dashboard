import tempfile
import unittest
import os
from pathlib import Path

import h5py
import numpy as np


from calcium_imaging_dashboard.db_builder.builder import (
    _combined_stimulus_data,
    _publish_session,
    _table_stimulus_data,
    _validated_arrays,
    build,
    discover_sessions,
    find_analysis_target,
    mapped_sessions,
)
from calcium_imaging_dashboard.db_builder.loader import CaimanLoader, MinianLoader


class BuilderTests(unittest.TestCase):
    def test_builder_interface_exposes_min1pipe_and_not_cascade(self):
        frontend = (
            Path(__file__).parents[3]
            / "src"
            / "calcium_imaging_dashboard"
            / "db_builder"
            / "frontend"
            / "index.html"
        ).read_text(encoding="utf-8")
        self.assertIn("MIN1PIPE", frontend)
        self.assertNotIn("Cascade", frontend)

    @staticmethod
    def _write_caiman_result(path, with_quality):
        path.mkdir()
        with h5py.File(path / "caiman_results.hdf5", "w") as handle:
            estimates = handle.create_group("estimates")
            estimates.create_dataset("dims", data=np.array([2, 3]))
            footprints = np.arange(12, dtype=np.float64).reshape(6, 2)
            from scipy.sparse import csc_matrix
            sparse = csc_matrix(footprints)
            spatial = estimates.create_group("A")
            spatial.create_dataset("data", data=sparse.data)
            spatial.create_dataset("indices", data=sparse.indices)
            spatial.create_dataset("indptr", data=sparse.indptr)
            spatial.create_dataset("shape", data=np.array(sparse.shape))
            estimates.create_dataset("C", data=np.ones((2, 5)))
            estimates.create_dataset("b0", data=np.arange(6, dtype=np.float64))
            if with_quality:
                estimates.create_dataset("idx_components", data=np.array([0]))
                estimates.create_dataset("SNR_comp", data=np.array([4.0, np.inf]))

    def test_caiman_source_quality_is_optional_and_partially_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            result_dir = Path(directory) / "caiman-analysis"
            self._write_caiman_result(result_dir, with_quality=True)
            loaded = CaimanLoader().load(str(result_dir))
            self.assertEqual(
                set(loaded["source_quality"]), {"Accepted", "TemporalSNR"}
            )
            np.testing.assert_array_equal(
                loaded["source_quality"]["Accepted"], [1, -1]
            )
            self.assertEqual(loaded["source_quality"]["TemporalSNR"][0], 4.0)
            self.assertTrue(np.isnan(loaded["source_quality"]["TemporalSNR"][1]))

    def test_caiman_without_source_quality_omits_source_group(self):
        with tempfile.TemporaryDirectory() as directory:
            result_dir = Path(directory) / "caiman-analysis"
            self._write_caiman_result(result_dir, with_quality=False)
            loaded = CaimanLoader().load(str(result_dir))
            self.assertNotIn("source_quality", loaded)

    def test_build_writes_schema_version_1_2(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "input"
            analysis = root / "Cohort" / "Animal" / "Type" / "Session" / "caiman-analysis"
            analysis.parent.mkdir(parents=True)
            self._write_caiman_result(analysis, with_quality=False)
            output = Path(directory) / "database.h5"
            rules = [
                {"label": "CohortName"},
                {"label": "MouseName"},
                {"label": "SessionType"},
                {"label": "SessionNumber"},
            ]
            written = build(
                str(output),
                str(root),
                "caiman-analysis",
                rules,
                {},
                progress_callback=lambda _message: None,
                analysis_pattern="caiman-analysis/**/caiman_results.hdf5",
            )
            self.assertEqual(written, 1)
            with h5py.File(output, "r") as handle:
                self.assertEqual(handle["Database"].attrs["SchemaVersion"], "1.2")

    def test_recursive_analysis_pattern_selects_newest_match_per_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "input"
            session = root / "Cohort" / "Animal" / "Type" / "Session"
            old_result = session / "caiman-analysis" / "2026-01-01"
            new_result = session / "caiman-analysis" / "2026-02-01"
            old_result.mkdir(parents=True)
            new_result.mkdir(parents=True)
            old_file = old_result / "caiman_results.hdf5"
            new_file = new_result / "caiman_results.hdf5"
            old_file.touch()
            new_file.touch()
            os.utime(old_file, ns=(1_000_000_000, 1_000_000_000))
            os.utime(new_file, ns=(2_000_000_000, 2_000_000_000))

            selected, count = find_analysis_target(
                str(session),
                "caiman-analysis/**/caiman_results.hdf5",
            )
            self.assertEqual(Path(selected), new_file)
            self.assertEqual(count, 2)

            sessions = discover_sessions(
                str(root),
                "caiman-analysis",
                depth_rules=[{}, {}, {}, {}],
                analysis_pattern="caiman-analysis/**/caiman_results.hdf5",
            )
            self.assertEqual(len(sessions), 1)
            self.assertEqual(Path(sessions[0]["analysis_dir"]), new_file)
            self.assertEqual(Path(sessions[0]["session_dir"]), session)
            self.assertEqual(sessions[0]["analysis_match_count"], 2)

            root_session = discover_sessions(
                str(session),
                "caiman-analysis",
                depth_rules=[],
                analysis_pattern="caiman-analysis/**/caiman_results.hdf5",
            )
            self.assertEqual(root_session[0]["rel_parts"], [])
            self.assertEqual(Path(root_session[0]["analysis_dir"]), new_file)

    def test_minian_core_and_event_arrays_are_retained(self):
        import zarr

        with tempfile.TemporaryDirectory() as directory:
            analysis = Path(directory) / "minian-analysis"
            data_dir = analysis / "2026-01-01_run" / "data"
            data_dir.mkdir(parents=True)
            arrays = {
                "A": np.arange(12, dtype=float).reshape(2, 2, 3),
                "C": np.arange(10, dtype=float).reshape(2, 5),
                "max_proj": np.arange(6, dtype=float).reshape(2, 3),
                "S": np.arange(10, dtype=float).reshape(2, 5) + 100,
            }
            for name, values in arrays.items():
                group = zarr.open_group(str(data_dir / f"{name}.zarr"), mode="w")
                group.create_dataset(name, data=values)
            loaded = MinianLoader().load(str(analysis))
            np.testing.assert_array_equal(loaded["spatial"], arrays["A"])
            np.testing.assert_array_equal(loaded["temporal"], arrays["C"])
            np.testing.assert_array_equal(loaded["max_proj"], arrays["max_proj"])
            np.testing.assert_array_equal(
                loaded["deconvolved_events"], arrays["S"]
            )

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
                    {
                        "max_projection_source": "test:mip",
                        "deconvolved_events": temporal * 2,
                        "delta_f_over_f": temporal * 3,
                        "source_quality": {
                            "Accepted": np.array([1, -1], dtype=np.int8),
                            "TemporalSNR": np.array([3.0, np.nan]),
                        },
                    },
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
                self.assertIn("DeconvolvedEvents", calcium)
                self.assertIn("DeltaFOverF", calcium)
                self.assertEqual(
                    set(calcium["CellQuality"].keys()),
                    {"FootprintArea", "FootprintEccentricity", "TemporalContrast", "Source"},
                )
                np.testing.assert_array_equal(
                    calcium["CellQuality/Source/Accepted"][:], [1, -1]
                )
                self.assertNotIn("Sentinel", calcium)

    def test_csv_stimulus_rows_are_grouped_and_invalid_rows_warn(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stimulus.csv"
            path.write_text(
                "frame,stimulus\n1,Tone\n2,Light\n2,Tone\n0,Bad\n3.5,Bad\n",
                encoding="utf-8",
            )
            data, warnings = _table_stimulus_data(str(path))
            np.testing.assert_array_equal(data["Tone"], [1, 2])
            np.testing.assert_array_equal(data["Light"], [2])
            self.assertEqual(len(warnings), 2)

    def test_xlsx_stimulus_rows_are_grouped(self):
        from openpyxl import Workbook

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stimulus.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.append(["frame", "stimulus"])
            sheet.append([4, "AirPuff"])
            sheet.append([4, "Tone"])
            workbook.save(path)
            data, warnings = _table_stimulus_data(str(path))
            self.assertEqual(warnings, [])
            np.testing.assert_array_equal(data["AirPuff"], [4])
            np.testing.assert_array_equal(data["Tone"], [4])

    def test_combined_hdf5_stimulus_hierarchy_is_read(self):
        mapping = {
            "CohortName": "Cohort",
            "MouseName": "Animal",
            "SessionType": "Type",
            "SessionNumber": "Session",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stimulus.h5"
            with h5py.File(path, "w") as handle:
                group = handle.require_group(
                    "Database/Cohort/Animal/Type/Session/StimulusData"
                )
                group.create_dataset("Tone", data=np.array([1, 9]))
                group.create_dataset("Light", data=np.array([9]))
            data = _combined_stimulus_data(str(path), mapping)
            np.testing.assert_array_equal(data["Tone"], [1, 9])
            np.testing.assert_array_equal(data["Light"], [9])

    def test_combined_classic_mat_stimulus_hierarchy_is_read(self):
        from scipy.io import savemat

        mapping = {
            "CohortName": "Cohort",
            "MouseName": "Animal",
            "SessionType": "Type",
            "SessionNumber": "Session",
        }
        payload = {
            "Database": {
                "Cohort": {
                    "Animal": {
                        "Type": {
                            "Session": {
                                "StimulusData": {
                                    "Tone": np.array([2, 4]),
                                }
                            }
                        }
                    }
                }
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stimulus.mat"
            savemat(path, payload)
            data = _combined_stimulus_data(str(path), mapping)
            np.testing.assert_array_equal(data["Tone"], [2, 4])


if __name__ == "__main__":
    unittest.main()
