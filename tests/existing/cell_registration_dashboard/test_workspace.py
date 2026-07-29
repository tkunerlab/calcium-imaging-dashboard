import hashlib
from pathlib import Path

import h5py
import numpy as np

from calcium_imaging_dashboard.cell_registration_dashboard.database import CalciumImagingDatabase
from calcium_imaging_dashboard.cell_registration_dashboard.workspace import EditWorkspace


def _write_database(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        for session, offset in (("Session01", 0.0), ("Session02", 10.0)):
            group = handle.require_group(
                f"Database/CohortA/AnimalA/Training/{session}/CalciumData"
            )
            spatial = np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5) + offset
            temporal = np.arange(3 * 8, dtype=float).reshape(3, 8) + offset
            group.create_dataset("SpatialFootprints", data=spatial)
            group.create_dataset("TemporalFootprints", data=temporal)
            group.create_dataset("MaxProjection", data=np.max(spatial, axis=0))
            group.create_dataset("DeconvolvedEvents", data=temporal + 100)
            group.create_dataset("DeltaFOverF", data=temporal + 200)
            quality = group.require_group("CellQuality")
            quality.create_dataset("FootprintArea", data=np.array([1.0, 2.0, 3.0]))
            quality.create_dataset("TemporalContrast", data=np.array([4.0, 5.0, 6.0]))
            quality.create_dataset("FootprintEccentricity", data=np.array([0.1, 0.2, 0.3]))
            source = quality.require_group("Source")
            source.create_dataset("Accepted", data=np.array([1, 0, -1], dtype=np.int8))
            source.create_dataset("TemporalSNR", data=np.array([7.0, 8.0, 9.0]))


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _loader(database, key):
    def load():
        data = database.load_session_calcium_data(
            *key, warp_cached=False, include_workspace=False
        )
        return data["spatial_footprints"], data["temporal_footprints"]

    return load


def test_curated_paths_are_distinct_for_supported_extensions(tmp_path):
    for suffix in (".mat", ".h5", ".hdf5"):
        raw = tmp_path / f"calcium{suffix}"
        raw_path, processed_path = CalciumImagingDatabase._database_paths(str(raw))
        assert Path(raw_path) == raw
        assert Path(processed_path) == tmp_path / f"calcium_curated{suffix}"
        assert raw_path != processed_path

        restored_raw, restored_processed = CalciumImagingDatabase._database_paths(
            processed_path
        )
        assert Path(restored_raw) == raw
        assert restored_processed == processed_path


def test_processed_suffix_is_not_treated_as_an_application_checkpoint(tmp_path):
    supplied = tmp_path / "calcium_processed.mat"
    raw_path, curated_path = CalciumImagingDatabase._database_paths(str(supplied))
    assert Path(raw_path) == supplied
    assert Path(curated_path) == tmp_path / "calcium_processed_curated.mat"


def test_open_and_edit_do_not_create_or_change_processed_or_raw(tmp_path):
    raw = tmp_path / "calcium.h5"
    _write_database(raw)
    before = _digest(raw)

    database = CalciumImagingDatabase(str(raw))
    workspace = EditWorkspace()
    database.workspace = workspace
    key = ("CohortA", "AnimalA", "Training", "Session01")
    workspace.discard_indices({key: [1]}, {key: _loader(database, key)})

    assert _digest(raw) == before
    assert not Path(database.processed_db_path).exists()


def test_save_materializes_processed_and_preserves_raw(tmp_path):
    raw = tmp_path / "calcium.h5"
    _write_database(raw)
    before = _digest(raw)
    database = CalciumImagingDatabase(str(raw))
    workspace = EditWorkspace()
    database.workspace = workspace
    key = ("CohortA", "AnimalA", "Training", "Session01")

    workspace.discard_indices({key: [1]}, {key: _loader(database, key)})
    processed = Path(database.save_workspace())

    assert processed.exists()
    assert processed != raw
    assert _digest(raw) == before
    with h5py.File(raw, "r") as handle:
        assert handle["Database/CohortA/AnimalA/Training/Session01/CalciumData/SpatialFootprints"].shape[0] == 3
    with h5py.File(processed, "r") as handle:
        calcium = handle["Database/CohortA/AnimalA/Training/Session01/CalciumData"]
        assert calcium["SpatialFootprints"].shape[0] == 2
        np.testing.assert_array_equal(
            calcium["DeconvolvedEvents"][:, 0], [100.0, 116.0]
        )
        np.testing.assert_array_equal(
            calcium["DeltaFOverF"][:, 0], [200.0, 216.0]
        )
        np.testing.assert_array_equal(
            calcium["CellQuality/Source/Accepted"][:], [1, -1]
        )
        np.testing.assert_array_equal(
            calcium["CellQuality/FootprintArea"][:], [1.0, 3.0]
        )
    assert not workspace.status()["dirty"]


def test_merge_uses_max_envelope_and_invalidates_native_quality(tmp_path):
    raw = tmp_path / "calcium.h5"
    _write_database(raw)
    database = CalciumImagingDatabase(str(raw))
    workspace = EditWorkspace()
    database.workspace = workspace
    key = ("CohortA", "AnimalA", "Training", "Session01")
    loader = _loader(database, key)

    workspace.merge_indices(key, [0, 1], loader)
    curated = Path(database.save_workspace())

    with h5py.File(curated, "r") as handle:
        calcium = handle["Database/CohortA/AnimalA/Training/Session01/CalciumData"]
        np.testing.assert_array_equal(
            calcium["DeconvolvedEvents"][0], np.maximum(
                np.arange(8) + 100, np.arange(8, 16) + 100
            )
        )
        np.testing.assert_array_equal(
            calcium["DeltaFOverF"][0], np.maximum(
                np.arange(8) + 200, np.arange(8, 16) + 200
            )
        )
        assert calcium["CellQuality/Source/Accepted"][0] == -1
        assert np.isnan(calcium["CellQuality/Source/TemporalSNR"][0])
        for name in ("FootprintArea", "TemporalContrast", "FootprintEccentricity"):
            values = calcium[f"CellQuality/{name}"][:]
            assert values.shape == (2,)
            assert np.all(np.isfinite(values))
        assert calcium["CellQuality/FootprintArea"][1] == 3.0


def test_matching_reorders_and_pads_optional_cell_arrays(tmp_path):
    raw = tmp_path / "calcium.h5"
    _write_database(raw)
    database = CalciumImagingDatabase(str(raw))
    matching = np.array([[2.0], [0.0], [np.nan]])
    footprints = [
        {"idx": [0], "vals": [1.0], "norm": 1.0}
        for _ in range(3)
    ]
    database.save_matching_results(
        "AnimalA",
        matching,
        ["Training_Session01"],
        {},
        np.zeros((3, 2)),
        footprints,
        {
            "max_dist": 10.0,
            "min_overlap": 0.1,
            "cost_weight": 0.5,
            "overlap_type": "cosine",
        },
        cohort_name="CohortA",
        save_path=str(tmp_path / "matching.mat"),
    )
    with h5py.File(database.processed_db_path, "r") as handle:
        calcium = handle["Database/CohortA/AnimalA/Training/Session01/CalciumData"]
        assert calcium["DeconvolvedEvents"][0, 0] == 116.0
        assert calcium["DeconvolvedEvents"][1, 0] == 100.0
        assert np.isnan(calcium["DeconvolvedEvents"][2, 0])
        assert calcium["DeltaFOverF"][0, 0] == 216.0
        assert np.isnan(calcium["DeltaFOverF"][2, 0])
        np.testing.assert_array_equal(
            calcium["CellQuality/Source/Accepted"][:], [-1, 1, -1]
        )


def test_overview_command_undoes_all_sessions_atomically(tmp_path):
    raw = tmp_path / "calcium.h5"
    _write_database(raw)
    database = CalciumImagingDatabase(str(raw))
    workspace = EditWorkspace()
    database.workspace = workspace
    first = ("CohortA", "AnimalA", "Training", "Session01")
    second = ("CohortA", "AnimalA", "Training", "Session02")
    loaders = {first: _loader(database, first), second: _loader(database, second)}

    workspace.discard_indices(
        {first: [0], second: [2]}, loaders, label="Overview discard"
    )
    assert workspace.overlay(first, *loaders[first]())[0].shape[0] == 2
    assert workspace.overlay(second, *loaders[second]())[0].shape[0] == 2

    assert workspace.undo() == "Overview discard"
    assert workspace.overlay(first, *loaders[first]())[0].shape[0] == 3
    assert workspace.overlay(second, *loaders[second]())[0].shape[0] == 3

    assert workspace.redo() == "Overview discard"
    assert workspace.overlay(first, *loaders[first]())[0].shape[0] == 2
    assert workspace.overlay(second, *loaders[second]())[0].shape[0] == 2
