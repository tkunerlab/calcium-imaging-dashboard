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


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _loader(database, key):
    def load():
        data = database.load_session_calcium_data(
            *key, warp_cached=False, include_workspace=False
        )
        return data["spatial_footprints"], data["temporal_footprints"]

    return load


def test_processed_paths_are_distinct_for_supported_extensions(tmp_path):
    for suffix in (".mat", ".h5", ".hdf5"):
        raw = tmp_path / f"calcium{suffix}"
        raw_path, processed_path = CalciumImagingDatabase._database_paths(str(raw))
        assert Path(raw_path) == raw
        assert Path(processed_path) == tmp_path / f"calcium_processed{suffix}"
        assert raw_path != processed_path

        restored_raw, restored_processed = CalciumImagingDatabase._database_paths(
            processed_path
        )
        assert Path(restored_raw) == raw
        assert restored_processed == processed_path


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
        assert handle["Database/CohortA/AnimalA/Training/Session01/CalciumData/SpatialFootprints"].shape[0] == 2
    assert not workspace.status()["dirty"]


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
