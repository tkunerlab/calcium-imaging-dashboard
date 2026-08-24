from pathlib import Path

import h5py
import numpy as np

from calcium_imaging_dashboard.cell_registration_dashboard.database import (
    CalciumImagingDatabase,
)
from calcium_imaging_dashboard.cell_registration_dashboard.workspace import EditWorkspace


SESSION_KEY = ("Cohort", "Animal", "Training", "Session01")
CALCIUM_PATH = "Database/Cohort/Animal/Training/Session01/CalciumData"


def _write_compressed_database(path: Path) -> tuple[int, int]:
    spatial = np.zeros((24, 96, 96), dtype=np.float32)
    temporal = np.zeros((24, 1000), dtype=np.float32)
    for index in range(spatial.shape[0]):
        spatial[index, index : index + 4, index : index + 4] = index + 1
        temporal[index, index * 10 : index * 10 + 20] = index + 1

    with h5py.File(path, "w") as handle:
        group = handle.require_group(CALCIUM_PATH)
        for name, values, chunks in (
            ("SpatialFootprints", spatial, (1, 96, 96)),
            ("TemporalFootprints", temporal, (24, 1000)),
            ("MaxProjection", np.max(spatial, axis=0), (96, 96)),
        ):
            group.create_dataset(
                name,
                data=values,
                chunks=chunks,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
    return spatial.nbytes, temporal.nbytes


def _loader(database):
    def load():
        data = database.load_session_calcium_data(
            *SESSION_KEY, warp_cached=False, include_workspace=False
        )
        return data["spatial_footprints"], data["temporal_footprints"]

    return load


def _assert_compact_float32_datasets(path: Path) -> None:
    with h5py.File(path, "r") as handle:
        group = handle[CALCIUM_PATH]
        for name in ("SpatialFootprints", "TemporalFootprints"):
            dataset = group[name]
            assert dataset.dtype == np.dtype("float32")
            assert dataset.compression == "gzip"
            assert dataset.compression_opts == 4
            assert dataset.shuffle
            assert dataset.chunks is not None


def test_workspace_saves_keep_large_arrays_compressed_and_compact(tmp_path):
    raw = tmp_path / "calcium.h5"
    spatial_bytes, temporal_bytes = _write_compressed_database(raw)
    database = CalciumImagingDatabase(str(raw))
    workspace = EditWorkspace()
    database.workspace = workspace
    loader = _loader(database)

    workspace.discard_indices({SESSION_KEY: [1]}, {SESSION_KEY: loader})
    curated = Path(database.save_workspace())

    _assert_compact_float32_datasets(curated)
    dense_payload_size = spatial_bytes + temporal_bytes
    assert curated.stat().st_size < dense_payload_size // 5

    first_save_size = curated.stat().st_size
    workspace.discard_indices({SESSION_KEY: [1]}, {SESSION_KEY: loader})
    database.save_workspace()

    _assert_compact_float32_datasets(curated)
    assert curated.stat().st_size < dense_payload_size // 5
    assert curated.stat().st_size < first_save_size * 2


def test_alignment_rewrite_keeps_projection_and_footprints_compressed(tmp_path):
    raw = tmp_path / "calcium.h5"
    _write_compressed_database(raw)
    database = CalciumImagingDatabase(str(raw))

    database.save_aligned_warps(
        *SESSION_KEY,
        np.eye(2, 3, dtype=np.float32),
        "rigid",
        nudge_angle=1.0,
    )

    with h5py.File(database.processed_db_path, "r") as handle:
        group = handle[CALCIUM_PATH]
        for name in ("SpatialFootprints", "MaxProjection"):
            dataset = group[name]
            assert dataset.dtype == np.dtype("float32")
            assert dataset.compression == "gzip"
            assert dataset.compression_opts == 4
            assert dataset.shuffle
