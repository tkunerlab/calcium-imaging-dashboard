"""Coordinated persistence for the dashboard working copy."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np


class SaveCoordinator:
    """Materialize cells, alignment, and matching from one workspace revision."""

    def __init__(self, database, workspace) -> None:
        self.database = database
        self.workspace = workspace

    @staticmethod
    def _matching_filename(database, mouse: str, cohort: str | None) -> str:
        prefix = f"{cohort}_{mouse}" if cohort else mouse
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", prefix)
        return os.path.join(database.var_dir, f"CellMatching_{safe}.mat")

    def save(self, snapshot) -> dict:
        database = self.database
        final_processed = database.processed_db_path
        source = database._working_read_path()
        output_dir = os.path.dirname(final_processed) or "."
        os.makedirs(output_dir, exist_ok=True)

        fd, staging_db = tempfile.mkstemp(
            prefix=f".{Path(final_processed).stem}.workspace.",
            suffix=Path(final_processed).suffix,
            dir=output_dir,
        )
        os.close(fd)
        shutil.copy2(source, staging_db)

        original_processed = database.processed_db_path
        original_active = database.active_db_path
        original_mode = database.view_mode
        matching_outputs: list[tuple[str, str]] = []
        saved_revision = snapshot.revision

        try:
            database.processed_db_path = staging_db
            database.active_db_path = staging_db
            database.view_mode = "working"

            database.save_workspace(
                mark_saved=False,
                refresh_metadata=False,
                payload=(snapshot.cells, snapshot.deleted),
            )

            for mouse, mouse_cache in snapshot.alignment.items():
                if not isinstance(mouse_cache, dict):
                    continue
                cohort = mouse_cache.get("__cohort__")
                sessions = database.get_sessions_for_mouse(mouse, cohort)
                for session in sessions:
                    name = session["display_name"]
                    cached = mouse_cache.get(name)
                    if not cached:
                        continue
                    database.save_aligned_warps(
                        session["cohort"],
                        session["mouse"],
                        session["session_type"],
                        session["session_name"],
                        cached.get(
                            "transform", np.eye(2, 3, dtype=np.float32)
                        ),
                        cached.get("mode", "translation"),
                        float(cached.get("dx", 0.0)),
                        float(cached.get("dy", 0.0)),
                        float(cached.get("rotation", 0.0)),
                        float(cached.get("scale", 1.0)),
                    )

            for mouse, cache in snapshot.matching.items():
                if not isinstance(cache, dict) or "matching_matrix" not in cache:
                    continue
                cohort = cache.get("cohort")
                final_match = self._matching_filename(database, mouse, cohort)
                match_dir = os.path.dirname(final_match) or "."
                fd, staging_match = tempfile.mkstemp(
                    prefix=f".{Path(final_match).stem}.", suffix=".mat", dir=match_dir
                )
                os.close(fd)
                database.save_matching_results(
                    mouse_name=mouse,
                    matching_matrix=cache["matching_matrix"],
                    session_names=cache["session_names"],
                    alignment_shifts=cache.get("alignment_shifts", {}),
                    master_centroids=cache["master_centroids"],
                    master_footprints=cache.get("master_footprints", []),
                    params=cache["params"],
                    cohort_name=cohort,
                    save_path=staging_match,
                    refresh_metadata=False,
                )
                matching_outputs.append((staging_match, final_match))

            with h5py.File(staging_db, "r") as handle:
                if database.db_var_name not in handle:
                    raise ValueError("Processed checkpoint is missing the database root.")

            os.replace(staging_db, final_processed)
            for staging_match, final_match in matching_outputs:
                os.replace(staging_match, final_match)

            database.processed_db_path = final_processed
            database.active_db_path = final_processed
            database.view_mode = "working"
            database.mouse_to_cohort = {}
            database.metadata = database._scan_metadata()
            saved_clean = self.workspace.mark_saved(snapshot.revision)
            return {
                "path": final_processed,
                "matching_paths": [path for _, path in matching_outputs],
                "revision": saved_revision,
                "saved_clean": saved_clean,
            }
        except Exception:
            for staging_match, _ in matching_outputs:
                if os.path.exists(staging_match):
                    os.remove(staging_match)
            if os.path.exists(staging_db):
                os.remove(staging_db)
            raise
        finally:
            database.processed_db_path = original_processed
            if database.active_db_path != final_processed:
                database.active_db_path = original_active
                database.view_mode = original_mode
