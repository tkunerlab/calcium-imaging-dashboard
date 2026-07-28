import os
import shutil
import h5py
import numpy as np
import scipy.io
import tempfile
import re
from pathlib import Path
from threading import RLock

class CalciumImagingDatabase:
    def __init__(self, db_path=None):
        if db_path:
            db_path = db_path.replace("\\", "/")
            if os.path.isdir(db_path):
                raise IsADirectoryError(
                    "A database file is required; select a .mat, .h5, or .hdf5 file."
                )
            self.db_path = db_path
            self.var_dir = os.path.dirname(self.db_path)
            self.raw_db_path, self.processed_db_path = self._database_paths(self.db_path)
        else:
            self.var_dir = ""
            self.raw_db_path = ""
            self.processed_db_path = ""
            
        self.view_mode = "working"
        self.active_db_path = self._working_read_path()
        self.db_var_name = "Database"
        self.workspace = None
        self._save_lock = RLock()
        
        # Discover mice and sessions using lazy traversal
        self.mouse_to_cohort = {}
        self.metadata = {}
        if os.path.exists(self._read_path()):
            self.metadata = self._scan_metadata()

    @staticmethod
    def _database_paths(db_path):
        path = Path(db_path)
        suffix = path.suffix
        stem = path.stem
        if stem.endswith("_processed"):
            raw = path.with_name(f"{stem[:-10]}{suffix}")
            processed = path
        else:
            raw = path
            processed = path.with_name(f"{stem}_processed{suffix}")
        return str(raw), str(processed)

    def _working_read_path(self):
        return self.processed_db_path if os.path.exists(self.processed_db_path) else self.raw_db_path

    def _read_path(self):
        if self.view_mode == "raw":
            return self.raw_db_path
        return self._working_read_path()

    def set_view_mode(self, mode):
        self.view_mode = "raw" if mode == "raw" else "working"
        self.active_db_path = self._read_path()
        self.mouse_to_cohort = {}
        self.metadata = self._scan_metadata() if os.path.exists(self.active_db_path) else {}

    def _prepare_processed_write(self):
        """Return the only path that mutable operations are allowed to open."""
        self._ensure_processed_db_copy()
        self.view_mode = "working"
        self.active_db_path = self.processed_db_path
        return self.processed_db_path

    def _ensure_processed_db_copy(self):
        """Ensures that [DatabaseName]_processed.mat exists as a copy of [DatabaseName].mat."""
        if not os.path.exists(self.processed_db_path):
            if not os.path.exists(self.raw_db_path):
                raise FileNotFoundError(
                    f"Raw database mat file not found at {self.raw_db_path}."
                )
            print(f"Creating a copy of database -> {self.processed_db_path}...")
            shutil.copy2(self.raw_db_path, self.processed_db_path)
            print("Copy created successfully.")

    def _scan_metadata(self):
        """Crawl the HDF5 group structure to build a list of cohorts, mice, and sessions."""
        metadata = {}
        with h5py.File(self._read_path(), 'r') as f:
            # Prefer the current schema name, while remaining compatible with a
            # single unambiguous non-system root from earlier databases.
            var_names = [k for k in f.keys() if not k.startswith('#')]
            if not var_names:
                raise ValueError("No valid database root found in the HDF5 file.")
            if "Database" in var_names:
                self.db_var_name = "Database"
            elif len(var_names) == 1:
                self.db_var_name = var_names[0]
            else:
                raise ValueError(
                    "Multiple database roots were found. Rename the intended root "
                    "to 'Database' or provide a file with one non-system root."
                )
            
            db_grp = f[self.db_var_name]
            
            # Dynamically discover cohort names (keys in root variable excluding subsystems)
            cohorts = [k for k in db_grp.keys() if not k.startswith('#') and k not in ['Metadata', 'CellMatching']]
            
            for cohort_name in cohorts:
                cohort_grp = db_grp[cohort_name]
                metadata[cohort_name] = {}
                
                for mouse_name in cohort_grp.keys():
                    if mouse_name.startswith('#') or mouse_name in ['Metadata', 'CellMatching']:
                        continue
                    
                      # Check if we have standard group keys (ignore internal HDF5/MATLAB subsystem keys)
                    mouse_grp = cohort_grp[mouse_name]
                    self.mouse_to_cohort[mouse_name] = cohort_name
                    
                    has_calcium = False
                    sessions_dict = {}
                    
                    for session_type in mouse_grp.keys():
                        if session_type in ['Metadata', 'CellMatching']:
                            continue
                        
                        stype_grp = mouse_grp[session_type]
                        sessions_dict[session_type] = []
                        
                        for sess_name in stype_grp.keys():
                            sess_grp = stype_grp[sess_name]
                            
                            # Check if CalciumData is present with SpatialFootprints
                            has_cal = False
                            n_cells = 0
                            if 'CalciumData' in sess_grp:
                                cal_grp = sess_grp['CalciumData']
                                if 'SpatialFootprints' in cal_grp:
                                    # SpatialFootprints is of shape (N, W, H) in HDF5. 
                                    # If 2D (shape is W x H), there is only 1 cell.
                                    ds_shape = cal_grp['SpatialFootprints'].shape
                                    if len(ds_shape) == 2:
                                        n_cells = 1
                                    else:
                                        n_cells = ds_shape[0]
                                    if n_cells > 0:
                                        has_cal = True
                                        has_calcium = True
                                        
                            sessions_dict[session_type].append({
                                "name": sess_name,
                                "has_calcium": has_cal,
                                "n_cells": n_cells
                            })
                            
                    # Only add mice with valid calcium data to our active dashboard lists
                    if has_calcium:
                        metadata[cohort_name][mouse_name] = sessions_dict
                        
        return metadata

    def get_mice_list(self):
        """Returns sorted list of all mice that possess calcium data."""
        mice = set()
        for cohort_name, cohort_mice in self.metadata.items():
            mice.update(cohort_mice.keys())
        return sorted(mice)

    def get_sessions_for_mouse(self, mouse_name, cohort_name=None):
        """Returns a chronological ordered list of all calcium sessions for a given mouse."""
        cohort = cohort_name or self.mouse_to_cohort.get(mouse_name)
        if not cohort or mouse_name not in self.metadata[cohort]:
            return []
        
        sessions_list = []
        mouse_data = self.metadata[cohort][mouse_name]
        
        # Sort session types and session numbers to ensure chronological sequence
        for stype in sorted(mouse_data.keys()):
            # Find and sort sessions under this session type
            for sess in sorted(mouse_data[stype], key=lambda x: x["name"]):
                if sess["has_calcium"]:
                    item = {
                        "cohort": cohort,
                        "mouse": mouse_name,
                        "session_type": stype,
                        "session_name": sess["name"],
                        "n_cells": sess["n_cells"],
                        "display_name": f"{stype}_{sess['name']}"
                    }
                    key = (cohort, mouse_name, stype, sess["name"])
                    if self.workspace is None or not self.workspace.is_deleted(key):
                        sessions_list.append(item)
        return sessions_list

    def load_session_calcium_data(
        self,
        cohort,
        mouse,
        session_type,
        session,
        cached_alignment=None,
        warp_cached=True,
        include_workspace=True,
        include_temporal=True,
    ):
        """Loads specific session calcium imaging variables on-demand, optionally applying alignment on-the-fly."""
        data = {}
        with h5py.File(self._read_path(), 'r') as f:
            cal_path = f"{self.db_var_name}/{cohort}/{mouse}/{session_type}/{session}/CalciumData"
            if cal_path not in f:
                raise ValueError(f"CalciumData not found at {cal_path}")
            
            cal_grp = f[cal_path]
            
            # 1. MaxProjection: (W, H) in HDF5 -> (H, W) in NumPy
            mip = cal_grp['MaxProjection'][:]
            mip = np.transpose(mip)
            
            # 2. SpatialFootprints: (N, W, H) in HDF5 -> (N, H, W) in NumPy
            sf = cal_grp['SpatialFootprints'][:]
            if sf.ndim == 2:
                # Handle single-cell 2D matrices gracefully
                sf = np.expand_dims(sf, axis=0)
            sf = np.transpose(sf, (0, 2, 1))
            
            # 3. TemporalFootprints: (N, T) in HDF5 -> (N, T) in NumPy
            tf = None
            if include_temporal:
                tf = cal_grp['TemporalFootprints'][:]
                if tf.ndim == 1:
                    # Handle single-cell 1D traces gracefully
                    tf = np.expand_dims(tf, axis=0)

            key = (cohort, mouse, session_type, session)
            if include_workspace and self.workspace is not None and self.view_mode != "raw":
                arrays = self.workspace.arrays_if_loaded(key)
                if arrays is not None:
                    sf = arrays[0]
                    if include_temporal:
                        tf = arrays[1]
            if include_temporal:
                data['temporal_footprints'] = tf
            
            # 4. AlignmentShift: [dx, dy]
            dx, dy = 0.0, 0.0
            if 'AlignmentShift' in cal_grp:
                shift_ds = cal_grp['AlignmentShift'][:]
                # Handle MATLAB [1, 2] or HDF5 (2, 1) shape
                if shift_ds.ndim == 2:
                    if shift_ds.shape[0] == 2:
                        dx = float(shift_ds[0, 0])
                        dy = float(shift_ds[1, 0])
                    else:
                        dx = float(shift_ds[0, 0])
                        dy = float(shift_ds[0, 1])
                elif shift_ds.ndim == 1:
                    dx = float(shift_ds[0])
                    dy = float(shift_ds[1])
            data['alignment_shift'] = [dx, dy]
            
            # Apply warp on-the-fly if warp_cached is True
            if warp_cached:
                if cached_alignment is not None:
                    # Apply cached alignment (including manual nudges)
                    mode = cached_alignment["mode"]
                    dx_align = cached_alignment["dx"]
                    dy_align = cached_alignment["dy"]
                    rotation = cached_alignment.get("rotation", 0.0)
                    scale = cached_alignment.get("scale", 1.0)
                    
                    from .alignment import warp_image_rigid, warp_footprints_rigid, warp_image_non_rigid, warp_footprints_non_rigid
                    
                    if mode == 'non-rigid':
                        displacement = np.copy(cached_alignment["transform"])
                        displacement[:, :, 0] -= dx_align
                        displacement[:, :, 1] -= dy_align
                        mip = warp_image_non_rigid(mip, displacement)
                        sf = warp_footprints_non_rigid(sf, displacement)
                    else:
                        from .alignment import compose_warp_matrix_from_params
                        warp_mat = compose_warp_matrix_from_params(dx_align, dy_align, rotation, scale, cx=sf.shape[2]/2.0, cy=sf.shape[1]/2.0)
                        mip = warp_image_rigid(mip, warp_mat)
                        sf = warp_footprints_rigid(sf, warp_mat)
                        
                    data['alignment_shift'] = [dx_align, dy_align] # return the active shifts
                elif dx != 0.0 or dy != 0.0:
                    # AlignmentShift stores the forward content movement that
                    # places this session into reference coordinates.
                    from .alignment import compose_warp_matrix_from_params, warp_image_rigid, warp_footprints_rigid
                    warp_mat = compose_warp_matrix_from_params(
                        dx, dy, 0.0, 1.0, cx=sf.shape[2] / 2.0, cy=sf.shape[1] / 2.0
                    )
                    mip = warp_image_rigid(mip, warp_mat)
                    sf = warp_footprints_rigid(sf, warp_mat)
                    
            data['max_projection'] = mip.tolist()
            data['spatial_footprints'] = sf
            
        return data

    def load_session_summary(self, cohort, mouse, session_type, session):
        """Read count and alignment metadata without loading imaging stacks."""
        cal_path = f"{self.db_var_name}/{cohort}/{mouse}/{session_type}/{session}/CalciumData"
        with h5py.File(self._read_path(), "r") as f:
            cal_grp = f[cal_path]
            tf = cal_grp["TemporalFootprints"]
            n_cells = int(cal_grp.attrs.get("ActiveCellCount", tf.shape[0]))
            dx = dy = 0.0
            if "AlignmentShift" in cal_grp:
                shift = np.asarray(cal_grp["AlignmentShift"][:]).reshape(-1)
                if shift.size >= 2:
                    dx, dy = float(shift[0]), float(shift[1])

        key = (cohort, mouse, session_type, session)
        if self.workspace is not None and self.view_mode != "raw":
            arrays = self.workspace.arrays_if_loaded(key)
            if arrays is not None:
                n_cells = int(arrays[0].shape[0])
        return {"n_cells": n_cells, "alignment_shift": [dx, dy]}

    def load_temporal_rows(self, cohort, mouse, session_type, session, indices):
        """Load only requested trace rows, including staged workspace edits."""
        key = (cohort, mouse, session_type, session)
        if self.workspace is not None and self.view_mode != "raw":
            arrays = self.workspace.arrays_if_loaded(key)
            if arrays is not None:
                temporal = arrays[1]
                return [temporal[index] if 0 <= index < temporal.shape[0] else None for index in indices]

        cal_path = f"{self.db_var_name}/{cohort}/{mouse}/{session_type}/{session}/CalciumData"
        rows = []
        with h5py.File(self._read_path(), "r") as f:
            dataset = f[f"{cal_path}/TemporalFootprints"]
            for index in indices:
                rows.append(dataset[index, :] if 0 <= index < dataset.shape[0] else None)
        return rows

    def load_session_preview(self, cohort, mouse, session_type, session):
        """Load only MIP and spatial sum for Overview/alignment thumbnails."""
        cal_path = f"{self.db_var_name}/{cohort}/{mouse}/{session_type}/{session}/CalciumData"
        with h5py.File(self._read_path(), "r") as f:
            cal_grp = f[cal_path]
            mip = np.transpose(cal_grp["MaxProjection"][:])
            spatial = cal_grp["SpatialFootprints"][:]
            if spatial.ndim == 2:
                spatial = spatial[None, :, :]
            spatial = np.transpose(spatial, (0, 2, 1))
        key = (cohort, mouse, session_type, session)
        if self.workspace is not None and self.view_mode != "raw":
            arrays = self.workspace.arrays_if_loaded(key)
            if arrays is not None:
                spatial = arrays[0]
        return mip, np.sum(spatial, axis=0)

    def save_workspace(self, mark_saved=True, refresh_metadata=True, payload=None):
        """Atomically materialize staged workspace edits into processed output."""
        if self.workspace is None:
            return self.processed_db_path
        changes, deleted = payload if payload is not None else self.workspace.save_payload()
        if not changes and not deleted:
            return self.processed_db_path if os.path.exists(self.processed_db_path) else None

        with self._save_lock:
            source = self._working_read_path()
            output = self.processed_db_path
            output_dir = os.path.dirname(output) or "."
            os.makedirs(output_dir, exist_ok=True)
            fd, temp_path = tempfile.mkstemp(
                prefix=f".{Path(output).stem}.", suffix=Path(output).suffix, dir=output_dir
            )
            os.close(fd)
            try:
                shutil.copy2(source, temp_path)
                with h5py.File(temp_path, "r+") as f:
                    for key in deleted:
                        cohort, mouse, session_type, session = key
                        path = f"{self.db_var_name}/{cohort}/{mouse}/{session_type}/{session}"
                        if path in f:
                            del f[path]
                    for key, (spatial, temporal) in changes.items():
                        cohort, mouse, session_type, session = key
                        cal_path = f"{self.db_var_name}/{cohort}/{mouse}/{session_type}/{session}/CalciumData"
                        if cal_path not in f:
                            raise ValueError(f"CalciumData not found at {cal_path}")
                        cal_grp = f[cal_path]
                        sf_stored = np.transpose(spatial, (0, 2, 1)) if spatial.ndim == 3 else spatial
                        for name, array in (
                            ("SpatialFootprints", sf_stored),
                            ("TemporalFootprints", temporal),
                        ):
                            if name in cal_grp:
                                del cal_grp[name]
                            ds = cal_grp.create_dataset(name, data=array, dtype="float64")
                            ds.attrs["MATLAB_class"] = b"double"
                            ds.attrs["H5PATH"] = f"/{cal_path.replace('/', '')}".encode("utf-8")
                    f.flush()
                os.replace(temp_path, output)
            except Exception:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise

            if mark_saved:
                self.workspace.mark_saved()
            self.view_mode = "working"
            self.active_db_path = output
            if refresh_metadata:
                self.mouse_to_cohort = {}
                self.metadata = self._scan_metadata()
            return output

    def save_alignment_shifts(self, shifts_dict):
        """Saves alignment shifts [dx, dy] in-place for multiple sessions.
        
        shifts_dict format:
        {
            "Experimental/Mouse56/Reward100pct/Session03": [dx, dy],
            ...
        }
        """
        with h5py.File(self._prepare_processed_write(), 'r+') as f:
            for path_suffix, shift in shifts_dict.items():
                dx, dy = shift
                cal_path = f"{self.db_var_name}/{path_suffix}/CalciumData"
                if cal_path not in f:
                    continue
                
                cal_grp = f[cal_path]
                shift_path = f"{cal_path}/AlignmentShift"
                
                # Delete existing dataset to allow recreation (handles sizing/metadata cleanly)
                if 'AlignmentShift' in cal_grp:
                    del cal_grp['AlignmentShift']
                
                # Create a 2x1 dataset matching MATLAB's column-major matrix transpose
                shift_data = np.array([[dx], [dy]], dtype='float64')
                ds = f.create_dataset(shift_path, data=shift_data, dtype='float64')
                ds.attrs['MATLAB_class'] = b'double'
                ds.attrs['H5PATH'] = f"/{cal_path.replace('/', '')}".encode('utf-8')

    def clean_session_cells(self, cohort, mouse, session_type, session, keep_indices):
        """Discards cells outside the mask by slicing Spatial and Temporal Footprints in-place."""
        with h5py.File(self._prepare_processed_write(), 'r+') as f:
            cal_path = f"{self.db_var_name}/{cohort}/{mouse}/{session_type}/{session}/CalciumData"
            if cal_path not in f:
                return
            
            cal_grp = f[cal_path]
            
            # Load, slice, and rewrite SpatialFootprints (N, W, H)
            sf_path = f"{cal_path}/SpatialFootprints"
            sf = cal_grp['SpatialFootprints'][:]
            sf_sliced = sf[keep_indices, :, :]
            
            del cal_grp['SpatialFootprints']
            ds_sf = f.create_dataset(sf_path, data=sf_sliced, dtype='float64')
            ds_sf.attrs['MATLAB_class'] = b'double'
            ds_sf.attrs['H5PATH'] = f"/{cal_path.replace('/', '')}".encode('utf-8')
            
            # Load, slice, and rewrite TemporalFootprints (N, T)
            tf_path = f"{cal_path}/TemporalFootprints"
            tf = cal_grp['TemporalFootprints'][:]
            tf_sliced = tf[keep_indices, :]
            
            del cal_grp['TemporalFootprints']
            ds_tf = f.create_dataset(tf_path, data=tf_sliced, dtype='float64')
            ds_tf.attrs['MATLAB_class'] = b'double'
            ds_tf.attrs['H5PATH'] = f"/{cal_path.replace('/', '')}".encode('utf-8')
            
        # Update our cached session list counts
        self.metadata = self._scan_metadata()
        return len(keep_indices)

    def merge_session_cells(self, cohort, mouse, session_type, session, cell_indices):
        """Merges multiple cells in a session. Replaces the first index with mean spatial footprint
        and max envelope temporal trace, and deletes/discards the remaining cell indices."""
        if len(cell_indices) < 2:
            return
            
        with h5py.File(self._prepare_processed_write(), 'r+') as f:
            cal_path = f"{self.db_var_name}/{cohort}/{mouse}/{session_type}/{session}/CalciumData"
            if cal_path not in f:
                return
            
            cal_grp = f[cal_path]
            
            # Load arrays
            sf_path = f"{cal_path}/SpatialFootprints"
            sf = cal_grp['SpatialFootprints'][:]
            
            tf_path = f"{cal_path}/TemporalFootprints"
            tf = cal_grp['TemporalFootprints'][:]
            
            # Identify keep index and delete indices
            keep_idx = cell_indices[0]
            delete_idxs = cell_indices[1:]
            
            # Compute merged spatial footprint (average across selected, normalized to max 1.0)
            merged_spatial = np.mean(sf[cell_indices, :, :], axis=0)
            max_val = np.max(merged_spatial)
            if max_val > 0:
                merged_spatial = merged_spatial / max_val
                
            # Compute merged temporal trace (maximum envelope)
            merged_temporal = np.max(tf[cell_indices, :], axis=0)
            
            # Apply merged values to keep_idx
            sf[keep_idx, :, :] = merged_spatial
            tf[keep_idx, :] = merged_temporal
            
            # Create sliced keep indices (excluding delete_idxs)
            keep_indices = [i for i in range(sf.shape[0]) if i not in delete_idxs]
            
            sf_sliced = sf[keep_indices, :, :]
            tf_sliced = tf[keep_indices, :]
            
            # Save sliced back to file
            del cal_grp['SpatialFootprints']
            ds_sf = f.create_dataset(sf_path, data=sf_sliced, dtype='float64')
            ds_sf.attrs['MATLAB_class'] = b'double'
            ds_sf.attrs['H5PATH'] = f"/{cal_path.replace('/', '')}".encode('utf-8')
            
            del cal_grp['TemporalFootprints']
            ds_tf = f.create_dataset(tf_path, data=tf_sliced, dtype='float64')
            ds_tf.attrs['MATLAB_class'] = b'double'
            ds_tf.attrs['H5PATH'] = f"/{cal_path.replace('/', '')}".encode('utf-8')
            
        # Re-scan metadata structure
        self.metadata = self._scan_metadata()
        return len(keep_indices)

    def auto_merge_duplicates(self, cohort, mouse, session_type, session, dist_thresh, corr_thresh, overlap_thresh):
        """Auto-merges duplicate cells in a single session based on centroid distance, temporal correlation, and spatial overlap."""
        from .alignment import compute_centroids
        from .matching import get_sparse_footprints, compute_overlap
        with h5py.File(self._prepare_processed_write(), 'r+') as f:
            cal_path = f"{self.db_var_name}/{cohort}/{mouse}/{session_type}/{session}/CalciumData"
            if cal_path not in f:
                return 0
            
            cal_grp = f[cal_path]
            
            # Load arrays
            sf_path = f"{cal_path}/SpatialFootprints"
            sf = cal_grp['SpatialFootprints'][:] # Shape (N, W, H)
            
            # Note: transpose to (N, H, W) for get_sparse_footprints if stored as (N, W, H) in H5
            # get_sparse_footprints expects (N, H, W) shape, let's keep shape as loaded
            
            tf_path = f"{cal_path}/TemporalFootprints"
            tf = cal_grp['TemporalFootprints'][:] # Shape (N, T)
            
            n_cells = sf.shape[0]
            if n_cells < 2:
                return 0
                
            # 1. Compute Centroids
            centroids = compute_centroids(sf) # Shape (N, 2)
            
            # 2. Compute pairwise distances
            dx = centroids[:, 0:1] - centroids[:, 0:1].T
            dy = centroids[:, 1:2] - centroids[:, 1:2].T
            dist_mat = np.sqrt(dx*dx + dy*dy)
            
            # 3. Compute pairwise correlation of temporal traces
            tf_means = np.mean(tf, axis=1, keepdims=True)
            tf_stds = np.std(tf, axis=1, keepdims=True)
            # Avoid division by zero
            tf_stds[tf_stds == 0] = 1.0
            tf_normalized = (tf - tf_means) / tf_stds
            corr_mat = np.dot(tf_normalized, tf_normalized.T) / tf.shape[1]
            
            # 3b. Compute pairwise spatial overlap (using sparse footprints)
            sfs = get_sparse_footprints(sf)
            overlap_mat = np.zeros((n_cells, n_cells))
            for i in range(n_cells):
                for j in range(i + 1, n_cells):
                    val = compute_overlap(sfs[i], sfs[j], 'cosine')
                    overlap_mat[i, j] = val
                    overlap_mat[j, i] = val
            
            # 4. Build adjacency matrix
            adj = (dist_mat <= dist_thresh) & (corr_mat >= corr_thresh) & (overlap_mat >= overlap_thresh)
            np.fill_diagonal(adj, False) # remove self-connections
            
            # 5. Find connected components (BFS queue logic)
            visited = np.zeros(n_cells, dtype=bool)
            components = []
            for i in range(n_cells):
                if visited[i]:
                    continue
                comp = []
                queue = [i]
                visited[i] = True
                while queue:
                    curr = queue.pop(0)
                    comp.append(curr)
                    neighbors = np.where(adj[curr])[0]
                    for n in neighbors:
                        if not visited[n]:
                            visited[n] = True
                            queue.append(n)
                if len(comp) > 1:
                    components.append(comp)
                    
            if not components:
                return 0
                
            # Perform merge components
            cells_to_keep = np.ones(n_cells, dtype=bool)
            total_merged = 0
            
            for comp in components:
                keep_idx = comp[0]
                delete_idxs = comp[1:]
                cells_to_keep[delete_idxs] = False
                
                # Average spatial footprint
                merged_spatial = np.mean(sf[comp, :, :], axis=0)
                max_val = np.max(merged_spatial)
                if max_val > 0:
                    merged_spatial = merged_spatial / max_val
                sf[keep_idx, :, :] = merged_spatial
                
                # Max temporal trace
                merged_temporal = np.max(tf[comp, :], axis=0)
                tf[keep_idx, :] = merged_temporal
                
                total_merged += len(delete_idxs)
                
            # Slice and drop deleted cells
            sf_sliced = sf[cells_to_keep, :, :]
            tf_sliced = tf[cells_to_keep, :]
            
            # Save back to HDF5
            del cal_grp['SpatialFootprints']
            ds_sf = f.create_dataset(sf_path, data=sf_sliced, dtype='float64')
            ds_sf.attrs['MATLAB_class'] = b'double'
            ds_sf.attrs['H5PATH'] = f"/{cal_path.replace('/', '')}".encode('utf-8')
            
            del cal_grp['TemporalFootprints']
            ds_tf = f.create_dataset(tf_path, data=tf_sliced, dtype='float64')
            ds_tf.attrs['MATLAB_class'] = b'double'
            ds_tf.attrs['H5PATH'] = f"/{cal_path.replace('/', '')}".encode('utf-8')
            
        # Re-scan metadata structure
        self.metadata = self._scan_metadata()
        return total_merged

    def save_matching_results(self, mouse_name, matching_matrix, session_names, alignment_shifts, master_centroids, master_footprints, params, cohort_name=None, save_path=None, refresh_metadata=True):
        """Aligns session footprints/traces to master IDs in-place, and saves matching metadata."""
        cohort = cohort_name or self.mouse_to_cohort[mouse_name]
        
        # 1. Update [DatabaseName]_processed.mat in-place: reorder and align cell footprints/traces
        S = len(session_names)
        M = matching_matrix.shape[0]
        identity_by_display = {
            session_info["display_name"]: session_info
            for session_info in self.get_sessions_for_mouse(mouse_name, cohort)
        }
        
        print(f"Aligning and resizing session footprint arrays to match {M} master cell tracks...")
        
        with h5py.File(self._prepare_processed_write(), 'r+') as f:
            for s in range(S):
                sess_name = session_names[s]
                session_info = identity_by_display.get(sess_name)
                if session_info is None:
                    raise ValueError(f"Matching session is no longer present: {sess_name}")
                session_type = session_info["session_type"]
                session = session_info["session_name"]
                
                cal_path = f"{self.db_var_name}/{cohort}/{mouse_name}/{session_type}/{session}/CalciumData"
                if cal_path not in f:
                    continue
                
                cal_grp = f[cal_path]
                
                # Load current footprints & traces
                sf_orig = cal_grp['SpatialFootprints'][:] # Shape: (N_orig, W, H)
                tf_orig = cal_grp['TemporalFootprints'][:] # Shape: (N_orig, T)
                
                N_orig, W, H = sf_orig.shape
                T = tf_orig.shape[1]
                
                # Preallocate aligned/resized arrays
                # Footprints initialized to 0, Traces initialized to NaN
                sf_aligned = np.zeros((M, W, H), dtype='float64')
                tf_aligned = np.full((M, T), np.nan, dtype='float64')
                
                # Map matching indices
                for i in range(M):
                    local_idx = matching_matrix[i, s]
                    # MATLAB is 1-indexed, so NaN in python matching matrix is parsed as NaN.
                    # We store it in python as NaN, or 0-indexed integer.
                    if not np.isnan(local_idx):
                        idx_0 = int(local_idx) # Convert to 0-indexed Python integer
                        if 0 <= idx_0 < N_orig:
                            sf_aligned[i, :, :] = sf_orig[idx_0, :, :]
                            tf_aligned[i, :] = tf_orig[idx_0, :]

                cal_grp.attrs["ActiveCellCount"] = int(np.sum(~np.isnan(matching_matrix[:, s])))
                
                # Delete and rewrite SpatialFootprints
                sf_path = f"{cal_path}/SpatialFootprints"
                del cal_grp['SpatialFootprints']
                ds_sf = f.create_dataset(sf_path, data=sf_aligned, dtype='float64')
                ds_sf.attrs['MATLAB_class'] = b'double'
                ds_sf.attrs['H5PATH'] = f"/{cal_path.replace('/', '')}".encode('utf-8')
                
                # Delete and rewrite TemporalFootprints
                tf_path = f"{cal_path}/TemporalFootprints"
                del cal_grp['TemporalFootprints']
                ds_tf = f.create_dataset(tf_path, data=tf_aligned, dtype='float64')
                ds_tf.attrs['MATLAB_class'] = b'double'
                ds_tf.attrs['H5PATH'] = f"/{cal_path.replace('/', '')}".encode('utf-8')
                
        # 2. Save independent CellMatching_MouseName.mat using standard scipy.io.savemat (v7 format)
        prefix = f"{cohort}_{mouse_name}" if cohort_name else mouse_name
        safe_prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", prefix)
        save_file_name = f"CellMatching_{safe_prefix}.mat"
        save_path = save_path or os.path.join(self.var_dir, save_file_name)
        
        # Prepare shifts struct dictionary for MATLAB loading
        shifts_dict = {}
        for s_name, shift_coords in alignment_shifts.items():
            # MATLAB expects 1x2 double array
            shifts_dict[s_name] = np.array([shift_coords], dtype='float64')
            
        # Re-encode sparse footprints list of dicts for MATLAB representation
        # MATLAB cell arrays of structs are represented as a numpy object array in scipy.io.savemat
        master_footprints_encoded = np.empty((M,), dtype=object)
        for i in range(M):
            fp = master_footprints[i]
            # Convert 1-based indexing for index arrays to match MATLAB's output exactly
            master_footprints_encoded[i] = {
                'idx': np.array([[val] for val in fp['idx']], dtype='uint32'),
                'vals': np.array([[val] for val in fp['vals']], dtype='float64'),
                'norm': float(fp['norm'])
            }
            
        cell_matching_struct = {
            "MouseName": mouse_name,
            "MatchingMatrix": matching_matrix + 1.0, # Convert back to MATLAB 1-based indices (with NaNs preserved)
            "SessionNames": np.array(session_names, dtype=object),
            "AlignmentShifts": shifts_dict,
            "MasterCentroids": master_centroids,
            "MasterFootprints": master_footprints_encoded,
            "Parameters": {
                "MaxCentroidDistance": float(params["max_dist"]),
                "MinSpatialOverlap": float(params["min_overlap"]),
                "CostWeight": float(params["cost_weight"]),
                "OverlapType": params["overlap_type"]
            }
        }
        
        scipy.io.savemat(save_path, {"CellMatching": cell_matching_struct})
        print(f"Saved independent cell matching MAT file successfully to: {save_path}")
        
        # Reload metadata
        if refresh_metadata:
            self.metadata = self._scan_metadata()
        return save_path

    def save_aligned_warps(self, cohort, mouse, session_type, session, warp_matrix_or_displacement, mode, dx=0.0, dy=0.0, nudge_angle=0.0, nudge_scale=1.0):
        """Applies rigid/non-rigid warps and saves MaxProjection/SpatialFootprints directly to HDF5."""
        from .alignment import warp_image_rigid, warp_footprints_rigid, warp_image_non_rigid, warp_footprints_non_rigid
        
        cal_path = f"{self.db_var_name}/{cohort}/{mouse}/{session_type}/{session}/CalciumData"
        
        with h5py.File(self._prepare_processed_write(), 'r+') as f:
            if cal_path not in f:
                return
            
            cal_grp = f[cal_path]
            
            if mode == 'translation' and nudge_angle == 0.0 and nudge_scale == 1.0:
                # Just save the shift dx, dy (positive target -> ref)
                dx_backend = dx
                dy_backend = dy
                if 'AlignmentShift' in cal_grp:
                    del cal_grp['AlignmentShift']
                shift_data = np.array([[dx_backend], [dy_backend]], dtype='float64')
                ds = f.create_dataset(f"{cal_path}/AlignmentShift", data=shift_data, dtype='float64')
                ds.attrs['MATLAB_class'] = b'double'
                ds.attrs['H5PATH'] = f"/{cal_path.replace('/', '')}".encode('utf-8')
            else:
                # Load current projection and footprints
                mip = cal_grp['MaxProjection'][:]
                mip = np.transpose(mip) # Shape: (H, W)
                
                sf = cal_grp['SpatialFootprints'][:]
                sf = np.transpose(sf, (0, 2, 1)) # Shape: (N, H, W)
                
                if mode == 'non-rigid':
                    # warp_matrix_or_displacement is displacement of shape (H, W, 2)
                    displacement = np.copy(warp_matrix_or_displacement)
                    # Subtract manual nudge canvas translation dx, dy
                    displacement[:, :, 0] -= dx
                    displacement[:, :, 1] -= dy
                    
                    mip_warped = warp_image_non_rigid(mip, displacement)
                    sf_warped = warp_footprints_non_rigid(sf, displacement)
                else: # rigid or similarity or translation with nudges
                    from .alignment import compose_warp_matrix_from_params
                    warp_mat_updated = compose_warp_matrix_from_params(dx, dy, nudge_angle, nudge_scale, cx=sf.shape[2]/2.0, cy=sf.shape[1]/2.0)
                    
                    mip_warped = warp_image_rigid(mip, warp_mat_updated)
                    sf_warped = warp_footprints_rigid(sf, warp_mat_updated)
                    
                # Write warped MaxProjection
                del cal_grp['MaxProjection']
                ds_mip = f.create_dataset(f"{cal_path}/MaxProjection", data=np.transpose(mip_warped), dtype='float64')
                ds_mip.attrs['MATLAB_class'] = b'double'
                ds_mip.attrs['H5PATH'] = f"/{cal_path.replace('/', '')}".encode('utf-8')
                
                # Write warped SpatialFootprints
                del cal_grp['SpatialFootprints']
                ds_sf = f.create_dataset(f"{cal_path}/SpatialFootprints", data=np.transpose(sf_warped, (0, 2, 1)), dtype='float64')
                ds_sf.attrs['MATLAB_class'] = b'double'
                ds_sf.attrs['H5PATH'] = f"/{cal_path.replace('/', '')}".encode('utf-8')
                
                # Set AlignmentShift to [0, 0]
                if 'AlignmentShift' in cal_grp:
                    del cal_grp['AlignmentShift']
                shift_data = np.zeros((2, 1), dtype='float64')
                ds_shift = f.create_dataset(f"{cal_path}/AlignmentShift", data=shift_data, dtype='float64')
                ds_shift.attrs['MATLAB_class'] = b'double'
                ds_shift.attrs['H5PATH'] = f"/{cal_path.replace('/', '')}".encode('utf-8')

    def delete_session(self, cohort, mouse, session_type, session):
        """Deletes a session and its contents from the HDF5 file."""
        with h5py.File(self._prepare_processed_write(), 'r+') as f:
            sess_path = f"{self.db_var_name}/{cohort}/{mouse}/{session_type}/{session}"
            if sess_path in f:
                del f[sess_path]
                
            # Clean up empty session type directories
            type_path = f"{self.db_var_name}/{cohort}/{mouse}/{session_type}"
            if type_path in f and len(f[type_path].keys()) == 0:
                del f[type_path]
                
        self.metadata = self._scan_metadata()
