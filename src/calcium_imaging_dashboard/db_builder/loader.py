"""
loader.py — Data loaders for CaImAn and Minian analysis outputs.

Each loader:
  1. Finds the latest yyyy-mm-dd* subfolder inside the given analysis folder
     (or falls back to the analysis folder itself if none exist).
  2. Reads the relevant result files.
  3. Returns a dict with keys:
       spatial    : np.ndarray  (N, H, W)  float64
       temporal   : np.ndarray  (N, T)     float64
       max_proj   : np.ndarray  (H, W)     float64
"""

import os
import re
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}")


class DataNotFoundError(Exception):
    """Raised when required data files are absent in the expected location."""
    pass


def _find_latest_data_dir(analysis_dir: str) -> str:
    """
    Returns the latest yyyy-mm-dd* subfolder inside *analysis_dir*.
    Falls back to *analysis_dir* itself if no date-stamped subfolders exist.
    """
    try:
        entries = sorted(
            [e for e in os.listdir(analysis_dir)
             if DATE_PATTERN.match(e) and os.path.isdir(os.path.join(analysis_dir, e))],
            reverse=True
        )
    except FileNotFoundError:
        raise DataNotFoundError(f"Analysis folder not found: {analysis_dir}")

    if entries:
        return os.path.join(analysis_dir, entries[0])
    return analysis_dir


# ──────────────────────────────────────────────────────────────────────────────
# CaImAn loader
# ──────────────────────────────────────────────────────────────────────────────

class CaimanLoader:
    """Loads calcium imaging data from a CaImAn results HDF5 file."""

    RESULT_FILENAME = "caiman_results.hdf5"

    def load(self, analysis_dir: str) -> dict:
        """
        Parameters
        ----------
        analysis_dir : str
            Path to the *caiman-analysis* folder.

        Returns
        -------
        dict with keys: spatial (N,H,W), temporal (N,T), max_proj (H,W)
        """
        import h5py
        import scipy.sparse

        data_dir = _find_latest_data_dir(analysis_dir)
        result_path = os.path.join(data_dir, self.RESULT_FILENAME)

        if not os.path.isfile(result_path):
            raise DataNotFoundError(
                f"caiman_results.hdf5 not found in: {data_dir}"
            )

        with h5py.File(result_path, "r") as f:
            estimates = f["estimates"]

            # ── Field-of-view dimensions ────────────────────────────────────
            if "dims" in estimates:
                dims_raw = estimates["dims"][:]
                if dims_raw.ndim == 2:
                    dims_raw = dims_raw.flatten()
                H, W = int(dims_raw[0]), int(dims_raw[1])
            else:
                raise DataNotFoundError(
                    f"estimates/dims not found in {result_path}. "
                    "Cannot determine field-of-view size."
                )

            # ── Spatial footprints (A): sparse d×K → dense N×H×W ──────────
            # CaImAn stores A as CSC sparse matrix components under estimates/A
            A_grp = estimates["A"]
            if "data" in A_grp and "indices" in A_grp and "indptr" in A_grp:
                data_vals = A_grp["data"][:]
                indices   = A_grp["indices"][:]
                indptr    = A_grp["indptr"][:]
                shape_ds  = A_grp["shape"][:] if "shape" in A_grp else np.array([H * W, 0])
                d, K = int(shape_ds[0]), int(shape_ds[1])
                A_sparse = scipy.sparse.csc_matrix(
                    (data_vals, indices, indptr), shape=(d, K)
                )
                A_dense = A_sparse.toarray()  # (d, K) = (H*W, N)
            elif hasattr(A_grp, "shape"):
                # Stored as a plain dense matrix
                A_dense = A_grp[:]
                d, K = A_dense.shape
            else:
                raise DataNotFoundError(
                    f"Cannot parse estimates/A in {result_path}."
                )

            K = A_dense.shape[1]  # number of cells
            # Reshape to (K, H, W) — CaImAn pixel order is Fortran (column-major)
            spatial = A_dense.T.reshape(K, H, W, order="F").astype(np.float64)

            # ── Temporal footprints (C): K×T ────────────────────────────────
            if "C" not in estimates:
                raise DataNotFoundError(f"estimates/C not found in {result_path}.")
            C = estimates["C"][:]
            if C.ndim == 1:
                C = C[np.newaxis, :]
            temporal = C.astype(np.float64)  # (K, T)
            if temporal.shape[0] != K:
                # Possibly transposed
                temporal = temporal.T

            # ── MaxProjection ────────────────────────────────────────────────
            # Priority order (each silently skips to the next on failure):
            #   1. b0           — per-pixel constant baseline (always present)
            #   2. .mmap file   — true max-projection of the raw movie
            #   3. b × mean(f)  — background spatial × mean temporal bg
            #   4. Cn           — local correlation image
            #   5. Top-level HDF5 keys (some pipelines save mean_image etc.)
            #   6. Max across spatial footprints (last resort)
            max_proj = None
            max_projection_source = None

            # 1. b0 — per-pixel constant baseline fitted by CNMF.
            #    Shape (d,) = (H*W,), Fortran order. Always present when CNMF
            #    ran without a rank-1 background (b/f are NoneType).
            if max_proj is None and "b0" in estimates:
                try:
                    b0 = estimates["b0"][:]
                    if b0.shape == (H * W,):
                        max_proj = b0.reshape(H, W, order="F").astype(np.float64)
                        max_projection_source = "caiman:estimates/b0"
                except Exception:
                    pass

            # 2. .mmap file — true per-pixel maximum across all movie frames.
            #    CaImAn stores the mmap path in params/data/fnames. We also
            #    search the data directory as a fallback. Movie is read in
            #    500-frame chunks to keep memory bounded.
            if max_proj is None:
                try:
                    mmap_path = None

                    # a) Try params/data/fnames stored in the HDF5
                    try:
                        fnames_ds = f["params"]["data"]["fnames"]
                        raw = fnames_ds[()]
                        # Stored as bytes or str scalar / array
                        if isinstance(raw, (bytes, np.bytes_)):
                            candidate = raw.decode("utf-8", errors="ignore")
                        elif isinstance(raw, np.ndarray) and raw.size > 0:
                            candidate = raw.flat[0]
                            if isinstance(candidate, (bytes, np.bytes_)):
                                candidate = candidate.decode("utf-8", errors="ignore")
                            else:
                                candidate = str(candidate)
                        else:
                            candidate = str(raw)
                        # The stored path may point to a .mmap or a source file;
                        # resolve the matching .mmap alongside it.
                        base_dir = os.path.dirname(candidate)
                        for fn in os.listdir(base_dir) if os.path.isdir(base_dir) else []:
                            if fn.endswith(".mmap"):
                                mmap_path = os.path.join(base_dir, fn)
                                break
                    except Exception:
                        pass

                    # b) Fallback: search the data_dir itself
                    if mmap_path is None:
                        for fn in os.listdir(data_dir):
                            if fn.endswith(".mmap"):
                                mmap_path = os.path.join(data_dir, fn)
                                break

                    if mmap_path is not None and os.path.isfile(mmap_path):
                        import re as _re
                        m = _re.search(
                            r"d1_(\d+)_d2_(\d+)_d3_(\d+)_order_([CF])_frames_(\d+)",
                            os.path.basename(mmap_path)
                        )
                        if m:
                            md1, md2, md3 = int(m.group(1)), int(m.group(2)), int(m.group(3))
                            morder, mT = m.group(4), int(m.group(5))
                            md = md1 * md2 * md3
                            CHUNK = 500
                            mm = np.memmap(mmap_path, dtype="float32", mode="r",
                                           shape=(md, mT) if morder == "C" else (mT, md))
                            max_flat = np.full(md, -np.inf, dtype=np.float64)
                            for s in range(0, mT, CHUNK):
                                e = min(s + CHUNK, mT)
                                if morder == "C":
                                    max_flat = np.maximum(max_flat, mm[:, s:e].max(axis=1))
                                else:
                                    max_flat = np.maximum(max_flat, mm[s:e, :].max(axis=0))
                            del mm
                            max_proj = max_flat.reshape(md1, md2, order="F").astype(np.float64)
                            max_projection_source = "caiman:movie_mmap_max"
                except Exception:
                    pass

            # 3. b × mean(f) — background reconstruction
            if max_proj is None:
                try:
                    b_raw = estimates["b"][:]
                    f_raw = estimates["f"][:]
                    # Guard against NoneType scalars stored as object dtype
                    if b_raw.dtype == object or f_raw.dtype == object:
                        raise ValueError("b or f is NoneType")
                    if b_raw.ndim == 1:
                        b_raw = b_raw[:, np.newaxis]
                    if f_raw.ndim == 1:
                        f_raw = f_raw[np.newaxis, :]
                    bg_mean = b_raw @ f_raw.mean(axis=1)
                    max_proj = bg_mean.reshape(H, W, order="F").astype(np.float64)
                    max_projection_source = "caiman:background_mean"
                except Exception:
                    pass

            # 4. Cn — local correlation image, shape (H,W) or flattened (H*W,)
            if max_proj is None and "Cn" in estimates:
                try:
                    Cn = estimates["Cn"][:]
                    if Cn.ndim == 1 and Cn.size == H * W:
                        Cn = Cn.reshape(H, W, order="F")
                    elif Cn.shape == (W, H):
                        Cn = Cn.T
                    if Cn.ndim == 2:
                        max_proj = Cn.astype(np.float64)
                        max_projection_source = "caiman:estimates/Cn"
                except Exception:
                    pass

            # 5. Top-level HDF5 keys (some CaImAn / pipeline versions save these)
            if max_proj is None:
                try:
                    for candidate in ("mean_image", "max_image", "meanImg", "maxImg"):
                        if candidate in f:
                            img = f[candidate][:]
                            if img.ndim == 2:
                                if img.shape == (W, H):
                                    img = img.T
                                max_proj = img.astype(np.float64)
                                max_projection_source = f"caiman:{candidate}"
                                break
                except Exception:
                    pass

            # 6. Last resort: max pixel value across all spatial footprints
            if max_proj is None:
                max_proj = np.max(spatial, axis=0).astype(np.float64)
                max_projection_source = "caiman:spatial_footprints_max"

        return {
            "spatial":   spatial,    # (N, H, W)
            "temporal":  temporal,   # (N, T)
            "max_proj":  max_proj,   # (H, W)
            "max_projection_source": max_projection_source,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Minian loader
# ──────────────────────────────────────────────────────────────────────────────

class MinianLoader:
    """Loads calcium imaging data from a Minian .zarr output directory."""

    def load(self, analysis_dir: str) -> dict:
        """
        Parameters
        ----------
        analysis_dir : str
            Path to the *minian-analysis* folder.

        Returns
        -------
        dict with keys: spatial (N,H,W), temporal (N,T), max_proj (H,W)
        """
        import zarr

        data_dir = _find_latest_data_dir(analysis_dir)

        # Minian may store zarr files directly in data_dir or in a `data/` sub-folder
        candidates = [
            data_dir,
            os.path.join(data_dir, "data"),
        ]

        zarr_base = None
        for c in candidates:
            if os.path.isdir(os.path.join(c, "A.zarr")):
                zarr_base = c
                break

        if zarr_base is None:
            raise DataNotFoundError(
                f"A.zarr not found under {data_dir} or {data_dir}/data"
            )

        def _load_zarr(name):
            path = os.path.join(zarr_base, name)
            if not os.path.isdir(path):
                raise DataNotFoundError(f"{name} not found at {zarr_base}")
            return zarr.open(path, mode="r")

        def _zarr_to_array(zarr_obj, var_name: str) -> np.ndarray:
            """
            Minian zarr stores are xarray-backed zarr Groups. Each .zarr folder
            contains the data array under a key matching the variable name (e.g.
            A.zarr → group["A"]), plus string coordinate arrays (unit_id, height,
            width, frame...). We must index by variable name to avoid loading those
            string coordinates.

            Fallback: pick the first numeric child array if the name isn't present.
            """
            import zarr as _zarr
            if isinstance(zarr_obj, _zarr.Array):
                # Already a plain array (rare, but handle it)
                return np.array(zarr_obj)

            # It's a Group — try by variable name first
            if var_name in zarr_obj:
                return np.array(zarr_obj[var_name])

            # Fallback: find the first numeric child array
            for key in zarr_obj.keys():
                child = zarr_obj[key]
                if isinstance(child, _zarr.Array) and np.issubdtype(child.dtype, np.number):
                    return np.array(child)

            raise DataNotFoundError(
                f"No numeric array found inside zarr group for variable '{var_name}'"
            )

        # ── Spatial footprints: A.zarr → group["A"] → (unit_id, height, width) ──
        A_z = _load_zarr("A.zarr")
        A = _zarr_to_array(A_z, "A").astype(np.float64)    # (N, H, W)
        if A.ndim == 2:
            A = A[np.newaxis, :, :]

        # ── Temporal footprints: C.zarr → group["C"] → (unit_id, frame) ─────────
        C_z = _load_zarr("C.zarr")
        C = _zarr_to_array(C_z, "C").astype(np.float64)    # (N, T)
        if C.ndim == 1:
            C = C[np.newaxis, :]

        # ── MaxProjection: max_proj.zarr → group["max_proj"] → (height, width) ──
        mp_z = _load_zarr("max_proj.zarr")
        max_proj = _zarr_to_array(mp_z, "max_proj").astype(np.float64)
        if max_proj.ndim > 2:
            max_proj = max_proj.squeeze()

        return {
            "spatial":   A,         # (N, H, W)
            "temporal":  C,         # (N, T)
            "max_proj":  max_proj,  # (H, W)
            "max_projection_source": "minian:max_proj.zarr/max_proj",
        }


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def get_loader(analysis_type: str):
    """Returns the appropriate loader for the given analysis type string."""
    loaders = {
        "caiman-analysis": CaimanLoader(),
        "minian-analysis": MinianLoader(),
    }
    loader = loaders.get(analysis_type)
    if loader is None:
        raise ValueError(
            f"Unsupported analysis type '{analysis_type}'. "
            f"Supported: {list(loaders.keys())}"
        )
    return loader
