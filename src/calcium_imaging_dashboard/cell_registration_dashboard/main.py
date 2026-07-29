import os
import argparse
import socket
import sys
import json
import base64
import itertools
import re
import traceback
import cv2
import uvicorn
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import webbrowser
from threading import Timer
import scipy
import scipy.io

# Import local modules
from calcium_imaging_dashboard import __version__
from .database import CalciumImagingDatabase
from .save_coordinator import SaveCoordinator
from .workspace import EditWorkspace
from .quality import histogram
from .alignment import (
    compute_ncc,
    compute_stack_coherence,
    register_images,
    warp_image_rigid,
    warp_image_non_rigid,
    warp_footprints_rigid,
    warp_footprints_non_rigid,
    compute_centroids,
    compose_warp_matrix_from_params,
    compose_displacement_fields,
    compute_alignment_nccs,
)
from .alignment_models import resolve_alignment_reference
from .matching import (
    get_sparse_footprints,
    shift_sparse_footprint,
    match_cells_across_sessions,
    optimize_matching_parameters,
    compute_matching_distributions,
    compute_overlap,
    summarize_matching,
    delete_master_cell_groups,
)

app = FastAPI(title="Calcium Imaging Cell Processing Dashboard")
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")

# Global database instance
db = CalciumImagingDatabase()
workspace = EditWorkspace()
db.workspace = workspace

# Grid search results directory (persisted JSON files per session)
grid_results_dir = ""

# Workspace-owned in-memory domains. These aliases preserve the existing read
# paths while Undo/Redo now restores the exact alignment and matching state.
alignment_cache = workspace.alignment_state
matching_cache = workspace.matching_state


def ensure_cache_scope(mouse, cohort=None):
    """Prevent same-named animals in different cohorts from sharing state."""
    if not cohort:
        return
    current = alignment_cache.get(mouse)
    if current is not None and current.get("__cohort__") not in (None, cohort):
        alignment_cache.pop(mouse, None)
    alignment_cache.setdefault(mouse, {})["__cohort__"] = cohort
    match = matching_cache.get(mouse)
    if match is not None and match.get("cohort") not in (None, cohort):
        matching_cache.pop(mouse, None)

# Helper function to convert 2D array to base64 PNG
def array_to_png_base64(arr):
    arr = np.nan_to_num(arr)
    mx = np.max(arr)
    mn = np.min(arr)
    if mx > mn:
        norm_arr = ((arr - mn) / (mx - mn) * 255.0).astype('uint8')
    else:
        norm_arr = np.zeros(arr.shape, dtype='uint8')
    
    # Encode as PNG
    success, encoded_img = cv2.imencode('.png', norm_arr)
    if success:
        img_str = base64.b64encode(encoded_img).decode("utf-8")
        return f"data:image/png;base64,{img_str}"
    return ""

def decompose_canvas_matrix(M, cx=304.0, cy=304.0):
    scale = float(np.sqrt(M[0, 0]**2 + M[1, 0]**2))
    rot_rad = np.arctan2(M[1, 0], M[0, 0])
    rot = float(np.degrees(rot_rad))
    cos_r = np.cos(rot_rad)
    sin_r = np.sin(rot_rad)
    dx = float(M[0, 2] - cx + scale * cos_r * cx - scale * sin_r * cy)
    dy = float(M[1, 2] - cy + scale * sin_r * cx + scale * cos_r * cy)
    return dx, dy, rot, scale

def decompose_warp_matrix(matrix, cx=304.0, cy=304.0):
    """Decomposes a 2x3 backend warp matrix into canvas space parameters (dx, dy, rotation, scale)."""
    W_3x3 = np.eye(3, dtype=np.float32)
    W_3x3[:2, :] = matrix
    return decompose_canvas_matrix(W_3x3, cx=cx, cy=cy)

# API Endpoints
@app.get("/api/metadata")
def get_metadata():
    """Retrieve full database hierarchy metadata."""
    try:
        return {
            "mice": db.get_mice_list(),
            "cohorts": db.metadata
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import shutil
from fastapi.responses import FileResponse

def ensure_write_on_processed_db():
    """Switch from raw preview to the in-memory working view without writing disk."""
    switched = db.view_mode == "raw"
    if switched:
        db.set_view_mode("working")
    return switched


def workspace_key(cohort, mouse, session_type, session_name):
    return workspace.key(cohort, mouse, session_type, session_name)


def workspace_loader(cohort, mouse, session_type, session_name):
    def load_arrays():
        data = db.load_session_calcium_data(
            cohort,
            mouse,
            session_type,
            session_name,
            warp_cached=False,
            include_workspace=False,
        )
        return data["spatial_footprints"], data["temporal_footprints"]

    return load_arrays


def invalidate_matching(mouse):
    return matching_cache.pop(mouse, None) is not None


def ordered_sessions(mouse, cohort=None, session_order=None):
    """Return sessions in the explicit sidebar/matching order when supplied."""
    sessions = db.get_sessions_for_mouse(mouse, cohort)
    if not session_order:
        return sessions
    by_name = {session["display_name"]: session for session in sessions}
    ordered = [by_name.pop(name) for name in session_order if name in by_name]
    ordered.extend(session for session in sessions if session["display_name"] in by_name)
    return ordered


def cell_matching_path(mouse, cohort=None):
    prefix = f"{cohort}_{mouse}" if cohort else mouse
    safe = "".join(char if char.isalnum() or char in "_.-" else "_" for char in prefix)
    scoped = os.path.join(db.var_dir, f"CellMatching_{safe}.mat")
    legacy = os.path.join(db.var_dir, f"CellMatching_{mouse}.mat")
    return scoped if cohort or not os.path.exists(legacy) else legacy


def safe_scope_filename(cohort, mouse, suffix):
    """Return a portable filename whose identity includes the full animal scope."""
    prefix = f"{cohort}_{mouse}" if cohort else mouse
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", prefix)
    safe_suffix = re.sub(r"[^A-Za-z0-9_.-]+", "_", suffix)
    return f"{safe}_{safe_suffix}"

class SetDbTypeRequest(BaseModel):
    db_type: str

@app.post("/api/set-db-type")
def set_db_type(req: SetDbTypeRequest):
    try:
        db.set_view_mode(req.db_type)
        return {"status": "success", "db_path": db.active_db_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/browse-database")
def browse_database():
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        file_path = filedialog.askopenfilename(
            title="Select Calcium Database File (.mat, .h5, .hdf5)",
            filetypes=[
                ("Calcium Database files", "*.mat *.h5 *.hdf5"),
                ("MATLAB files", "*.mat"),
                ("HDF5 files", "*.h5 *.hdf5"),
                ("All files", "*.*")
            ]
        )
        root.destroy()
        
        if not file_path:
            return {"status": "cancelled"}
            
        # Standardize path slashes for Windows/cross-platform
        file_path = os.path.abspath(file_path).replace("\\", "/")
        return {"status": "success", "db_path": file_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/load-database")
def load_database(req: Dict[str, str]):
    global db, grid_results_dir
    db_path = req.get("db_path")
    if not db_path:
        raise HTTPException(status_code=400, detail="db_path is required")
    if not os.path.exists(db_path):
        raise HTTPException(status_code=404, detail=f"Database file not found at: {db_path}")
    
    try:
        # Re-initialize the CalciumImagingDatabase instance with the new path
        db = CalciumImagingDatabase(db_path)
        workspace.clear()
        db.workspace = workspace
        # Update grid_results_dir to be next to the database
        grid_results_dir = os.path.join(
            os.path.dirname(db_path), ".calcium-imaging-dashboard", "grid-search"
        )
        os.makedirs(grid_results_dir, exist_ok=True)
        
        # Clear caches since we're loading a new database
        alignment_cache.clear()
        matching_cache.clear()
        
        return {
            "status": "success",
            "db_path": db_path,
            "mice": db.get_mice_list()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/current-database")
def get_current_database():
    return {
        "db_path": db.active_db_path.replace("\\", "/") if db.active_db_path else "",
        "raw_db_path": db.raw_db_path.replace("\\", "/") if db.raw_db_path else "",
        "clean_db_path": db.processed_db_path.replace("\\", "/") if db.processed_db_path else "",
        "view_mode": db.view_mode,
        "mice": db.get_mice_list()
    }


@app.get("/api/workspace/status")
def get_workspace_status():
    return workspace.status()


@app.get("/api/workspace/view-state")
def get_workspace_view_state(mouse: str, cohort: Optional[str] = None):
    ensure_cache_scope(mouse, cohort)
    cache = matching_cache.get(mouse)
    if not isinstance(cache, dict) or "matching_matrix" not in cache:
        return {"has_matching": False, **workspace.status()}
    matrix = np.asarray(cache["matching_matrix"], dtype=float)
    serializable = [
        [None if np.isnan(value) or value < 0 else int(value) for value in row]
        for row in matrix
    ]
    return {
        "has_matching": True,
        "matching_matrix": serializable,
        "master_centroids": np.nan_to_num(
            cache.get("master_centroids", np.empty((0, 2))), nan=0.0
        ).tolist(),
        "session_centroids": [
            np.asarray(values).tolist()
            for values in cache.get("session_centroids", [])
        ],
        **summarize_matching(matrix),
        **workspace.status(),
    }


@app.post("/api/workspace/undo")
def undo_workspace():
    label = workspace.undo()
    return {"status": "success", "action": label, **workspace.status()}


@app.post("/api/workspace/redo")
def redo_workspace():
    label = workspace.redo()
    return {"status": "success", "action": label, **workspace.status()}


@app.post("/api/workspace/discard")
def discard_workspace():
    workspace.clear()
    matching_cache.clear()
    return {"status": "success", **workspace.status()}


@app.post("/api/workspace/save")
def save_workspace():
    try:
        for mouse, cache in list(matching_cache.items()):
            if isinstance(cache, dict) and "matching_matrix" in cache:
                hydrate_matching_cache(mouse, cache.get("cohort"), cache)
        saved = SaveCoordinator(db, workspace).save(workspace.save_snapshot())
        return {
            "status": "success",
            **saved,
            "message": "Cells, alignments, and matching saved as one working checkpoint.",
            **workspace.status(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/frontend/images/{image_name}")
def get_frontend_image(image_name: str):
    img_path = os.path.join(os.path.dirname(__file__), "frontend", "images", image_name)
    if os.path.exists(img_path):
        return FileResponse(img_path)
    raise HTTPException(status_code=404, detail="Image not found")


@app.get("/api/mouse/{mouse_name}/sessions")
def get_mouse_sessions(mouse_name: str, cohort: Optional[str] = None):
    """Retrieve ordered session list for a specific mouse."""
    try:
        ensure_cache_scope(mouse_name, cohort)
        sessions = db.get_sessions_for_mouse(mouse_name, cohort)
        
        # Merge with in-memory alignment shifts if cached
        mouse_cache = alignment_cache.get(mouse_name, {})
        for s in sessions:
            s_name = s["display_name"]
            
            # Load from database to get true active cell count & shift
            try:
                summary = db.load_session_summary(
                    s["cohort"], s["mouse"], s["session_type"], s["session_name"]
                )
                s["n_cells"] = summary["n_cells"]
                
                if s_name in mouse_cache:
                    s["alignment_shift"] = [mouse_cache[s_name]["dx"], mouse_cache[s_name]["dy"]]
                    s["alignment_rotation"] = mouse_cache[s_name].get("rotation", 0.0)
                    s["alignment_scale"] = mouse_cache[s_name].get("scale", 1.0)
                    s["ncc"] = mouse_cache[s_name]["ncc"]
                    s["mip_ncc"] = mouse_cache[s_name].get("mip_ncc")
                    s["footprints_ncc"] = mouse_cache[s_name].get("footprints_ncc")
                    s["mode"] = mouse_cache[s_name]["mode"]
                else:
                    s["alignment_shift"] = list(summary["alignment_shift"])
                    s["alignment_rotation"] = 0.0
                    s["alignment_scale"] = 1.0
                    s["ncc"] = None
                    s["mip_ncc"] = None
                    s["footprints_ncc"] = None
                    s["mode"] = "translation"
            except Exception as exc:
                print(f"Could not load session summary for {s_name}: {exc}")
                s["alignment_shift"] = [0.0, 0.0]
                s["alignment_rotation"] = 0.0
                s["alignment_scale"] = 1.0
                s["ncc"] = None
                s["mip_ncc"] = None
                s["footprints_ncc"] = None
                s["mode"] = "translation"
                    
        return sessions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/session-calcium/{cohort}/{mouse}/{session_type}/{session_name}")
def get_session_calcium(cohort: str, mouse: str, session_type: str, session_name: str, warp: bool = True):
    """Loads session imaging data, encodes projections as PNGs, and extracts centroids & sparse footprints."""
    try:
        cached_align = None
        display_name = f"{session_type}_{session_name}"
        if warp:
            if mouse in alignment_cache and display_name in alignment_cache[mouse]:
                cached_align = alignment_cache[mouse][display_name]
                
        data = db.load_session_calcium_data(
            cohort,
            mouse,
            session_type,
            session_name,
            cached_alignment=cached_align,
            warp_cached=warp,
            include_temporal=False,
        )
        sf = data['spatial_footprints'] # Shape: (N, H, W)
        
        # Calculate Footprint Sum
        spatial_sum = np.sum(sf, axis=0)
        
        # Encode projections as PNG base64 for fast web rendering
        max_proj_png = array_to_png_base64(data['max_projection'])
        spatial_sum_png = array_to_png_base64(spatial_sum)
        
        # Compute centroids and extract sparse footprints
        centroids = compute_centroids(sf).tolist()
        sparse_fps = get_sparse_footprints(sf)
        
        # Convert sparse footprint numpy arrays to lists for JSON serialization
        serialized_fps = []
        for fp in sparse_fps:
            serialized_fps.append({
                "idx": fp["idx"].tolist(),
                "vals": fp["vals"].tolist(),
                "norm": fp["norm"]
            })
            
        # Get shift from cache if present (temporary nudges)
        shift = [data['alignment_shift'][0], data['alignment_shift'][1]]
        rotation = 0.0
        scale = 1.0
        cached_mode = "translation"
        
        if mouse in alignment_cache and display_name in alignment_cache[mouse]:
            shift = [alignment_cache[mouse][display_name]["dx"], alignment_cache[mouse][display_name]["dy"]]
            rotation = alignment_cache[mouse][display_name].get("rotation", 0.0)
            scale = alignment_cache[mouse][display_name].get("scale", 1.0)
            cached_mode = alignment_cache[mouse][display_name]["mode"]
            
        n_active_cells = db.load_session_summary(
            cohort, mouse, session_type, session_name
        )["n_cells"]
        
        return {
            "max_projection_png": max_proj_png,
            "spatial_sum_png": spatial_sum_png,
            "centroids": centroids,
            "sparse_footprints": serialized_fps,
            "alignment_shift": shift,
            "alignment_rotation": rotation,
            "alignment_scale": scale,
            "alignment_mode": cached_mode,
            "n_cells": n_active_cells,
            "width": int(sf.shape[2]),
            "height": int(sf.shape[1]),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/session-traces/{cohort}/{mouse}/{session_type}/{session_name}")
def get_session_traces(cohort: str, mouse: str, session_type: str, session_name: str, cells: str):
    """Fetches temporal traces on-demand for requested cell indices (0-indexed)."""
    try:
        indices = [int(c) for c in cells.split(",") if c.strip() != ""]
        rows = db.load_temporal_rows(cohort, mouse, session_type, session_name, indices)
        return [row.tolist() if row is not None else [] for row in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/session-preview/{cohort}/{mouse}/{session_type}/{session_name}")
def get_session_preview(cohort: str, mouse: str, session_type: str, session_name: str):
    try:
        mip, spatial_sum = db.load_session_preview(cohort, mouse, session_type, session_name)
        return {
            "max_projection_png": array_to_png_base64(mip),
            "spatial_sum_png": array_to_png_base64(spatial_sum),
            "width": int(mip.shape[1]),
            "height": int(mip.shape[0]),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

class SaveShiftsRequest(BaseModel):
    mouse_name: str
    cohort: Optional[str] = None
    shifts: Dict[str, List[float]] # { "Experimental/Mouse56/Reward100pct/Session03": [dx, dy] }

@app.post("/api/save-shifts")
def save_shifts(req: SaveShiftsRequest):
    """Commits alignment shifts (translation) back to database file."""
    try:
        db.save_workspace()
        db.save_alignment_shifts(req.shifts)
        workspace.clear()
        
        # Clear cache for this mouse
        if req.mouse_name in alignment_cache:
            del alignment_cache[req.mouse_name]
            
        return {"status": "success", "message": f"Shifts saved successfully to {os.path.basename(db.processed_db_path)}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class CachedAlignment(BaseModel):
    mode: str
    dx: float
    dy: float
    rotation: float
    scale: float

class UpdateCacheRequest(BaseModel):
    mouse: str
    cohort: Optional[str] = None
    alignments: Dict[str, CachedAlignment] # { session_display_name: CachedAlignment }

@app.post("/api/update-alignment-cache")
def update_alignment_cache(req: UpdateCacheRequest):
    try:
        ensure_cache_scope(req.mouse, req.cohort)
        updates = {}
        for s_name, align in req.alignments.items():
            current = alignment_cache.get(req.mouse, {}).get(s_name, {})
            updates[s_name] = {
                "mode": align.mode,
                "transform": current.get(
                    "transform", np.eye(2, 3, dtype=np.float32)
                ),
                "dx": align.dx,
                "dy": align.dy,
                "rotation": align.rotation,
                "scale": align.scale,
                "ncc": current.get("ncc", 1.0),
                "rel_dx": current.get("rel_dx", align.dx),
                "rel_dy": current.get("rel_dy", align.dy),
                "rel_rotation": current.get("rel_rotation", align.rotation),
                "rel_scale": current.get("rel_scale", align.scale),
            }
        matching_invalidated = req.mouse in matching_cache
        workspace.update_alignments(
            req.mouse,
            req.cohort,
            updates,
            label="Adjust alignment",
        )
        return {
            "status": "success",
            "matching_invalidated": matching_invalidated,
            "workspace": workspace.status(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# In-memory cache for raw nudge images to avoid repeatedly reading large HDF5 footprint stacks from disk
# Structure: { "mouse": str, "ref_session": str, "act_session": str, "ref_mip": np.ndarray, "act_mip": np.ndarray, "ref_sf": np.ndarray, "act_sf": np.ndarray }
nudge_image_cache = {}

class ComputeNCCRequest(BaseModel):
    mouse: str
    cohort: Optional[str] = None
    ref_session_name: str
    active_session_name: str
    dx: float
    dy: float
    rotation: float
    scale: float
    mode: str
    source: str
    strategy: str

@app.post("/api/compute-ncc")
def api_compute_ncc(req: ComputeNCCRequest):
    global nudge_image_cache
    try:
        ensure_cache_scope(req.mouse, req.cohort)
        # Check cache hit
        cache_hit = False
        if (nudge_image_cache.get("mouse") == req.mouse and
            nudge_image_cache.get("cohort") == req.cohort and
            nudge_image_cache.get("workspace_revision") == workspace.revision and
            nudge_image_cache.get("ref_session") == req.ref_session_name and 
            nudge_image_cache.get("act_session") == req.active_session_name):
            cache_hit = True
            
        if cache_hit:
            if req.source == 'MIP':
                ref_img = nudge_image_cache["ref_mip"]
                act_img = nudge_image_cache["act_mip"]
            else:
                ref_img = nudge_image_cache["ref_sf"]
                act_img = nudge_image_cache["act_sf"]
        else:
            # Resolve sessions
            sessions = db.get_sessions_for_mouse(req.mouse, req.cohort)
            ref_sess = next((s for s in sessions if s["display_name"] == req.ref_session_name), None)
            act_sess = next((s for s in sessions if s["display_name"] == req.active_session_name), None)
            
            if not ref_sess or not act_sess:
                raise ValueError("Reference or Active session not found.")
                
            # Load raw images
            ref_data = db.load_session_calcium_data(ref_sess["cohort"], ref_sess["mouse"], ref_sess["session_type"], ref_sess["session_name"], warp_cached=False, include_temporal=False)
            act_data = db.load_session_calcium_data(act_sess["cohort"], act_sess["mouse"], act_sess["session_type"], act_sess["session_name"], warp_cached=False, include_temporal=False)
            
            ref_mip = np.array(ref_data["max_projection"])
            act_mip = np.array(act_data["max_projection"])
            ref_sf = np.sum(ref_data["spatial_footprints"], axis=0)
            act_sf = np.sum(act_data["spatial_footprints"], axis=0)
            
            # Update cache
            nudge_image_cache = {
                "mouse": req.mouse,
                "cohort": req.cohort,
                "workspace_revision": workspace.revision,
                "ref_session": req.ref_session_name,
                "act_session": req.active_session_name,
                "ref_mip": ref_mip,
                "act_mip": act_mip,
                "ref_sf": ref_sf,
                "act_sf": act_sf
            }
            
            if req.source == 'MIP':
                ref_img = ref_mip
                act_img = act_mip
            else:
                ref_img = ref_sf
                act_img = act_sf
                
        # Resolve neighbor reference session accumulated shifts for sequential mode
        ref_dx, ref_dy = 0.0, 0.0
        ref_rot = 0.0
        ref_scale = 1.0
        
        if req.strategy == 'Sequential':
            display_name = req.ref_session_name
            if req.mouse in alignment_cache and display_name in alignment_cache[req.mouse]:
                ref_dx = alignment_cache[req.mouse][display_name]["dx"]
                ref_dy = alignment_cache[req.mouse][display_name]["dy"]
                ref_rot = alignment_cache[req.mouse][display_name].get("rotation", 0.0)
                ref_scale = alignment_cache[req.mouse][display_name].get("scale", 1.0)
            else:
                # Check HDF5 database shift directly
                sessions = db.get_sessions_for_mouse(req.mouse, req.cohort)
                ref_sess = next((s for s in sessions if s["display_name"] == req.ref_session_name), None)
                if ref_sess:
                    ref_data = db.load_session_calcium_data(ref_sess["cohort"], ref_sess["mouse"], ref_sess["session_type"], ref_sess["session_name"], warp_cached=False, include_temporal=False)
                    ref_dx, ref_dy = ref_data.get('alignment_shift', [0.0, 0.0])
                    
        # Apply transformations to warp both into global reference space
        from .alignment import compose_warp_matrix_from_params, warp_image_rigid, compute_ncc
        
        if req.strategy == 'Sequential':
            M_ref = compose_warp_matrix_from_params(ref_dx, ref_dy, ref_rot, ref_scale)
            ref_img_warped = warp_image_rigid(ref_img, M_ref)
        else:
            ref_img_warped = ref_img
            
        M_act = compose_warp_matrix_from_params(req.dx, req.dy, req.rotation, req.scale)
        warped_img = warp_image_rigid(act_img, M_act)
        
        # Compute NCC
        ncc_val = compute_ncc(ref_img_warped, warped_img, downsample=True)
        return {"ncc": ncc_val}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/mouse/{mouse_name}/overview")
def get_mouse_overview(
    mouse_name: str,
    cohort: Optional[str] = None,
    session_order: Optional[str] = None,
):
    """Aggregates projection images and shifts all session centroids for a global overview."""
    try:
        order = session_order.split("|") if session_order else None
        sessions = ordered_sessions(mouse_name, cohort, order)
        if not sessions:
            raise ValueError(f"No sessions found for mouse {mouse_name}")
            
        # Find reference session (session with most cells)
        max_cells = -1
        ref_sess = None
        for s in sessions:
            if s["n_cells"] > max_cells:
                max_cells = s["n_cells"]
                ref_sess = s
                
        # Load reference session projection as background image
        ref_data = db.load_session_calcium_data(
            ref_sess["cohort"], ref_sess["mouse"], ref_sess["session_type"], ref_sess["session_name"],
            include_temporal=False,
        )
        ref_sf = ref_data["spatial_footprints"]
        spatial_sum = np.sum(ref_sf, axis=0)
        spatial_sum_png = array_to_png_base64(spatial_sum)
        
        H, W = ref_sf.shape[1], ref_sf.shape[2]
        
        # Aggregate all shifted centroids and build combined MIP
        all_centroids = []
        combined_mip = np.zeros((H, W), dtype=np.float32)
        combined_spatial_sum = np.zeros((H, W), dtype=np.float32)
        aligned_mips = []
        aligned_spatial_sums = []
        
        for s_idx, s in enumerate(sessions):
            cached_align = None
            s_name = s["display_name"]
            if mouse_name in alignment_cache and s_name in alignment_cache[mouse_name]:
                cached_align = alignment_cache[mouse_name][s_name]
                
            s_data = db.load_session_calcium_data(
                s["cohort"], s["mouse"], s["session_type"], s["session_name"],
                cached_alignment=cached_align, warp_cached=True, include_temporal=False,
            )
            sf = s_data["spatial_footprints"]
            n_cells = sf.shape[0]
            session_spatial_sum = np.sum(sf, axis=0, dtype=np.float32)
            aligned_spatial_sums.append(session_spatial_sum)
            combined_spatial_sum += session_spatial_sum
            
            # Already warped on-the-fly, so no additional shift is applied
            raw_centroids = compute_centroids(sf)
            for idx in range(n_cells):
                shifted_x = float(raw_centroids[idx, 0])
                shifted_y = float(raw_centroids[idx, 1])
                all_centroids.append({
                    "x": shifted_x,
                    "y": shifted_y,
                    "session_idx": s_idx,
                    "cell_idx": idx,
                    "display_name": s_name,
                    "cohort": s["cohort"],
                    "mouse": s["mouse"],
                    "session_type": s["session_type"],
                    "session_name": s["session_name"],
                })
                
            # Align and overlay MIP (already warped on-the-fly)
            mip = np.array(s_data["max_projection"], dtype=np.float32)
            aligned_mips.append(mip)
            combined_mip = np.maximum(combined_mip, mip)
            
        combined_mip_png = array_to_png_base64(combined_mip)
        combined_spatial_sum_png = array_to_png_base64(combined_spatial_sum)
        mip_coherence = compute_stack_coherence(aligned_mips)
        sf_coherence = compute_stack_coherence(aligned_spatial_sums)

        return {
            "spatial_sum_png": spatial_sum_png,
            "combined_spatial_sum_png": combined_spatial_sum_png,
            "combined_mip_png": combined_mip_png,
            "centroids": all_centroids,
            "ref_session_display_name": ref_sess["display_name"],
            "total_cells": len(all_centroids),
            "width": int(W),
            "height": int(H),
            "stack_coherence": {
                "mip": mip_coherence,
                "footprints": sf_coherence,
                "combined": float(np.nanmean([mip_coherence, sf_coherence])),
                "n_sessions": len(aligned_mips),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class CleanAllCellsRequest(BaseModel):
    mouse: str
    cohort: Optional[str] = None
    cx: float
    cy: float
    rx: float
    ry: float
    theta: float = 0.0

@app.post("/api/clean-all-sessions")
def clean_all_sessions(req: CleanAllCellsRequest):
    """Filters centroids in shifted/world space against a rotated oval mask and cleans all sessions."""
    try:
        db_switched = ensure_write_on_processed_db()
        sessions = db.get_sessions_for_mouse(req.mouse, req.cohort)
        cleaned_counts = {}
        keep_requests = {}
        loaders = {}
        
        cos_t = np.cos(-req.theta)
        sin_t = np.sin(-req.theta)
        
        for s in sessions:
            s_name = s["display_name"]
            cached_align = None
            if req.mouse in alignment_cache and s_name in alignment_cache[req.mouse]:
                cached_align = alignment_cache[req.mouse][s_name]
                
            s_data = db.load_session_calcium_data(
                s["cohort"], s["mouse"], s["session_type"], s["session_name"],
                cached_alignment=cached_align, warp_cached=True, include_temporal=False,
            )
            sf = s_data["spatial_footprints"]
            n_cells = sf.shape[0]
            
            # Already warped on-the-fly, so no additional shift is applied
            dx, dy = 0.0, 0.0
            
            raw_centroids = compute_centroids(sf)
            keep_indices = []
            
            for idx in range(n_cells):
                shifted_x = raw_centroids[idx, 0] + dx
                shifted_y = raw_centroids[idx, 1] + dy
                
                # Rotate and translate relative to ellipse center
                tx = shifted_x - req.cx
                ty = shifted_y - req.cy
                local_x = tx * cos_t - ty * sin_t
                local_y = tx * sin_t + ty * cos_t
                
                # Check if inside ellipse
                term_x = (local_x ** 2) / (req.rx ** 2) if req.rx > 0 else 9999
                term_y = (local_y ** 2) / (req.ry ** 2) if req.ry > 0 else 9999
                if term_x + term_y <= 1.0:
                    keep_indices.append(idx)
                    
            key = workspace_key(s["cohort"], s["mouse"], s["session_type"], s["session_name"])
            keep_requests[key] = keep_indices
            loaders[key] = workspace_loader(*key)
            cleaned_counts[s_name] = n_cells - len(keep_indices)

        workspace.discard_indices(
            keep_requests,
            loaders,
            keep_selected=True,
            label="Clean all sessions with Overview ellipse",
        )
        matching_invalidated = invalidate_matching(req.mouse)
                
        return {
            "status": "success",
            "discarded": cleaned_counts,
            "db_switched": db_switched,
            "matching_invalidated": matching_invalidated,
            "workspace": workspace.status(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class CleanSelectedOverviewRequest(BaseModel):
    mouse: str
    cohort: Optional[str] = None
    cells: List[Dict[str, Any]]

@app.post("/api/clean-selected-overview")
def clean_selected_overview(req: CleanSelectedOverviewRequest):
    """Discards specific cell indices across multiple sessions."""
    try:
        db_switched = ensure_write_on_processed_db()
        sessions = db.get_sessions_for_mouse(req.mouse, req.cohort)
        
        by_session = {}
        identity_to_index = {
            (s["cohort"], s["mouse"], s["session_type"], s["session_name"]): index
            for index, s in enumerate(sessions)
        }
        for c in req.cells:
            identity = (
                c.get("cohort"), c.get("mouse"), c.get("session_type"), c.get("session_name")
            )
            s_idx = identity_to_index.get(identity, c.get("session_idx", -1))
            c_idx = c["cell_idx"]
            by_session.setdefault(s_idx, []).append(c_idx)
            
        discarded_counts = {}
        requests = {}
        loaders = {}
        for s_idx, cell_indices in by_session.items():
            if s_idx < 0 or s_idx >= len(sessions):
                continue
            sess = sessions[s_idx]
            key = workspace_key(sess["cohort"], sess["mouse"], sess["session_type"], sess["session_name"])
            requests[key] = cell_indices
            loaders[key] = workspace_loader(*key)
            discarded_counts[sess["display_name"]] = len(set(cell_indices))

        workspace.discard_indices(
            requests, loaders, label="Discard selected Overview cells"
        )
        matching_invalidated = invalidate_matching(req.mouse)
            
        return {
            "status": "success",
            "discarded": discarded_counts,
            "db_switched": db_switched,
            "matching_invalidated": matching_invalidated,
            "workspace": workspace.status(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class CleanUnselectedOverviewRequest(BaseModel):
    mouse: str
    cohort: Optional[str] = None
    selected_cells: List[Dict[str, Any]]

@app.post("/api/clean-unselected-overview")
def clean_unselected_overview(req: CleanUnselectedOverviewRequest):
    """Keeps only specific cells across multiple sessions and discards everything else."""
    try:
        db_switched = ensure_write_on_processed_db()
        sessions = db.get_sessions_for_mouse(req.mouse, req.cohort)
        
        by_session = {}
        identity_to_index = {
            (s["cohort"], s["mouse"], s["session_type"], s["session_name"]): index
            for index, s in enumerate(sessions)
        }
        for c in req.selected_cells:
            identity = (
                c.get("cohort"), c.get("mouse"), c.get("session_type"), c.get("session_name")
            )
            s_idx = identity_to_index.get(identity, c.get("session_idx", -1))
            c_idx = c["cell_idx"]
            by_session.setdefault(s_idx, []).append(c_idx)
            
        discarded_counts = {}
        requests = {}
        loaders = {}
        for s_idx, sess in enumerate(sessions):
            # The indices to keep for this session are exactly the ones selected
            keep_indices = by_session.get(s_idx, [])
            
            n_cells = db.load_session_summary(
                sess["cohort"], sess["mouse"], sess["session_type"], sess["session_name"]
            )["n_cells"]
            
            key = workspace_key(sess["cohort"], sess["mouse"], sess["session_type"], sess["session_name"])
            requests[key] = keep_indices
            loaders[key] = workspace_loader(*key)
            discarded_counts[sess["display_name"]] = n_cells - len(keep_indices)

        workspace.discard_indices(
            requests,
            loaders,
            keep_selected=True,
            label="Keep selected Overview cells",
        )
        matching_invalidated = invalidate_matching(req.mouse)
            
        return {
            "status": "success",
            "discarded": discarded_counts,
            "db_switched": db_switched,
            "matching_invalidated": matching_invalidated,
            "workspace": workspace.status(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class CleanCellsRequest(BaseModel):
    cohort: str
    mouse: str
    session_type: str
    session_name: str
    keep_indices: List[int]

@app.post("/api/clean-cells")
def clean_cells(req: CleanCellsRequest):
    """Discards cells outside the mask by slicing footprints in-place."""
    try:
        db_switched = ensure_write_on_processed_db()
        key = workspace_key(req.cohort, req.mouse, req.session_type, req.session_name)
        workspace.discard_indices(
            {key: req.keep_indices},
            {key: workspace_loader(*key)},
            keep_selected=True,
            label=f"Clean {req.session_type}_{req.session_name}",
        )
        new_count = len(req.keep_indices)
        matching_invalidated = invalidate_matching(req.mouse)
        return {
            "status": "success",
            "new_cell_count": new_count,
            "db_switched": db_switched,
            "matching_invalidated": matching_invalidated,
            "workspace": workspace.status(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class MergeCellsRequest(BaseModel):
    cohort: str
    mouse: str
    session_type: str
    session_name: str
    cell_indices: List[int]

@app.post("/api/merge-cells")
def merge_cells(req: MergeCellsRequest):
    """Merges selected cells inside a session."""
    try:
        db_switched = ensure_write_on_processed_db()
        key = workspace_key(req.cohort, req.mouse, req.session_type, req.session_name)
        new_count = workspace.merge_indices(
            key,
            req.cell_indices,
            workspace_loader(*key),
            label=f"Merge cells in {req.session_type}_{req.session_name}",
        )
        matching_invalidated = invalidate_matching(req.mouse)
        return {
            "status": "success",
            "new_cell_count": new_count,
            "db_switched": db_switched,
            "matching_invalidated": matching_invalidated,
            "workspace": workspace.status(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class CellQualityRequest(BaseModel):
    cohort: str
    mouse: str
    is_overview: bool
    session_type: str = ""
    session_name: str = ""
    selected_cell_index: Optional[int] = None


def _quality_sessions(req: CellQualityRequest):
    if req.is_overview:
        return db.get_sessions_for_mouse(req.mouse, req.cohort or None)
    return [{
        "cohort": req.cohort,
        "mouse": req.mouse,
        "session_type": req.session_type,
        "session_name": req.session_name,
        "display_name": f"{req.session_type}_{req.session_name}",
    }]


@app.post("/api/cell-quality-stats")
def get_cell_quality_stats(req: CellQualityRequest):
    """Return common and optional native-QC distributions for the active scope."""
    try:
        collected = {
            "FootprintArea": [],
            "TemporalContrast": [],
            "FootprintEccentricity": [],
            "SourceTemporalSNR": [],
            "SpatialCorrelation": [],
            "ClassifierScore": [],
        }
        accepted_counts = {"accepted": 0, "rejected": 0, "unknown": 0}
        selected_cell = None
        for session in _quality_sessions(req):
            quality = db.load_session_quality(
                session["cohort"],
                session["mouse"],
                session["session_type"],
                session["session_name"],
            )
            for name, values in quality["common"].items():
                collected[name].extend(np.asarray(values).tolist())
            source = quality["source"]
            for name, output_name in (
                ("TemporalSNR", "SourceTemporalSNR"),
                ("SpatialCorrelation", "SpatialCorrelation"),
                ("ClassifierScore", "ClassifierScore"),
            ):
                if name in source:
                    collected[output_name].extend(
                        np.asarray(source[name])[np.isfinite(source[name])].tolist()
                    )
            if "Accepted" in source:
                statuses = np.asarray(source["Accepted"]).reshape(-1)
                accepted_counts["accepted"] += int(np.count_nonzero(statuses == 1))
                accepted_counts["rejected"] += int(np.count_nonzero(statuses == 0))
                accepted_counts["unknown"] += int(np.count_nonzero(statuses == -1))

            if (
                req.selected_cell_index is not None
                and session["session_type"] == req.session_type
                and session["session_name"] == req.session_name
                and 0 <= req.selected_cell_index < len(quality["common"]["FootprintArea"])
            ):
                index = req.selected_cell_index
                selected_cell = {
                    "index": index,
                    "display_name": session["display_name"],
                    "common": {
                        name: float(values[index])
                        for name, values in quality["common"].items()
                    },
                    "source": {
                        name: (
                            int(values[index]) if name == "Accepted" else float(values[index])
                        )
                        for name, values in source.items()
                        if index < len(values)
                    },
                }

        histograms = {}
        for name, values in collected.items():
            counts, bins = histogram(np.asarray(values))
            histograms[name] = {"counts": counts, "bins": bins}
        return {
            "n_cells": len(collected["FootprintArea"]),
            "histograms": histograms,
            "accepted_counts": accepted_counts,
            "selected_cell": selected_cell,
            "metric_definition": {
                "footprint_area": "Pixels above 20% of each cell's peak footprint intensity.",
                "temporal_contrast": "99th-percentile amplitude above median divided by trace standard deviation.",
                "footprint_eccentricity": "Spatial-covariance eccentricity of the 20%-peak support.",
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


class AutoCleanRequest(CellQualityRequest):
    min_footprint_area: Optional[float] = None
    max_footprint_area: Optional[float] = None
    min_temporal_contrast: Optional[float] = None
    max_footprint_eccentricity: Optional[float] = None
    flag_source_rejected: bool = False
    min_source_snr: Optional[float] = None
    min_spatial_correlation: Optional[float] = None
    min_classifier_score: Optional[float] = None


@app.post("/api/autoclean-sessions")
def run_autoclean(req: AutoCleanRequest):
    """Detect review candidates without mutating the database or workspace."""
    try:
        threshold_names = (
            "min_footprint_area", "max_footprint_area", "min_temporal_contrast",
            "max_footprint_eccentricity", "min_source_snr",
            "min_spatial_correlation", "min_classifier_score",
        )
        for name in threshold_names:
            value = getattr(req, name)
            if value is not None and not np.isfinite(value):
                raise ValueError(f"{name} must be finite when enabled.")
        for name in (
            "min_footprint_area", "max_footprint_area",
            "min_temporal_contrast", "min_source_snr",
        ):
            value = getattr(req, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative.")
        for name in (
            "max_footprint_eccentricity",
            "min_classifier_score",
        ):
            value = getattr(req, name)
            if value is not None and not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1.")
        if (
            req.min_spatial_correlation is not None
            and not -1 <= req.min_spatial_correlation <= 1
        ):
            raise ValueError("min_spatial_correlation must be between -1 and 1.")
        if (
            req.min_footprint_area is not None
            and req.max_footprint_area is not None
            and req.min_footprint_area > req.max_footprint_area
        ):
            raise ValueError("Minimum footprint area cannot exceed maximum footprint area.")
        sessions_result = []
        for session in _quality_sessions(req):
            quality = db.load_session_quality(
                session["cohort"],
                session["mouse"],
                session["session_type"],
                session["session_name"],
            )
            common = quality["common"]
            source = quality["source"]
            candidates = []
            cell_count = len(common["FootprintArea"])
            for index in range(cell_count):
                failures = []

                def fail(rule, value):
                    failures.append({"rule": rule, "value": float(value)})

                area = common["FootprintArea"][index]
                temporal_contrast = common["TemporalContrast"][index]
                eccentricity = common["FootprintEccentricity"][index]
                if req.min_footprint_area is not None and area < req.min_footprint_area:
                    fail("min_footprint_area", area)
                if req.max_footprint_area is not None and area > req.max_footprint_area:
                    fail("max_footprint_area", area)
                if (
                    req.min_temporal_contrast is not None
                    and temporal_contrast < req.min_temporal_contrast
                ):
                    fail("min_temporal_contrast", temporal_contrast)
                if (
                    req.max_footprint_eccentricity is not None
                    and eccentricity > req.max_footprint_eccentricity
                ):
                    fail("max_footprint_eccentricity", eccentricity)

                accepted = source.get("Accepted")
                if req.flag_source_rejected and accepted is not None and accepted[index] == 0:
                    fail("source_rejected", accepted[index])
                for source_name, threshold, rule in (
                    ("TemporalSNR", req.min_source_snr, "min_source_snr"),
                    ("SpatialCorrelation", req.min_spatial_correlation, "min_spatial_correlation"),
                    ("ClassifierScore", req.min_classifier_score, "min_classifier_score"),
                ):
                    values = source.get(source_name)
                    if (
                        threshold is not None
                        and values is not None
                        and index < len(values)
                        and np.isfinite(values[index])
                        and values[index] < threshold
                    ):
                        fail(rule, values[index])
                if failures:
                    candidates.append({"index": index, "failed_rules": failures})
            sessions_result.append({
                "cohort": session["cohort"],
                "mouse": session["mouse"],
                "session_type": session["session_type"],
                "session_name": session["session_name"],
                "display_name": session["display_name"],
                "candidate_indices": [item["index"] for item in candidates],
                "candidates": candidates,
            })
        return {
            "status": "success",
            "sessions": sessions_result,
            "total_candidates": int(sum(len(item["candidates"]) for item in sessions_result)),
            "mutated": False,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


class PairwiseStatsRequest(BaseModel):
    cohort: str
    mouse: str
    session_type: str
    session_name: str
    is_overview: bool

@app.post("/api/pairwise-stats")
def get_pairwise_stats(req: PairwiseStatsRequest):
    try:
        all_dists = []
        all_corrs = []
        all_overlaps = []
        
        from .matching import get_sparse_footprints, compute_overlap
        
        if req.is_overview:
            sessions = db.get_sessions_for_mouse(req.mouse, req.cohort or None)
            for s in sessions:
                s_data = db.load_session_calcium_data(
                    s["cohort"], s["mouse"], s["session_type"], s["session_name"]
                )
                sf = s_data["spatial_footprints"]
                tf = s_data["temporal_footprints"]
                n_cells = sf.shape[0]
                if n_cells < 2:
                    continue
                centroids = compute_centroids(sf)
                # Distances
                dx = centroids[:, 0:1] - centroids[:, 0:1].T
                dy = centroids[:, 1:2] - centroids[:, 1:2].T
                dist_mat = np.sqrt(dx*dx + dy*dy)
                # Pearson
                tf_means = np.mean(tf, axis=1, keepdims=True)
                tf_stds = np.std(tf, axis=1, keepdims=True)
                tf_stds[tf_stds == 0] = 1.0
                tf_normalized = (tf - tf_means) / tf_stds
                corr_mat = np.dot(tf_normalized, tf_normalized.T) / tf.shape[1]
                
                # Overlaps
                sfs = get_sparse_footprints(sf)
                overlap_mat = np.zeros((n_cells, n_cells))
                for i in range(n_cells):
                    for j in range(i + 1, n_cells):
                        val = compute_overlap(sfs[i], sfs[j], 'cosine')
                        overlap_mat[i, j] = val
                        overlap_mat[j, i] = val
                
                triu_idx = np.triu_indices(n_cells, k=1)
                all_dists.extend(dist_mat[triu_idx].tolist())
                all_corrs.extend(corr_mat[triu_idx].tolist())
                
                overs = overlap_mat[triu_idx]
                all_overlaps.extend(overs[overs > 0.0].tolist())
        else:
            s_data = db.load_session_calcium_data(
                req.cohort, req.mouse, req.session_type, req.session_name
            )
            sf = s_data["spatial_footprints"]
            tf = s_data["temporal_footprints"]
            n_cells = sf.shape[0]
            if n_cells >= 2:
                centroids = compute_centroids(sf)
                dx = centroids[:, 0:1] - centroids[:, 0:1].T
                dy = centroids[:, 1:2] - centroids[:, 1:2].T
                dist_mat = np.sqrt(dx*dx + dy*dy)
                # Pearson
                tf_means = np.mean(tf, axis=1, keepdims=True)
                tf_stds = np.std(tf, axis=1, keepdims=True)
                tf_stds[tf_stds == 0] = 1.0
                tf_normalized = (tf - tf_means) / tf_stds
                corr_mat = np.dot(tf_normalized, tf_normalized.T) / tf.shape[1]
                
                # Overlaps
                sfs = get_sparse_footprints(sf)
                overlap_mat = np.zeros((n_cells, n_cells))
                for i in range(n_cells):
                    for j in range(i + 1, n_cells):
                        val = compute_overlap(sfs[i], sfs[j], 'cosine')
                        overlap_mat[i, j] = val
                        overlap_mat[j, i] = val
                
                triu_idx = np.triu_indices(n_cells, k=1)
                all_dists.extend(dist_mat[triu_idx].tolist())
                all_corrs.extend(corr_mat[triu_idx].tolist())
                
                overs = overlap_mat[triu_idx]
                all_overlaps.extend(overs[overs > 0.0].tolist())
                
        # Handle cases with no cell pairs
        if not all_dists:
            return {
                "dist_counts": [], "dist_bins": [],
                "corr_counts": [], "corr_bins": [],
                "overlap_counts": [], "overlap_bins": []
            }
            
        dist_hist, dist_edges = np.histogram(all_dists, bins=50, range=(0, 100))
        corr_hist, corr_edges = np.histogram(all_corrs, bins=40, range=(-1, 1))
        overlap_hist, overlap_edges = np.histogram(all_overlaps, bins=40, range=(0.001, 1.0))
        
        return {
            "dist_counts": dist_hist.tolist(),
            "dist_bins": [float((dist_edges[i] + dist_edges[i+1])/2) for i in range(len(dist_hist))],
            "corr_counts": corr_hist.tolist(),
            "corr_bins": [float((corr_edges[i] + corr_edges[i+1])/2) for i in range(len(corr_hist))],
            "overlap_counts": overlap_hist.tolist(),
            "overlap_bins": [float((overlap_edges[i] + overlap_edges[i+1])/2) for i in range(len(overlap_hist))]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class AutoMergeRequest(BaseModel):
    mouse: str
    is_overview: bool
    dist_thresh: float
    corr_thresh: float
    overlap_thresh: float
    cohort: str = ""
    session_type: str = ""
    session_name: str = ""


def duplicate_cell_groups(spatial, temporal, dist_thresh, corr_thresh, overlap_thresh):
    """Return disjoint duplicate components without mutating session data."""
    n_cells = spatial.shape[0]
    if n_cells < 2:
        return []
    centroids = compute_centroids(spatial)
    delta = centroids[:, None, :] - centroids[None, :, :]
    distances = np.linalg.norm(delta, axis=2)
    centered = temporal - np.mean(temporal, axis=1, keepdims=True)
    std = np.std(temporal, axis=1, keepdims=True)
    std[std == 0] = 1.0
    normalized = centered / std
    correlations = normalized @ normalized.T / max(1, temporal.shape[1])
    sparse = get_sparse_footprints(spatial)
    adjacency = (distances <= dist_thresh) & (correlations >= corr_thresh)
    np.fill_diagonal(adjacency, False)
    candidates = np.transpose(np.triu(adjacency, k=1).nonzero())
    adjacency[:] = False
    for left, right in candidates:
        if compute_overlap(sparse[left], sparse[right], "cosine") >= overlap_thresh:
            adjacency[left, right] = adjacency[right, left] = True

    visited = np.zeros(n_cells, dtype=bool)
    groups = []
    for start in range(n_cells):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in np.flatnonzero(adjacency[current]):
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(int(neighbor))
        if len(component) > 1:
            groups.append(sorted(component))
    return groups

@app.post("/api/automerge-sessions")
def run_automerge(req: AutoMergeRequest):
    try:
        db_switched = ensure_write_on_processed_db()
        discarded_counts = {}
        merge_requests = {}
        loaders = {}
        if req.is_overview:
            sessions = db.get_sessions_for_mouse(req.mouse, req.cohort or None)
            for s in sessions:
                data = db.load_session_calcium_data(
                    s["cohort"], s["mouse"], s["session_type"], s["session_name"], warp_cached=False
                )
                groups = duplicate_cell_groups(
                    data["spatial_footprints"], data["temporal_footprints"],
                    req.dist_thresh, req.corr_thresh, req.overlap_thresh,
                )
                key = workspace_key(s["cohort"], s["mouse"], s["session_type"], s["session_name"])
                merge_requests[key] = groups
                loaders[key] = workspace_loader(*key)
        else:
            data = db.load_session_calcium_data(
                req.cohort, req.mouse, req.session_type, req.session_name, warp_cached=False
            )
            groups = duplicate_cell_groups(
                data["spatial_footprints"], data["temporal_footprints"],
                req.dist_thresh, req.corr_thresh, req.overlap_thresh,
            )
            key = workspace_key(req.cohort, req.mouse, req.session_type, req.session_name)
            merge_requests[key] = groups
            loaders[key] = workspace_loader(*key)

        counts = workspace.merge_groups(merge_requests, loaders)
        for key, count in counts.items():
            if count:
                discarded_counts[f"{key[2]}_{key[3]}"] = count
        matching_invalidated = invalidate_matching(req.mouse)
                
        return {
            "status": "success",
            "merged_counts": discarded_counts,
            "db_switched": db_switched,
            "matching_invalidated": matching_invalidated,
            "workspace": workspace.status(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class DeleteSessionRequest(BaseModel):
    cohort: str
    mouse: str
    session_type: str
    session_name: str

@app.post("/api/delete-session")
def delete_session(req: DeleteSessionRequest):
    """Deletes a session from the database entirely."""
    try:
        db_switched = ensure_write_on_processed_db()
        key = workspace_key(req.cohort, req.mouse, req.session_type, req.session_name)
        workspace.delete_session(key, workspace_loader(*key))
        matching_invalidated = invalidate_matching(req.mouse)
        return {
            "status": "success",
            "message": f"Session {req.session_name} staged for deletion.",
            "db_switched": db_switched,
            "matching_invalidated": matching_invalidated,
            "workspace": workspace.status(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== GRID SEARCH RESULTS PERSISTENCE ====================

class GridResultsSaveRequest(BaseModel):
    mouse: str
    cohort: Optional[str] = None
    session_key: str  # e.g. "Reward100pct_Session01" or "_overview"
    rows: List[Dict[str, Any]]

@app.post("/api/grid-search-results/save")
def save_grid_results(req: GridResultsSaveRequest):
    """Saves grid search result rows for a session (or overview) to disk as JSON."""
    try:
        filename = safe_scope_filename(req.cohort, req.mouse, f"{req.session_key}.json")
        filepath = os.path.join(grid_results_dir, filename)
        
        # Load existing rows
        existing = []
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                existing = json.load(f)
        
        # Merge: natural key includes session_name so per-session rows in the overview file
        # are kept separately (not collapsed to one row per combo).
        def row_key(r):
            return (
                r.get("mode", ""),
                r.get("strategy", ""),
                r.get("source", ""),
                r.get("ref_session", ""),
                r.get("session_name", ""),
                float(r.get("demons_smoothing", 1.5)),
            )
        
        existing_map = {row_key(r): i for i, r in enumerate(existing)}
        warnings = []
        
        for new_row in req.rows:
            k = row_key(new_row)
            if k in existing_map:
                old = existing[existing_map[k]]
                # Compare NCC values to decide skip/overwrite
                old_ncc = round(old.get("mip_ncc", 0.0), 4)
                new_ncc = round(new_row.get("mip_ncc", 0.0), 4)
                if abs(old_ncc - new_ncc) < 1e-4:
                    # Same result — skip silently
                    pass
                else:
                    # Different result — overwrite with warning
                    warnings.append(f"Overwriting row for {k}: NCC {old_ncc:.4f} -> {new_ncc:.4f}")
                    existing[existing_map[k]] = new_row
            else:
                existing.append(new_row)
                existing_map[k] = len(existing) - 1
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)
        
        return {"status": "success", "warnings": warnings, "total_rows": len(existing)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/grid-search-results/load")
def load_grid_results(mouse: str, session_key: str, cohort: Optional[str] = None):
    """Loads persisted grid search results for a session or overview from disk."""
    try:
        filename = safe_scope_filename(cohort, mouse, f"{session_key}.json")
        filepath = os.path.join(grid_results_dir, filename)
        legacy_path = os.path.join(grid_results_dir, f"{mouse}_{session_key}.json")
        if not os.path.exists(filepath) and not cohort and os.path.exists(legacy_path):
            filepath = legacy_path
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                rows = json.load(f)
        else:
            rows = []
        return {"status": "success", "rows": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class AlignmentRequest(BaseModel):
    mouse: str
    cohort: Optional[str] = None
    ref_session_idx: int
    alignment_mode: str # 'translation', 'rigid', 'similarity', 'non-rigid'
    alignment_source: str # 'MIP', 'Footprints'
    alignment_strategy: str # 'Direct', 'Sequential'
    demons_smoothing: float
    target_session_name: str = "" # If empty, align all. Otherwise, align only this session.
    session_order: Optional[List[str]] = None

@app.post("/api/run-alignment")
def run_alignment(req: AlignmentRequest):
    with workspace.transaction(
        "Run session alignment", alignment=True, matching=True
    ):
        result = _run_alignment(req)
        sessions = ordered_sessions(req.mouse, req.cohort, req.session_order)
        index_by_name = {
            session["display_name"]: index for index, session in enumerate(sessions)
        }
        for item in result.get("alignments", []):
            active_index = index_by_name.get(item["session_name"])
            if active_index is None:
                continue
            reference = resolve_alignment_reference(
                sessions,
                active_index,
                req.ref_session_idx,
                req.alignment_strategy,
            )
            reference_name = sessions[reference.reference_index]["display_name"]
            root_name = sessions[reference.root_index]["display_name"]
            item.setdefault("reference_session", reference_name)
            item.setdefault("root_session", root_name)
            cached = alignment_cache.get(req.mouse, {}).get(item["session_name"])
            if cached is not None:
                cached["reference_session"] = reference_name
                cached["root_session"] = root_name
                cached["strategy"] = req.alignment_strategy
                cached["accumulated_transform"] = {
                    "dx": float(item.get("dx", 0.0)),
                    "dy": float(item.get("dy", 0.0)),
                    "rotation": float(item.get("rotation", 0.0)),
                    "scale": float(item.get("scale", 1.0)),
                }
                cached["local_transform"] = {
                    "dx": float(item.get("rel_dx", item.get("dx", 0.0))),
                    "dy": float(item.get("rel_dy", item.get("dy", 0.0))),
                    "rotation": float(
                        item.get("rel_rotation", item.get("rotation", 0.0))
                    ),
                    "scale": float(item.get("rel_scale", item.get("scale", 1.0))),
                }
        _attach_dual_source_alignment_ncc(req, sessions, result.get("alignments", []))
    result["workspace"] = workspace.status()
    return result


def _attach_dual_source_alignment_ncc(req, sessions, results):
    """Add MIP and footprint NCC values using the strategy's visual reference."""
    if not results:
        return
    mips = []
    spatial_sums = []
    for session in sessions:
        data = db.load_session_calcium_data(
            session["cohort"], session["mouse"], session["session_type"],
            session["session_name"], warp_cached=False, include_temporal=False,
        )
        mips.append(np.asarray(data["max_projection"]))
        spatial_sums.append(np.sum(data["spatial_footprints"], axis=0))

    index_by_name = {
        session["display_name"]: index for index, session in enumerate(sessions)
    }
    mouse_cache = alignment_cache.get(req.mouse, {})
    for item in results:
        active_index = index_by_name.get(item["session_name"])
        if active_index is None:
            continue
        reference = resolve_alignment_reference(
            sessions, active_index, req.ref_session_idx, req.alignment_strategy
        )
        if active_index == reference.reference_index:
            scores = {"mip_ncc": 1.0, "footprints_ncc": 1.0}
        else:
            active_transform = mouse_cache.get(item["session_name"], {}).get("transform")
            if active_transform is None:
                continue
            reference_transform = None
            if (
                req.alignment_strategy == "Sequential"
                and reference.reference_index != reference.root_index
            ):
                reference_name = sessions[reference.reference_index]["display_name"]
                reference_transform = mouse_cache.get(reference_name, {}).get("transform")
            scores = compute_alignment_nccs(
                mips, spatial_sums, active_index, reference.reference_index,
                req.alignment_mode, active_transform, reference_transform,
                downsample=True,
            )
        item.update(scores)
        cached = mouse_cache.get(item["session_name"])
        if cached is not None:
            cached.update(scores)


def _run_alignment(req: AlignmentRequest):
    """Runs rigid or non-rigid image alignment across sessions for a mouse."""
    try:
        ensure_cache_scope(req.mouse, req.cohort)
        invalidate_matching(req.mouse)
        sessions = ordered_sessions(req.mouse, req.cohort, req.session_order)
        S = len(sessions)
        if S == 0:
            raise ValueError("No calcium sessions found.")
            
        # Load all session projection images
        images = []
        for s in sessions:
            s_data = db.load_session_calcium_data(s["cohort"], s["mouse"], s["session_type"], s["session_name"], warp_cached=False, include_temporal=False)
            if req.alignment_source == 'MIP':
                images.append(np.array(s_data["max_projection"]))
            else: # Footprints
                sf = s_data["spatial_footprints"]
                images.append(np.sum(sf, axis=0))
                
        # Reference Session details
        ref_idx = req.ref_session_idx
        ref_img = images[ref_idx]
        H, W = ref_img.shape
        
        # Initialize in-memory cache for this mouse if not present
        if req.mouse not in alignment_cache:
            alignment_cache[req.mouse] = {}
            
        results = []
        
        # Handle Single Session Alignment Trigger
        if req.target_session_name:
            target_idx = next((i for i, s in enumerate(sessions) if s["display_name"] == req.target_session_name), None)
            if target_idx is None:
                raise ValueError(f"Target session {req.target_session_name} not found.")
                
            if req.alignment_strategy == 'Direct':
                if target_idx == ref_idx:
                    alignment_cache[req.mouse][req.target_session_name] = {
                        "mode": req.alignment_mode,
                        "transform": np.eye(2, 3, dtype=np.float32) if req.alignment_mode != 'non-rigid' else np.zeros((H, W, 2)),
                        "dx": 0.0,
                        "dy": 0.0,
                        "rotation": 0.0,
                        "scale": 1.0,
                        "ncc": 1.0,
                        "rel_dx": 0.0,
                        "rel_dy": 0.0,
                        "rel_rotation": 0.0,
                        "rel_scale": 1.0
                    }
                    return {"status": "success", "alignments": [{"session_name": req.target_session_name, "dx": 0.0, "dy": 0.0, "rotation": 0.0, "scale": 1.0, "ncc": 1.0, "rel_dx": 0.0, "rel_dy": 0.0, "rel_rotation": 0.0, "rel_scale": 1.0}]}
                
                target_img = images[target_idx]
                tformOrD, ncc = register_images(ref_img, target_img, req.alignment_mode, req.demons_smoothing)
                if req.alignment_mode == 'non-rigid':
                    dx, dy = 0.0, 0.0
                    rotation = 0.0
                    scale = 1.0
                else:
                    dx, dy, rotation, scale = decompose_warp_matrix(tformOrD, W / 2.0, H / 2.0)
                    
                alignment_cache[req.mouse][req.target_session_name] = {
                    "mode": req.alignment_mode,
                    "transform": tformOrD,
                    "dx": dx,
                    "dy": dy,
                    "rotation": rotation,
                    "scale": scale,
                    "ncc": float(ncc),
                    "rel_dx": dx,
                    "rel_dy": dy,
                    "rel_rotation": rotation,
                    "rel_scale": scale
                }
                return {"status": "success", "alignments": [{"session_name": req.target_session_name, "dx": dx, "dy": dy, "rotation": rotation, "scale": scale, "ncc": float(ncc), "rel_dx": dx, "rel_dy": dy, "rel_rotation": rotation, "rel_scale": scale}]}
            else:
                # Sequential single-session crawl starting from ref_idx
                accum_tform = np.eye(3, 3, dtype=np.float32)
                accum_disp = None
                alignments_to_return = []
                
                if target_idx > ref_idx:
                    for s in range(ref_idx + 1, target_idx + 1):
                        tformOrD, ncc = register_images(images[s - 1], images[s], req.alignment_mode, req.demons_smoothing)
                        s_name = sessions[s]["display_name"]
                        if req.alignment_mode == 'non-rigid':
                            if accum_disp is None:
                                prev_disp = np.zeros((H, W, 2), dtype=np.float32)
                                accum_disp = np.copy(tformOrD)
                            else:
                                prev_disp = np.copy(accum_disp)
                                accum_disp = compose_displacement_fields(accum_disp, tformOrD)
                            # Warp intermediate and neighbor, then compute NCC
                            warped_act = warp_image_non_rigid(images[s], accum_disp)
                            warped_ref = warp_image_non_rigid(images[s - 1], prev_disp)
                            s_ncc = compute_ncc(warped_ref, warped_act, downsample=True)
                            
                            alignment_cache[req.mouse][s_name] = {
                                "mode": req.alignment_mode,
                                "transform": accum_disp,
                                "dx": 0.0,
                                "dy": 0.0,
                                "rotation": 0.0,
                                "scale": 1.0,
                                "ncc": float(s_ncc)
                            }
                            alignments_to_return.append({
                                "session_name": s_name,
                                "dx": 0.0,
                                "dy": 0.0,
                                "rotation": 0.0,
                                "scale": 1.0,
                                "ncc": float(s_ncc)
                            })
                        else:
                            tform_3x3 = np.vstack([tformOrD, [0, 0, 1]])
                            
                            # Accumulate with RIGHT multiplication: accum = accum @ tform_3x3
                            prev_tform = np.copy(accum_tform)
                            accum_tform = accum_tform @ tform_3x3
                            
                            # Warp intermediate and neighbor, then compute NCC
                            final_tform_act = accum_tform[:2, :]
                            final_tform_ref = prev_tform[:2, :]
                            warped_act = warp_image_rigid(images[s], final_tform_act)
                            warped_ref = warp_image_rigid(images[s - 1], final_tform_ref)
                            s_ncc = compute_ncc(warped_ref, warped_act, downsample=True)
                            
                            accum_dx, accum_dy, rotation, scale = decompose_warp_matrix(final_tform_act, W / 2.0, H / 2.0)
                            rel_dx, rel_dy, rel_rotation, rel_scale = decompose_warp_matrix(tformOrD, W / 2.0, H / 2.0)
                            
                            alignment_cache[req.mouse][s_name] = {
                                "mode": req.alignment_mode,
                                "transform": final_tform_act,
                                "dx": accum_dx,
                                "dy": accum_dy,
                                "rotation": rotation,
                                "scale": scale,
                                "rel_dx": rel_dx,
                                "rel_dy": rel_dy,
                                "rel_rotation": rel_rotation,
                                "rel_scale": rel_scale,
                                "ncc": float(s_ncc)
                            }
                            alignments_to_return.append({
                                "session_name": s_name,
                                "dx": accum_dx,
                                "dy": accum_dy,
                                "rotation": rotation,
                                "scale": scale,
                                "rel_dx": rel_dx,
                                "rel_dy": rel_dy,
                                "rel_rotation": rel_rotation,
                                "rel_scale": rel_scale,
                                "ncc": float(s_ncc)
                            })
                elif target_idx < ref_idx:
                    for s in range(ref_idx - 1, target_idx - 1, -1):
                        tformOrD, ncc = register_images(images[s + 1], images[s], req.alignment_mode, req.demons_smoothing)
                        s_name = sessions[s]["display_name"]
                        if req.alignment_mode == 'non-rigid':
                            if accum_disp is None:
                                prev_disp = np.zeros((H, W, 2), dtype=np.float32)
                                accum_disp = np.copy(tformOrD)
                            else:
                                prev_disp = np.copy(accum_disp)
                                accum_disp = compose_displacement_fields(accum_disp, tformOrD)
                            # Warp intermediate and neighbor, then compute NCC
                            warped_act = warp_image_non_rigid(images[s], accum_disp)
                            warped_ref = warp_image_non_rigid(images[s + 1], prev_disp)
                            s_ncc = compute_ncc(warped_ref, warped_act, downsample=True)
                            
                            alignment_cache[req.mouse][s_name] = {
                                "mode": req.alignment_mode,
                                "transform": accum_disp,
                                "dx": 0.0,
                                "dy": 0.0,
                                "rotation": 0.0,
                                "scale": 1.0,
                                "ncc": float(s_ncc)
                            }
                            alignments_to_return.append({
                                "session_name": s_name,
                                "dx": 0.0,
                                "dy": 0.0,
                                "rotation": 0.0,
                                "scale": 1.0,
                                "ncc": float(s_ncc)
                            })
                        else:
                            tform_3x3 = np.vstack([tformOrD, [0, 0, 1]])
                            
                            # Accumulate with RIGHT multiplication: accum = accum @ tform_3x3
                            prev_tform = np.copy(accum_tform)
                            accum_tform = accum_tform @ tform_3x3
                            
                            # Warp intermediate and neighbor, then compute NCC
                            final_tform_act = accum_tform[:2, :]
                            final_tform_ref = prev_tform[:2, :]
                            warped_act = warp_image_rigid(images[s], final_tform_act)
                            warped_ref = warp_image_rigid(images[s + 1], final_tform_ref)
                            s_ncc = compute_ncc(warped_ref, warped_act, downsample=True)
                            
                            accum_dx, accum_dy, rotation, scale = decompose_warp_matrix(final_tform_act, W / 2.0, H / 2.0)
                            rel_dx, rel_dy, rel_rotation, rel_scale = decompose_warp_matrix(tformOrD, W / 2.0, H / 2.0)
                            
                            alignment_cache[req.mouse][s_name] = {
                                "mode": req.alignment_mode,
                                "transform": final_tform_act,
                                "dx": accum_dx,
                                "dy": accum_dy,
                                "rotation": rotation,
                                "scale": scale,
                                "rel_dx": rel_dx,
                                "rel_dy": rel_dy,
                                "rel_rotation": rel_rotation,
                                "rel_scale": rel_scale,
                                "ncc": float(s_ncc)
                            }
                            alignments_to_return.append({
                                "session_name": s_name,
                                "dx": accum_dx,
                                "dy": accum_dy,
                                "rotation": rotation,
                                "scale": scale,
                                "rel_dx": rel_dx,
                                "rel_dy": rel_dy,
                                "rel_rotation": rel_rotation,
                                "rel_scale": rel_scale,
                                "ncc": float(s_ncc)
                            })
                                
                return {"status": "success", "alignments": alignments_to_return}
                
        # Handle Mouse-Wide All-Sessions Alignment (triggered when activeSession === "overview")
        if req.alignment_strategy == 'Direct':
            for s in range(S):
                sess = sessions[s]
                s_name = sess["display_name"]
                
                if s == ref_idx:
                    alignment_cache[req.mouse][s_name] = {
                        "mode": req.alignment_mode,
                        "transform": np.eye(2, 3, dtype=np.float32) if req.alignment_mode != 'non-rigid' else np.zeros((H, W, 2)),
                        "dx": 0.0,
                        "dy": 0.0,
                        "rotation": 0.0,
                        "scale": 1.0,
                        "ncc": 1.0
                    }
                    results.append({"session_name": s_name, "dx": 0.0, "dy": 0.0, "rotation": 0.0, "scale": 1.0, "ncc": 1.0, "rel_dx": 0.0, "rel_dy": 0.0, "rel_rotation": 0.0, "rel_scale": 1.0})
                    continue
                
                target_img = images[s]
                tformOrD, ncc = register_images(ref_img, target_img, req.alignment_mode, req.demons_smoothing)
                
                if req.alignment_mode == 'non-rigid':
                    dx, dy = 0.0, 0.0
                    rotation = 0.0
                    scale = 1.0
                else:
                    dx, dy, rotation, scale = decompose_warp_matrix(tformOrD, W / 2.0, H / 2.0)
                    
                alignment_cache[req.mouse][s_name] = {
                    "mode": req.alignment_mode,
                    "transform": tformOrD,
                    "dx": dx,
                    "dy": dy,
                    "rotation": rotation,
                    "scale": scale,
                    "ncc": float(ncc),
                    "rel_dx": dx,
                    "rel_dy": dy,
                    "rel_rotation": rotation,
                    "rel_scale": scale
                }
                results.append({"session_name": s_name, "dx": dx, "dy": dy, "rotation": rotation, "scale": scale, "ncc": float(ncc), "rel_dx": dx, "rel_dy": dy, "rel_rotation": rotation, "rel_scale": scale})
        else:
            # Sequential Alignment (accumulated transformations anchored at ref_idx)
            accum_tforms = []
            accum_disps = [] # For non-rigid
            rel_dxs = []
            rel_dys = []
            
            # Precompute all adjacent registrations
            adj_transforms = {} # s -> transform/displacement relative to neighbor towards ref_idx
            for s in range(ref_idx + 1, S):
                ref_neighbor = images[s - 1]
                target_neighbor = images[s]
                tformOrD, _ = register_images(ref_neighbor, target_neighbor, req.alignment_mode, req.demons_smoothing)
                adj_transforms[s] = tformOrD
            for s in range(ref_idx - 1, -1, -1):
                ref_neighbor = images[s + 1]
                target_neighbor = images[s]
                tformOrD, _ = register_images(ref_neighbor, target_neighbor, req.alignment_mode, req.demons_smoothing)
                adj_transforms[s] = tformOrD
                
            for s in range(S):
                if s == ref_idx:
                    accum_tforms.append(np.eye(2, 3, dtype=np.float32))
                    accum_disps.append(np.zeros((H, W, 2), dtype=np.float32))
                    rel_dxs.append(0.0)
                    rel_dys.append(0.0)
                    continue
                
                if req.alignment_mode == 'non-rigid':
                    disp = np.zeros((H, W, 2), dtype=np.float32)
                    if s > ref_idx:
                        for k in range(ref_idx + 1, s + 1):
                            disp = compose_displacement_fields(disp, adj_transforms[k])
                    else:
                        for k in range(ref_idx - 1, s - 1, -1):
                            disp = compose_displacement_fields(disp, adj_transforms[k])
                    accum_tforms.append(np.eye(2, 3, dtype=np.float32))
                    accum_disps.append(disp)
                    rel_dxs.append(0.0)
                    rel_dys.append(0.0)
                else:
                    tform = np.eye(3, 3, dtype=np.float32)
                    if s > ref_idx:
                        for k in range(ref_idx + 1, s + 1):
                            tform_3x3 = np.vstack([adj_transforms[k], [0, 0, 1]])
                            tform = tform @ tform_3x3  # RIGHT multiplication!
                    else:
                        for k in range(ref_idx - 1, s - 1, -1):
                            tform_3x3 = np.vstack([adj_transforms[k], [0, 0, 1]])
                            tform = tform @ tform_3x3  # RIGHT multiplication!
                    accum_tforms.append(np.copy(tform[:2, :]))
                    accum_disps.append(np.zeros((H, W, 2), dtype=np.float32))
                    rel_dxs.append(float(adj_transforms[s][0, 2]))
                    rel_dys.append(float(adj_transforms[s][1, 2]))
                    
            for s in range(S):
                sess = sessions[s]
                s_name = sess["display_name"]
                
                # In sequential mode, compare with neighbor reference session (both warped to global ref space)
                # except for ref_idx itself which matches itself (ncc = 1.0)
                if s == ref_idx:
                    ncc = 1.0
                    neighbor_idx = ref_idx
                else:
                    neighbor_idx = resolve_alignment_reference(
                        sessions, s, ref_idx, "Sequential"
                    ).reference_index
                    if req.alignment_mode == 'non-rigid':
                        disp_act = accum_disps[s]
                        disp_ref = accum_disps[neighbor_idx]
                        warped_act = warp_image_non_rigid(images[s], disp_act)
                        warped_ref = warp_image_non_rigid(images[neighbor_idx], disp_ref)
                    else:
                        tform_act = accum_tforms[s]
                        tform_ref = accum_tforms[neighbor_idx]
                        warped_act = warp_image_rigid(images[s], tform_act)
                        warped_ref = warp_image_rigid(images[neighbor_idx], tform_ref)
                    ncc = compute_ncc(warped_ref, warped_act, downsample=True)
                    
                if req.alignment_mode == 'non-rigid':
                    alignment_cache[req.mouse][s_name] = {
                        "mode": req.alignment_mode,
                        "transform": accum_disps[s],
                        "dx": 0.0,
                        "dy": 0.0,
                        "rotation": 0.0,
                        "scale": 1.0,
                        "ncc": float(ncc)
                    }
                    results.append({
                        "session_name": s_name,
                        "dx": 0.0,
                        "dy": 0.0,
                        "rotation": 0.0,
                        "scale": 1.0,
                        "ncc": float(ncc),
                        "rel_dx": 0.0,
                        "rel_dy": 0.0,
                        "rel_rotation": 0.0,
                        "rel_scale": 1.0
                    })
                else:
                    tform = accum_tforms[s]
                    dx, dy, rotation, scale = decompose_warp_matrix(tform, W / 2.0, H / 2.0)
                    
                    alignment_cache[req.mouse][s_name] = {
                        "mode": req.alignment_mode,
                        "transform": tform,
                        "dx": dx,
                        "dy": dy,
                        "rotation": rotation,
                        "scale": scale,
                        "rel_dx": float(rel_dxs[s]),
                        "rel_dy": float(rel_dys[s]),
                        "ncc": float(ncc)
                    }
                    rel_dx_val = float(rel_dxs[s])
                    rel_dy_val = float(rel_dys[s])
                    # For sequential, decompose the adjacent transform to get rel rotation/scale
                    if s != ref_idx and s in adj_transforms and req.alignment_mode != 'non-rigid':
                        adj_rel_dx, adj_rel_dy, rel_rot, rel_scl = decompose_warp_matrix(adj_transforms[s], W / 2.0, H / 2.0)
                    else:
                        rel_rot = rotation
                        rel_scl = scale
                    results.append({
                        "session_name": s_name,
                        "reference_session": sessions[neighbor_idx]["display_name"],
                        "root_session": sessions[ref_idx]["display_name"],
                        "dx": dx,
                        "dy": dy,
                        "rotation": rotation,
                        "scale": scale,
                        "rel_dx": rel_dx_val,
                        "rel_dy": rel_dy_val,
                        "rel_rotation": rel_rot,
                        "rel_scale": rel_scl,
                        "ncc": float(ncc)
                    })
                
        return {"status": "success", "alignments": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class GridSearchRequest(BaseModel):
    mouse: str
    cohort: Optional[str] = None
    modes: List[str]           # e.g. ["translation", "rigid"]
    strategies: List[str]      # e.g. ["Direct", "Sequential"]
    sources: List[str]         # e.g. ["MIP", "Footprints"]
    ref_session_indices: List[int]   # indices into sessions list
    target_session_name: Optional[str] = ""
    demons_smoothing: float = 1.5
    session_order: Optional[List[str]] = None

@app.post("/api/run-grid-search")
def run_grid_search(req: GridSearchRequest):
    """Evaluates alignment quality metrics across all selected parameter combinations (Cartesian product).
    
    For each (mode, strategy, source, ref_session_idx) combination:
    - Runs image registration for every target session
    - Returns per-session rows with MIP NCC, Footprints NCC, displacement, and shift breakdown
    """
    try:
        sessions = ordered_sessions(req.mouse, req.cohort, req.session_order)
        S = len(sessions)
        if S == 0:
            raise ValueError("No calcium sessions found.")

        # Load MaxProjections (MIP) and Footprints Sum (SF) for all sessions
        mips = []
        sfs = []
        for s in sessions:
            s_data = db.load_session_calcium_data(
                s["cohort"], s["mouse"], s["session_type"], s["session_name"],
                warp_cached=False, include_temporal=False,
            )
            mips.append(np.array(s_data["max_projection"]))
            sf_stack = s_data["spatial_footprints"]
            sfs.append(np.sum(sf_stack, axis=0))

        img_H, img_W = mips[0].shape

        # Decide scope
        is_overview = not req.target_session_name
        if is_overview:
            target_indices = list(range(S))
        else:
            t_idx = next(
                (i for i, s in enumerate(sessions) if s["display_name"] == req.target_session_name),
                None
            )
            if t_idx is None:
                raise ValueError(f"Session '{req.target_session_name}' not found.")
            target_indices = [t_idx]

        all_results = []  # flat list of result rows
        registration_cache = {}

        def cached_registration(mode, source, reference_index, target_index):
            key = (
                mode,
                source,
                int(reference_index),
                int(target_index),
                float(req.demons_smoothing),
            )
            if key not in registration_cache:
                source_images = mips if source == "MIP" else sfs
                registration_cache[key] = register_images(
                    source_images[reference_index],
                    source_images[target_index],
                    mode,
                    req.demons_smoothing,
                )
            return registration_cache[key]

        for mode, strategy, source, ref_idx in itertools.product(
            req.modes, req.strategies, req.sources, req.ref_session_indices
        ):
            if ref_idx < 0 or ref_idx >= S:
                continue

            ref_session_name = sessions[ref_idx]["display_name"]

            for t_idx in target_indices:
                s_name = sessions[t_idx]["display_name"]

                # Reference session is trivially aligned to itself
                if t_idx == ref_idx:
                    all_results.append({
                        "session_name": s_name,
                        "mode": mode,
                        "strategy": strategy,
                        "source": source,
                        "ref_session": ref_session_name,
                        "demons_smoothing": req.demons_smoothing,
                        "mip_ncc": 1.0,
                        "footprints_ncc": 1.0,
                        "displacement": 0.0,
                        "accum_dx": 0.0,
                        "accum_dy": 0.0,
                        "accum_rotation": 0.0,
                        "accum_scale": 1.0,
                        "rel_dx": 0.0,
                        "rel_dy": 0.0,
                        "rel_rotation": 0.0,
                        "rel_scale": 1.0
                    })
                    continue

                accum_dx = 0.0
                accum_dy = 0.0
                accum_rot = 0.0
                accum_scl = 1.0
                rel_dx = 0.0
                rel_dy = 0.0
                rel_rot = 0.0
                rel_scl = 1.0
                mean_disp = 0.0

                if strategy == "Direct":
                    tformOrD, _ = cached_registration(mode, source, ref_idx, t_idx)
                    if mode == 'non-rigid':
                        disp = tformOrD
                        mean_disp = float(np.mean(np.hypot(disp[:, :, 0], disp[:, :, 1])))
                        mip_warped = warp_image_non_rigid(mips[t_idx], disp)
                        sf_warped = warp_image_non_rigid(sfs[t_idx], disp)
                    else:
                        W = tformOrD
                        pts = np.array([[0, 0, 1], [img_W, 0, 1], [0, img_H, 1], [img_W, img_H, 1], [img_W/2, img_H/2, 1]], dtype=np.float32)
                        mean_disp = float(np.mean([np.hypot(W[0]@pt - pt[0], W[1]@pt - pt[1]) for pt in pts]))
                        accum_dx, accum_dy, accum_rot, accum_scl = decompose_warp_matrix(W, img_W / 2.0, img_H / 2.0)
                        # For Direct strategy, relative == accumulated
                        rel_dx, rel_dy, rel_rot, rel_scl = accum_dx, accum_dy, accum_rot, accum_scl
                        mip_warped = warp_image_rigid(mips[t_idx], W)
                        sf_warped = warp_image_rigid(sfs[t_idx], W)
                else:
                    # Sequential: crawl step by step from ref_idx to t_idx
                    accum_tform = np.eye(3, 3, dtype=np.float32)
                    accum_disp = None
                    last_adj_tform = None
                    comparison_tform = None
                    comparison_disp = None

                    if t_idx > ref_idx:
                        step_range = range(ref_idx + 1, t_idx + 1)
                    else:
                        step_range = range(ref_idx - 1, t_idx - 1, -1)

                    for k in step_range:
                        neighbor_index = resolve_alignment_reference(
                            sessions, k, ref_idx, "Sequential"
                        ).reference_index
                        adj_tformOrD, _ = cached_registration(mode, source, neighbor_index, k)
                        if mode == 'non-rigid':
                            comparison_disp = (
                                np.zeros((img_H, img_W, 2), dtype=np.float32)
                                if accum_disp is None else np.copy(accum_disp)
                            )
                            if accum_disp is None:
                                accum_disp = np.copy(adj_tformOrD)
                            else:
                                accum_disp = compose_displacement_fields(accum_disp, adj_tformOrD)
                        else:
                            comparison_tform = np.copy(accum_tform[:2, :])
                            tform_3x3 = np.vstack([adj_tformOrD, [0, 0, 1]])
                            accum_tform = accum_tform @ tform_3x3
                            last_adj_tform = adj_tformOrD

                    if mode == 'non-rigid':
                        if accum_disp is None:
                            accum_disp = np.zeros((img_H, img_W, 2), dtype=np.float32)
                        mean_disp = float(np.mean(np.hypot(accum_disp[:, :, 0], accum_disp[:, :, 1])))
                        mip_warped = warp_image_non_rigid(mips[t_idx], accum_disp)
                        sf_warped = warp_image_non_rigid(sfs[t_idx], accum_disp)
                    else:
                        tform_2x3 = accum_tform[:2, :]
                        pts = np.array([[0, 0, 1], [img_W, 0, 1], [0, img_H, 1], [img_W, img_H, 1], [img_W/2, img_H/2, 1]], dtype=np.float32)
                        mean_disp = float(np.mean([np.hypot(tform_2x3[0]@pt - pt[0], tform_2x3[1]@pt - pt[1]) for pt in pts]))
                        accum_dx, accum_dy, accum_rot, accum_scl = decompose_warp_matrix(tform_2x3, img_W / 2.0, img_H / 2.0)
                        if last_adj_tform is not None:
                            rel_dx, rel_dy, rel_rot, rel_scl = decompose_warp_matrix(last_adj_tform, img_W / 2.0, img_H / 2.0)
                        else:
                            rel_dx, rel_dy, rel_rot, rel_scl = accum_dx, accum_dy, accum_rot, accum_scl
                        mip_warped = warp_image_rigid(mips[t_idx], tform_2x3)
                        sf_warped = warp_image_rigid(sfs[t_idx], tform_2x3)

                if strategy == "Sequential":
                    comparison_idx = resolve_alignment_reference(
                        sessions, t_idx, ref_idx, "Sequential"
                    ).reference_index
                    scores = compute_alignment_nccs(
                        mips,
                        sfs,
                        t_idx,
                        comparison_idx,
                        mode,
                        accum_disp if mode == "non-rigid" else tform_2x3,
                        comparison_disp if mode == "non-rigid" else comparison_tform,
                        downsample=True,
                    )
                    mip_ncc = scores["mip_ncc"]
                    fp_ncc = scores["footprints_ncc"]
                else:
                    mip_ncc = float(compute_ncc(mips[ref_idx], mip_warped))
                    fp_ncc = float(compute_ncc(sfs[ref_idx], sf_warped))

                all_results.append({
                    "session_name": s_name,
                    "mode": mode,
                    "strategy": strategy,
                    "source": source,
                    "ref_session": ref_session_name,
                    "demons_smoothing": req.demons_smoothing,
                    "mip_ncc": mip_ncc,
                    "footprints_ncc": fp_ncc,
                    "displacement": mean_disp,
                    "accum_dx": accum_dx,
                    "accum_dy": accum_dy,
                    "accum_rotation": accum_rot,
                    "accum_scale": accum_scl,
                    "rel_dx": rel_dx,
                    "rel_dy": rel_dy,
                    "rel_rotation": rel_rot,
                    "rel_scale": rel_scl
                })

        return {"status": "success", "results": all_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class SaveAlignmentRequest(BaseModel):
    mouse: str
    cohort: Optional[str] = None
    nudge_shifts: Dict[str, List[float]] # { "Reward100pct_Session03": [dx, dy] }
    nudge_rotations: Optional[Dict[str, float]] = None # { "Reward100pct_Session03": rot_angle }
    nudge_scales: Optional[Dict[str, float]] = None # { "Reward100pct_Session03": scale_factor }

@app.post("/api/save-alignment-warps")
def save_alignment_warps(req: SaveAlignmentRequest):
    """Saves alignments to HDF5. Warps images/footprints in database for rigid/similarity/non-rigid modes."""
    try:
        ensure_cache_scope(req.mouse, req.cohort)
        # A save button is a checkpoint for all cross-tab working changes.
        db.save_workspace()
        sessions = db.get_sessions_for_mouse(req.mouse, req.cohort)
        mouse_cache = alignment_cache.get(req.mouse, {})
        
        nudge_rots = req.nudge_rotations or {}
        nudge_scls = req.nudge_scales or {}
        
        for s in sessions:
            s_name = s["display_name"]
            dx, dy = req.nudge_shifts.get(s_name, [0.0, 0.0])
            rot_angle = nudge_rots.get(s_name, 0.0)
            scale_factor = nudge_scls.get(s_name, 1.0)
            
            # Check if alignment was run in cache
            if s_name in mouse_cache:
                mode = mouse_cache[s_name]["mode"]
                transform = mouse_cache[s_name]["transform"]
            else:
                mode = "translation"
                transform = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
                
            # Call DB save aligned warps
            db.save_aligned_warps(
                s["cohort"], s["mouse"], s["session_type"], s["session_name"],
                transform, mode, dx, dy, rot_angle, scale_factor
            )
            
        # Clear cache
        if req.mouse in alignment_cache:
            del alignment_cache[req.mouse]
        workspace.clear()
            
        return {"status": "success", "message": "Alignments applied and written back to database successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class MatchingRequest(BaseModel):
    mouse: str
    cohort: Optional[str] = None
    max_dist: float
    min_overlap: float
    cost_weight: float
    overlap_type: str
    session_order: Optional[List[str]] = None

@app.post("/api/run-matching")
def run_matching(req: MatchingRequest):
    with workspace.transaction("Run cell matching", matching=True):
        result = _run_matching(req)
    result["workspace"] = workspace.status()
    return result


def _run_matching(req: MatchingRequest):
    """Executes multi-session cell matching, utilizing active alignment shifts (nudge adjustments included)."""
    try:
        ensure_cache_scope(req.mouse, req.cohort)
        sessions = ordered_sessions(req.mouse, req.cohort, req.session_order)
        S = len(sessions)
        if S == 0:
            raise ValueError("No calcium sessions found.")
            
        # Build sessions list with shifted centroids and footprints
        processed_sessions = []
        session_names = []
        alignment_shifts = {}
        
        for s in sessions:
            s_name = s["display_name"]
            session_names.append(s_name)
            
            # Retrieve cached alignment if present
            cached_align = None
            if req.mouse in alignment_cache and s_name in alignment_cache[req.mouse]:
                cached_align = alignment_cache[req.mouse][s_name]
                
            s_data = db.load_session_calcium_data(
                s["cohort"], s["mouse"], s["session_type"], s["session_name"],
                cached_alignment=cached_align, warp_cached=True, include_temporal=False,
            )
            sf = s_data["spatial_footprints"]
            n_cells = sf.shape[0]
            
            # Already warped on-the-fly, so report active shifts but apply 0 shift to arrays
            dx_report = float(s_data["alignment_shift"][0])
            dy_report = float(s_data["alignment_shift"][1])
            alignment_shifts[s_name] = [dx_report, dy_report]
            
            # Compute centroids (already warped)
            shifted_centroids = compute_centroids(sf)
            
            # Extract sparse footprints (already warped)
            raw_fps = get_sparse_footprints(sf)
            
            processed_sessions.append({
                "session_name": s_name,
                "n_cells": n_cells,
                "centroids": shifted_centroids,
                "sparse_footprints": raw_fps
            })
            
        # Run matching loop
        matching_matrix, master_centroids, master_footprints = match_cells_across_sessions(
            processed_sessions, req.max_dist, req.min_overlap, req.cost_weight, req.overlap_type
        )
        
        # Save in memory cache
        matching_cache[req.mouse] = {
            "cohort": req.cohort,
            "matching_matrix": matching_matrix,
            "master_centroids": master_centroids,
            "master_footprints": master_footprints,
            "session_names": session_names,
            "alignment_shifts": alignment_shifts,
            "session_centroids": [
                sess["centroids"] for sess in processed_sessions
            ],
            "params": {
                "max_dist": req.max_dist,
                "min_overlap": req.min_overlap,
                "cost_weight": req.cost_weight,
                "overlap_type": req.overlap_type
            }
        }
        
        # Compute matching statistics distributions
        sess_centroids_array = [sess["centroids"] for sess in processed_sessions]
        sess_footprints_list = [sess["sparse_footprints"] for sess in processed_sessions]
        within_dists, between_dists, within_overs, between_overs = compute_matching_distributions(
            matching_matrix,
            sess_centroids_array,
            sess_footprints_list,
            master_footprints,
            master_centroids,
            req.overlap_type
        )
        
        # Centroid drift bins: 0 to 40 px, steps of 1px
        dist_bins = np.arange(0, 41, 1)
        within_dist_counts, _ = np.histogram(within_dists, bins=dist_bins)
        between_dist_counts, _ = np.histogram(between_dists, bins=dist_bins)
        
        # Overlap bins: 0.05 to 1.0, steps of 0.05 (excluding 0% overlap)
        overlap_bins = np.arange(0.05, 1.05, 0.05)
        within_overlap_counts, _ = np.histogram(within_overs, bins=overlap_bins)
        between_overlap_counts, _ = np.histogram(between_overs, bins=overlap_bins)
        
        # Prepare response (convert NaNs to nulls in JSON)
        clean_matrix = matching_matrix.tolist()
        clean_matrix_serializable = [[None if np.isnan(val) else int(val) for val in row] for row in clean_matrix]
        
        session_centroids_list = [arr.tolist() for arr in sess_centroids_array]
        matching_summary = summarize_matching(matching_matrix)
        
        return {
            "status": "success",
            "matching_matrix": clean_matrix_serializable,
            "master_centroids": np.nan_to_num(master_centroids, nan=0.0).tolist(),
            "session_centroids": session_centroids_list,
            **matching_summary,
            "dist_bins": dist_bins.tolist(),
            "within_dist_counts": within_dist_counts.tolist(),
            "between_dist_counts": between_dist_counts.tolist(),
            "overlap_bins": overlap_bins.tolist(),
            "within_overlap_counts": within_overlap_counts.tolist(),
            "between_overlap_counts": between_overlap_counts.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class OptimizeRequest(BaseModel):
    mouse: str
    cohort: Optional[str] = None
    target_count: int
    overlap_type: str
    session_order: Optional[List[str]] = None

@app.post("/api/optimize-matching")
def optimize_matching(req: OptimizeRequest):
    """Grid searches cost weight, centroid distance, and overlap thresholds to meet target cell count."""
    try:
        ensure_cache_scope(req.mouse, req.cohort)
        sessions = ordered_sessions(req.mouse, req.cohort, req.session_order)
        processed_sessions = []
        
        for s in sessions:
            s_name = s["display_name"]
            cached_align = None
            if req.mouse in alignment_cache and s_name in alignment_cache[req.mouse]:
                cached_align = alignment_cache[req.mouse][s_name]
            s_data = db.load_session_calcium_data(
                s["cohort"],
                s["mouse"],
                s["session_type"],
                s["session_name"],
                cached_alignment=cached_align,
                warp_cached=True,
                include_temporal=False,
            )
            sf = s_data["spatial_footprints"]
            n_cells = sf.shape[0]
            shifted_centroids = compute_centroids(sf)
            shifted_fps = get_sparse_footprints(sf)
                
            processed_sessions.append({
                "session_name": s_name,
                "n_cells": n_cells,
                "centroids": shifted_centroids,
                "sparse_footprints": shifted_fps
            })
            
        opt_params = optimize_matching_parameters(processed_sessions, req.target_count, req.overlap_type)
        return {"status": "success", "parameters": opt_params}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def hydrate_matching_cache(mouse, cohort, cache):
    """Fill optional fields when a saved matching file is reopened and edited."""
    session_names = cache["session_names"]
    sessions = ordered_sessions(mouse, cohort, session_names)
    if [session["display_name"] for session in sessions[:len(session_names)]] != list(session_names):
        raise ValueError("Saved matching sessions do not match the selected cohort database.")
    sessions = sessions[:len(session_names)]
    alignment_shifts = {}
    sparse_by_session = []
    for session in sessions:
        display_name = session["display_name"]
        cached_align = alignment_cache.get(mouse, {}).get(display_name)
        data = db.load_session_calcium_data(
            session["cohort"],
            session["mouse"],
            session["session_type"],
            session["session_name"],
            cached_alignment=cached_align,
            warp_cached=True,
            include_temporal=False,
        )
        alignment_shifts[display_name] = [float(value) for value in data["alignment_shift"]]
        sparse_by_session.append(get_sparse_footprints(data["spatial_footprints"]))

    matrix = cache["matching_matrix"]
    if len(cache.get("master_footprints", [])) != matrix.shape[0]:
        master_footprints = []
        for row in matrix:
            footprint = None
            for column, local_value in enumerate(row):
                if np.isnan(local_value) or column >= len(sparse_by_session):
                    continue
                local_index = int(local_value)
                if 0 <= local_index < len(sparse_by_session[column]):
                    footprint = sparse_by_session[column][local_index]
                    break
            if footprint is None:
                footprint = {
                    "idx": np.empty(0, dtype=np.uint32),
                    "vals": np.empty(0, dtype=np.float64),
                    "norm": 1.0,
                }
            master_footprints.append(footprint)
        cache["master_footprints"] = master_footprints
        cache["session_centroids"] = sess_centroids_array
    cache["alignment_shifts"] = alignment_shifts


@app.post("/api/save-matching")
def save_matching(req: Dict[str, str]):
    """Commits matched cell indexing alignments in-place and saves independent CellMatching MAT files."""
    try:
        mouse = req.get("mouse")
        cohort = req.get("cohort") or None
        ensure_cache_scope(mouse, cohort)
        if not mouse or mouse not in matching_cache:
            raise ValueError(f"No active matching results cached for mouse: {mouse}")
            
        cache = matching_cache[mouse]
        hydrate_matching_cache(mouse, cohort, cache)
        db.save_workspace()
        
        save_path = db.save_matching_results(
            mouse_name=mouse,
            matching_matrix=cache["matching_matrix"],
            session_names=cache["session_names"],
            alignment_shifts=cache["alignment_shifts"],
            master_centroids=cache["master_centroids"],
            master_footprints=cache["master_footprints"],
            params=cache["params"],
            cohort_name=req.get("cohort") or None,
        )
        
        # Clear cache
        del matching_cache[mouse]
        workspace.clear()
        
        return {"status": "success", "message": f"Matching results saved successfully to {os.path.basename(db.processed_db_path)} and {os.path.basename(save_path)}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import h5py

def load_cell_matching_mat(save_path: str):
    """Loads CellMatching mat file, supporting both standard v7 and v7.3 HDF5 formats."""
    try:
        mat_data = scipy.io.loadmat(save_path)
        struct = mat_data["CellMatching"]
        matching_matrix = struct["MatchingMatrix"][0, 0] - 1.0
        master_centroids = struct["MasterCentroids"][0, 0]
        sess_names_raw = struct["SessionNames"][0, 0]
        session_names = []
        for name in sess_names_raw:
            if isinstance(name, np.ndarray) and len(name) > 0:
                session_names.append(str(name[0]))
            else:
                session_names.append(str(name))
        return matching_matrix, master_centroids, session_names
    except NotImplementedError:
        with h5py.File(save_path, 'r') as f:
            grp = f["CellMatching"]
            matching_matrix = np.transpose(grp["MatchingMatrix"][:]) - 1.0
            master_centroids = np.transpose(grp["MasterCentroids"][:])
            sn_ds = grp["SessionNames"]
            session_names = []
            for ref_array in sn_ds:
                for ref in ref_array:
                    obj = f[ref]
                    char_data = obj[:]
                    s_name = "".join(chr(c) for c in char_data.flatten())
                    session_names.append(s_name)
            return matching_matrix, master_centroids, session_names

@app.get("/api/master-cell-footprints/{mouse}/{m_idx}")
def get_master_cell_footprints(mouse: str, m_idx: int, cohort: Optional[str] = None):
    """Fetches spatial footprints (warped) of a specific master cell across all sessions where it is present."""
    try:
        ensure_cache_scope(mouse, cohort)
        if mouse not in matching_cache:
            save_path = cell_matching_path(mouse, cohort)
            if os.path.exists(save_path):
                m_matrix, m_centroids, session_names = load_cell_matching_mat(save_path)
                matching_cache[mouse] = {
                    "cohort": cohort,
                    "matching_matrix": m_matrix,
                    "master_centroids": m_centroids,
                    "session_names": session_names,
                    "master_footprints": [],
                    "params": {
                        "max_dist": 12.0,
                        "min_overlap": 0.15,
                        "cost_weight": 0.5,
                        "overlap_type": "cosine"
                    }
                }
        
        if mouse in matching_cache:
            cache = matching_cache[mouse]
            matching_matrix = cache["matching_matrix"]
            session_names = cache["session_names"]
        else:
            return {"status": "error", "message": "No matching session active or saved."}
            
        sessions = ordered_sessions(mouse, cohort, session_names)
        
        results = []
        for s_idx, s in enumerate(sessions):
            if s_idx >= matching_matrix.shape[1]:
                continue
            local_idx_val = matching_matrix[m_idx, s_idx]
            if np.isnan(local_idx_val) or local_idx_val < 0:
                continue
                
            local_idx = int(local_idx_val)
            
            # Load calcium data with warps applied on-the-fly
            cached_align = None
            if mouse in alignment_cache and s["display_name"] in alignment_cache[mouse]:
                cached_align = alignment_cache[mouse][s["display_name"]]
                
            s_data = db.load_session_calcium_data(
                s["cohort"], s["mouse"], s["session_type"], s["session_name"],
                cached_alignment=cached_align, warp_cached=True, include_temporal=False,
            )
            
            sf_stack = s_data["spatial_footprints"]
            if local_idx < sf_stack.shape[0]:
                footprint = sf_stack[local_idx] # (H, W)
                max_val = np.max(footprint)
                thr = 0.01 * max_val
                if thr <= 0:
                    thr = 0.01
                r_idx, c_idx = np.where(footprint > thr)
                vals = footprint[r_idx, c_idx]
                
                results.append({
                    "session_name": s["display_name"],
                    "rows": r_idx.tolist(),
                    "cols": c_idx.tolist(),
                    "vals": vals.tolist(),
                    "centroid": [float(np.mean(c_idx)), float(np.mean(r_idx))] if len(c_idx) > 0 else [0.0, 0.0]
                })
                
        return {"status": "success", "footprints": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class CollisionsRequest(BaseModel):
    mouse: str
    cohort: Optional[str] = None
    master_indices: List[int]
    overlap_type: str = "cosine"

@app.post("/api/matching-collisions")
def get_matching_collisions(req: CollisionsRequest):
    try:
        mouse = req.mouse
        ensure_cache_scope(mouse, req.cohort)
        if mouse not in matching_cache:
            save_path = cell_matching_path(mouse, req.cohort)
            if os.path.exists(save_path):
                m_matrix, m_centroids, session_names = load_cell_matching_mat(save_path)
                matching_cache[mouse] = {
                    "cohort": req.cohort,
                    "matching_matrix": m_matrix,
                    "master_centroids": m_centroids,
                    "session_names": session_names,
                    "master_footprints": [],
                    "params": {
                        "max_dist": 12.0,
                        "min_overlap": 0.15,
                        "cost_weight": 0.5,
                        "overlap_type": "cosine"
                    }
                }
        
        if mouse in matching_cache:
            cache = matching_cache[mouse]
            matching_matrix = cache["matching_matrix"]
            session_names = cache["session_names"]
        else:
            return {"status": "success", "collisions": []}
        
        sessions = ordered_sessions(mouse, req.cohort, session_names)
        M, S = matching_matrix.shape
        
        collisions = []
        for s_idx, s in enumerate(sessions):
            if s_idx >= S:
                continue
            
            present_m_idxs = []
            for m_idx in req.master_indices:
                if m_idx < M:
                    val = matching_matrix[m_idx, s_idx]
                    if not np.isnan(val) and val >= 0:
                        present_m_idxs.append((m_idx, int(val)))
                        
            if len(present_m_idxs) >= 2:
                try:
                    s_data = db.load_session_calcium_data(s["cohort"], s["mouse"], s["session_type"], s["session_name"])
                    sf = s_data["spatial_footprints"]
                    tf = s_data["temporal_footprints"]
                    centroids = compute_centroids(sf)
                except Exception as e:
                    print(f"Error loading calcium data for collision stats in {s['session_name']}: {e}")
                    continue
                
                for k in range(len(present_m_idxs)):
                    for l in range(k + 1, len(present_m_idxs)):
                        m_idx1, local_idx1 = present_m_idxs[k]
                        m_idx2, local_idx2 = present_m_idxs[l]
                        
                        if local_idx1 < sf.shape[0] and local_idx2 < sf.shape[0]:
                            dist = float(np.linalg.norm(centroids[local_idx1] - centroids[local_idx2]))
                            
                            if tf.shape[1] > 0:
                                corr = np.corrcoef(tf[local_idx1], tf[local_idx2])[0, 1]
                                corr_val = float(corr) if not np.isnan(corr) else 0.0
                            else:
                                corr_val = 0.0
                                
                            from .matching import get_sparse_footprints
                            fp1 = get_sparse_footprints(sf[local_idx1:local_idx1+1])[0]
                            fp2 = get_sparse_footprints(sf[local_idx2:local_idx2+1])[0]
                            overlap = float(compute_overlap(fp1, fp2, req.overlap_type))
                            
                            collisions.append({
                                "cohort": s["cohort"],
                                "mouse": s["mouse"],
                                "session_type": s["session_type"],
                                "session_name": s["session_name"],
                                "display_name": s["display_name"],
                                "master_idx1": m_idx1,
                                "master_idx2": m_idx2,
                                "local_idx1": local_idx1,
                                "local_idx2": local_idx2,
                                "distance": dist,
                                "correlation": corr_val,
                                "overlap": overlap
                            })
                            
        return {"status": "success", "collisions": collisions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class IntraSessionMerge(BaseModel):
    cohort: str
    mouse: str
    session_type: str
    session_name: str
    cell_indices: List[int]

class CommitMergesRequest(BaseModel):
    mouse: str
    cohort: Optional[str] = None
    intra_session_merges: List[IntraSessionMerge]
    inter_session_merges: List[List[int]]


class DeleteMatchingGroupsRequest(BaseModel):
    mouse: str
    cohort: Optional[str] = None
    master_indices: List[int]


@app.post("/api/delete-matching-groups")
def delete_matching_groups(req: DeleteMatchingGroupsRequest):
    with workspace.transaction("Delete selected matched cell groups", matching=True):
        result = _delete_matching_groups(req)
    result["workspace"] = workspace.status()
    return result


def _delete_matching_groups(req: DeleteMatchingGroupsRequest):
    ensure_cache_scope(req.mouse, req.cohort)
    if req.mouse not in matching_cache:
        raise ValueError("Run or load cell matching before deleting groups.")
    cache = matching_cache[req.mouse]
    session_names = list(cache["session_names"])
    sessions = ordered_sessions(req.mouse, req.cohort, session_names)
    if len(sessions) != len(session_names):
        raise ValueError("Matching sessions no longer match the active database.")

    matrix, master_centroids, master_footprints, deleted_by_column = (
        delete_master_cell_groups(
            cache["matching_matrix"], cache["master_centroids"],
            cache.get("master_footprints", []), req.master_indices,
        )
    )
    discard_requests = {}
    loaders = {}
    for column, local_indices in enumerate(deleted_by_column):
        if not local_indices:
            continue
        session = sessions[column]
        key = workspace_key(
            session["cohort"], session["mouse"], session["session_type"],
            session["session_name"],
        )
        discard_requests[key] = local_indices
        loaders[key] = workspace_loader(*key)
    workspace.discard_indices(
        discard_requests, loaders, label="Delete cells in selected matching groups"
    )

    session_centroids = []
    session_footprints = []
    for session in sessions:
        cached_align = alignment_cache.get(req.mouse, {}).get(session["display_name"])
        data = db.load_session_calcium_data(
            session["cohort"], session["mouse"], session["session_type"],
            session["session_name"], cached_alignment=cached_align,
            warp_cached=True, include_temporal=False,
        )
        spatial = data["spatial_footprints"]
        session_centroids.append(compute_centroids(spatial))
        session_footprints.append(get_sparse_footprints(spatial))

    if len(master_footprints) != matrix.shape[0]:
        rebuilt_footprints = []
        for row in matrix:
            footprint = None
            for column, value in enumerate(row):
                if np.isnan(value) or value < 0:
                    continue
                local_index = int(value)
                if local_index < len(session_footprints[column]):
                    footprint = session_footprints[column][local_index]
                    break
            rebuilt_footprints.append(footprint)
        master_footprints = rebuilt_footprints

    for master_index in range(matrix.shape[0]):
        coordinates = []
        for column in range(matrix.shape[1]):
            value = matrix[master_index, column]
            if np.isnan(value) or value < 0:
                continue
            local_index = int(value)
            if local_index < len(session_centroids[column]):
                coordinates.append(session_centroids[column][local_index])
        if coordinates:
            master_centroids[master_index] = np.mean(coordinates, axis=0)

    updated_cache = dict(cache)
    updated_cache.update({
        "matching_matrix": matrix,
        "master_centroids": master_centroids,
        "master_footprints": master_footprints,
        "session_centroids": session_centroids,
    })
    matching_cache[req.mouse] = updated_cache

    within_dists, between_dists, within_overs, between_overs = (
        compute_matching_distributions(
            matrix, session_centroids, session_footprints, master_footprints,
            master_centroids,
            updated_cache.get("params", {}).get("overlap_type", "cosine"),
        )
    )
    dist_bins = np.arange(0, 41, 1)
    overlap_bins = np.arange(0.05, 1.05, 0.05)
    within_dist_counts, _ = np.histogram(within_dists, bins=dist_bins)
    between_dist_counts, _ = np.histogram(between_dists, bins=dist_bins)
    within_overlap_counts, _ = np.histogram(within_overs, bins=overlap_bins)
    between_overlap_counts, _ = np.histogram(between_overs, bins=overlap_bins)
    serializable_matrix = [
        [None if np.isnan(value) or value < 0 else int(value) for value in row]
        for row in matrix
    ]
    return {
        "status": "success",
        "deleted_groups": len(set(req.master_indices)),
        "deleted_cells": sum(len(values) for values in deleted_by_column),
        "matching_matrix": serializable_matrix,
        "master_centroids": np.nan_to_num(master_centroids, nan=0.0).tolist(),
        "session_centroids": [values.tolist() for values in session_centroids],
        "dist_bins": dist_bins.tolist(),
        "within_dist_counts": within_dist_counts.tolist(),
        "between_dist_counts": between_dist_counts.tolist(),
        "overlap_bins": overlap_bins.tolist(),
        "within_overlap_counts": within_overlap_counts.tolist(),
        "between_overlap_counts": between_overlap_counts.tolist(),
        **summarize_matching(matrix),
    }

@app.post("/api/commit-matching-merges")
def commit_matching_merges(req: CommitMergesRequest):
    with workspace.transaction("Merge matched cell tracks", matching=True):
        result = _commit_matching_merges(req)
    result["workspace"] = workspace.status()
    return result


def _commit_matching_merges(req: CommitMergesRequest):
    try:
        mouse = req.mouse
        ensure_cache_scope(mouse, req.cohort)
        
        if mouse not in matching_cache:
            save_path = cell_matching_path(mouse, req.cohort)
            if os.path.exists(save_path):
                m_matrix, m_centroids, session_names = load_cell_matching_mat(save_path)
                matching_cache[mouse] = {
                    "cohort": req.cohort,
                    "matching_matrix": m_matrix,
                    "master_centroids": m_centroids,
                    "session_names": session_names,
                    "master_footprints": [],
                    "params": {
                        "max_dist": 12.0,
                        "min_overlap": 0.15,
                        "cost_weight": 0.5,
                        "overlap_type": "cosine"
                    }
                }
                
        cache = matching_cache[mouse]
        matching_matrix = cache["matching_matrix"].copy()
        master_centroids = cache["master_centroids"].copy()
        master_footprints = cache.get("master_footprints", [])
        session_names = cache["session_names"]
        
        sessions = ordered_sessions(mouse, req.cohort, session_names)

        groups_by_key = {}
        loaders = {}
        for m in req.intra_session_merges:
            key = workspace_key(m.cohort, m.mouse, m.session_type, m.session_name)
            groups_by_key.setdefault(key, []).append(m.cell_indices)
            loaders[key] = workspace_loader(*key)

        # Pairwise collision suggestions can overlap. Collapse them into
        # disjoint connected components before mutating indices.
        normalized_by_key = {}
        for key, groups in groups_by_key.items():
            components = []
            for group in groups:
                pending = set(int(index) for index in group)
                overlapping = [component for component in components if component & pending]
                for component in overlapping:
                    pending.update(component)
                    components.remove(component)
                if len(pending) > 1:
                    components.append(pending)
            normalized_by_key[key] = [sorted(component) for component in components]

        session_columns = {
            workspace_key(s["cohort"], s["mouse"], s["session_type"], s["session_name"]): index
            for index, s in enumerate(sessions)
        }
        for key, groups in normalized_by_key.items():
            s_col = session_columns.get(key, -1)
            if s_col < 0 or s_col >= matching_matrix.shape[1]:
                continue
            for cell_indices in sorted(groups, key=lambda values: values[0], reverse=True):
                keep_idx = cell_indices[0]
                delete_idxs = sorted(cell_indices[1:], reverse=True)
                for master_index in range(matching_matrix.shape[0]):
                    value = matching_matrix[master_index, s_col]
                    if not np.isnan(value) and int(value) in cell_indices:
                        matching_matrix[master_index, s_col] = keep_idx
                for delete_index in delete_idxs:
                    mask = matching_matrix[:, s_col] > delete_index
                    matching_matrix[mask, s_col] -= 1
                    
        inter_components = []
        for group in req.inter_session_merges:
            pending = set(int(index) for index in group)
            overlapping = [component for component in inter_components if component & pending]
            for component in overlapping:
                pending.update(component)
                inter_components.remove(component)
            if len(pending) > 1:
                inter_components.append(pending)

        rows_to_remove = set()
        for component in inter_components:
            group = sorted(component)
            if group[-1] >= matching_matrix.shape[0]:
                continue
            target_row = group[0]
            other_rows = group[1:]
            for s_col in range(matching_matrix.shape[1]):
                local_values = {
                    int(matching_matrix[row_index, s_col])
                    for row_index in group
                    if not np.isnan(matching_matrix[row_index, s_col])
                    and matching_matrix[row_index, s_col] >= 0
                }
                if len(local_values) > 1:
                    raise ValueError(
                        f"Master rows {group} still contain an unresolved intra-session collision "
                        f"in {session_names[s_col]}."
                    )
            for other_row in other_rows:
                for s_col in range(matching_matrix.shape[1]):
                    val_target = matching_matrix[target_row, s_col]
                    val_other = matching_matrix[other_row, s_col]
                    is_target_empty = np.isnan(val_target) or val_target < 0
                    is_other_present = not np.isnan(val_other) and val_other >= 0
                    if is_target_empty and is_other_present:
                        matching_matrix[target_row, s_col] = val_other
            rows_to_remove.update(other_rows)

        if rows_to_remove:
            keep_mask = np.ones(matching_matrix.shape[0], dtype=bool)
            keep_mask[list(rows_to_remove)] = False
            matching_matrix = matching_matrix[keep_mask, :]
            master_centroids = master_centroids[keep_mask, :]
            if len(master_footprints) > 0:
                master_footprints = [master_footprints[idx] for idx in range(len(master_footprints)) if keep_mask[idx]]

        if normalized_by_key:
            workspace.merge_groups(
                normalized_by_key,
                loaders,
                label="Resolve matching intra-session collisions",
            )
                
        sess_centroids_array = []
        for s in sessions:
            try:
                s_data = db.load_session_calcium_data(s["cohort"], s["mouse"], s["session_type"], s["session_name"], include_temporal=False)
                sf = s_data["spatial_footprints"]
                centroids = compute_centroids(sf)
                sess_centroids_array.append(centroids)
            except Exception as exc:
                print(f"Could not recompute centroids for {s['display_name']}: {exc}")
                sess_centroids_array.append(None)
                
        for m_idx in range(matching_matrix.shape[0]):
            coords = []
            for s_col in range(matching_matrix.shape[1]):
                val = matching_matrix[m_idx, s_col]
                if not np.isnan(val) and val >= 0 and s_col < len(sess_centroids_array) and sess_centroids_array[s_col] is not None:
                    local_idx = int(val)
                    if local_idx < len(sess_centroids_array[s_col]):
                        coords.append(sess_centroids_array[s_col][local_idx])
            if len(coords) > 0:
                master_centroids[m_idx] = np.mean(coords, axis=0)
                
        cache["matching_matrix"] = matching_matrix
        cache["master_centroids"] = master_centroids
        cache["master_footprints"] = master_footprints
        
        clean_matrix = matching_matrix.tolist()
        clean_matrix_serializable = [[None if np.isnan(val) or val < 0 else int(val) for val in row] for row in clean_matrix]
        session_centroids_list = [arr.tolist() if arr is not None else [] for arr in sess_centroids_array]
        
        matching_summary = summarize_matching(matching_matrix)
        return {
            "status": "success",
            "matching_matrix": clean_matrix_serializable,
            "master_centroids": np.nan_to_num(master_centroids, nan=0.0).tolist(),
            "session_centroids": session_centroids_list,
            **matching_summary,
            "workspace": workspace.status(),
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Server index.html directly from FastAPI root
@app.get("/", response_class=HTMLResponse)
def read_root():
    frontend_path = os.path.join(os.path.dirname(__file__), "frontend", "index.html")
    if os.path.exists(frontend_path):
        with open(frontend_path, "r", encoding="utf-8") as f:
            return f.read()
    return """
    <html>
        <head><title>Calcium Imaging Dashboard</title></head>
        <body style="font-family: sans-serif; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; background-color: #0f172a; color: white;">
            <h1>Calcium Imaging Cell Processing Dashboard Backend Running</h1>
            <p>Please ensure that frontend/index.html is created.</p>
        </body>
    </html>
    """

@app.get("/documentation", response_class=HTMLResponse)
def read_documentation():
    doc_path = os.path.join(os.path.dirname(__file__), "frontend", "documentation.html")
    if os.path.exists(doc_path):
        with open(doc_path, "r", encoding="utf-8") as f:
            return f.read()
    return """
    <html>
        <head><title>Documentation Not Found</title></head>
        <body style="font-family: sans-serif; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; background-color: #0f172a; color: white;">
            <h1>Documentation HTML File Not Found</h1>
            <p>Please ensure that frontend/documentation.html is created.</p>
        </body>
    </html>
    """

def _port_is_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def run() -> None:
    parser = argparse.ArgumentParser(
        prog="cell-registration-dashboard",
        description="Launch the calcium-imaging cell-registration dashboard.",
    )
    parser.add_argument(
        "--database",
        type=os.path.abspath,
        help="Optional .mat, .h5, or .hdf5 database to load at startup.",
    )
    parser.add_argument("--port", type=int, default=8002, help="Local web port (default: 8002).")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args()

    if args.database:
        if not os.path.isfile(args.database):
            parser.error(f"database file not found: {args.database}")
        load_database({"db_path": args.database})

    host = "127.0.0.1"
    if not _port_is_available(host, args.port):
        parser.error(
            f"port {args.port} is already in use; close the other application "
            "or choose another port with --port"
        )

    Timer(1.2, lambda: webbrowser.open(f"http://{host}:{args.port}")).start()
    uvicorn.run(app, host=host, port=args.port, reload=False)


if __name__ == "__main__":
    run()
