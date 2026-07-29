"""
main.py — FastAPI backend for the Database Builder dashboard.

Endpoints:
  GET  /                        → Serve index.html
  GET  /api/browse-folder       → Open OS folder picker (tkinter), return path
  GET  /api/browse-file         → Open OS file-save picker (tkinter), return path
  POST /api/discover            → Walk up from analysis folder, return depth list
  POST /api/preview-regex       → Test a regex pattern against a sample string
  POST /api/build               → Build HDF5 database (background thread)
  GET  /api/build-status        → Poll build progress log lines
  POST /api/save-config         → Save mapping config to JSON
  POST /api/load-config         → Load mapping config from JSON

Run with:
  db-builder
"""

import os
import argparse
import socket
import sys
import re
import json
import queue
import threading
import webbrowser
from threading import Timer
from typing import Dict, List, Optional, Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from calcium_imaging_dashboard import __version__
from .builder import (
    ANALYSIS_SUFFIXES,
    DEFAULT_ANALYSIS_PATTERNS,
    build,
    discover_sessions,
    find_analysis_target,
)
from .config import save_config, load_config
from .loader import DataNotFoundError

# ──────────────────────────────────────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Calcium Imaging Database Builder")

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")

# ──────────────────────────────────────────────────────────────────────────────
# Global build state
# ──────────────────────────────────────────────────────────────────────────────

_log_queue: queue.Queue = queue.Queue()
_build_running: bool = False
_build_done: bool = False
_build_success: Optional[bool] = None   # True/False/None


# ──────────────────────────────────────────────────────────────────────────────
# Static files & HTML
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def serve_index():
    html_path = os.path.join(FRONTEND_DIR, "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


# ──────────────────────────────────────────────────────────────────────────────
# File / folder browser (tkinter — runs server-side, local use only)
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/api/browse-folder")
def browse_folder(title: str = "Select Folder"):
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askdirectory(title=title)
        root.destroy()
        if not path:
            return {"status": "cancelled"}
        return {"status": "success", "path": os.path.normpath(path).replace("\\", "/")}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/browse-save-file")
def browse_save_file(title: str = "Save Database As", default_ext: str = ".h5"):
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.asksaveasfilename(
            title=title,
            defaultextension=default_ext,
            filetypes=[("HDF5 database", "*.h5"), ("All files", "*.*")],
        )
        root.destroy()
        if not path:
            return {"status": "cancelled"}
        return {"status": "success", "path": os.path.normpath(path).replace("\\", "/")}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/browse-open-file")
def browse_open_file(title: str = "Open Config File"):
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title=title,
            filetypes=[("JSON config", "*.json"), ("All files", "*.*")],
        )
        root.destroy()
        if not path:
            return {"status": "cancelled"}
        return {"status": "success", "path": os.path.normpath(path).replace("\\", "/")}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/browse-stimulus-file")
def browse_stimulus_file(title: str = "Select Combined Stimulus File"):
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title=title,
            filetypes=[
                ("MAT/HDF5 files", "*.mat *.h5 *.hdf5"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        if not path:
            return {"status": "cancelled"}
        return {"status": "success", "path": os.path.normpath(path).replace("\\", "/")}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ──────────────────────────────────────────────────────────────────────────────
# Discovery
# ──────────────────────────────────────────────────────────────────────────────

class DiscoverRequest(BaseModel):
    analysis_folder: str
    analysis_type: str
    analysis_pattern: Optional[str] = None


@app.post("/api/discover")
def api_discover(req: DiscoverRequest):
    """
    Validates a representative session folder, then walks upward so the user
    can choose the hierarchy root and map every folder depth.

    Returns:
        root_path  : str         (the directory containing everything)
        depths     : list[str]   path segments from root down to the
                                 immediate PARENT of the -analysis folder
        sample     : list[str]   a representative folder name at each depth
    """
    session_folder = os.path.normpath(req.analysis_folder)
    if not os.path.isdir(session_folder):
        raise HTTPException(
            status_code=400,
            detail=f"Session folder not found: {session_folder}",
        )
    pattern = (
        (req.analysis_pattern or "").strip()
        or DEFAULT_ANALYSIS_PATTERNS.get(req.analysis_type.lower(), "")
    )
    try:
        selected_result, match_count = find_analysis_target(session_folder, pattern)
    except (DataNotFoundError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    parent = session_folder

    # Collect all path parts from drive root to parent
    parts = []
    current = parent
    while True:
        head, tail = os.path.split(current)
        if tail:
            parts.append(tail)
            current = head
        else:
            # On Windows, head == drive letter like "Z:\", tail == ""
            # On Unix, head == "/" when we're at root
            parts.append(current)  # include drive / root
            break
    parts.reverse()  # outermost first

    # Now scan for ALL -analysis folders under the parent tree to collect samples
    # at each depth below whatever root the user picks.
    # We return the *full* parts list so the UI can let the user pick the root.

    # For each discovered -analysis folder sibling, collect representative names
    # We need to find the root such that analysis folders sit under it.
    # Strategy: scan upward from `parent` to find a directory that contains
    # multiple -analysis siblings (or just return all parts and let UI handle it).

    # Find all analysis-folder parents by scanning upward a few levels
    # to identify where sibling sessions live
    potential_roots = []
    cur = parent
    for _ in range(10):
        cur_parent = os.path.dirname(cur)
        if cur_parent == cur:
            break
        potential_roots.append(cur_parent)
        cur = cur_parent

    # Build depth info: segments relative to each possible root
    # We return all path parts and let the UI pick the root depth
    return {
        "all_parts":      parts,           # all path parts from drive root
        "analysis_type":  req.analysis_type,
        "analysis_folder": session_folder.replace("\\", "/"),
        "analysis_pattern": pattern,
        "selected_result": selected_result.replace("\\", "/"),
        "match_count": match_count,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Regex preview
# ──────────────────────────────────────────────────────────────────────────────

class RegexPreviewRequest(BaseModel):
    pattern: str
    sample:  str


@app.post("/api/preview-regex")
def api_preview_regex(req: RegexPreviewRequest):
    try:
        m = re.match(req.pattern, req.sample)
        if not m:
            return {"matched": False, "groups": []}
        return {"matched": True, "groups": list(m.groups())}
    except re.error as exc:
        return {"matched": False, "error": str(exc), "groups": []}


# ──────────────────────────────────────────────────────────────────────────────
# Scan (resolve all sessions from root + mapping, return preview list)
# ──────────────────────────────────────────────────────────────────────────────

class ScanRequest(BaseModel):
    root_path:     str
    analysis_type: str
    depth_rules:   List[Dict[str, Any]]
    global_values: Dict[str, Optional[str]]
    analysis_pattern: Optional[str] = None


@app.post("/api/scan")
def api_scan(req: ScanRequest):
    """Dry-run: discover sessions and apply mapping, return preview without loading data."""
    from .builder import apply_mapping

    sessions = discover_sessions(
        req.root_path,
        req.analysis_type,
        depth_rules=req.depth_rules,
        analysis_pattern=req.analysis_pattern,
    )
    preview = []
    destination_counts: Dict[str, int] = {}
    for s in sessions:
        mapping = apply_mapping(s, req.depth_rules, req.global_values)
        destination = None
        if mapping is not None:
            destination = "/".join(
                mapping[field]
                for field in ("CohortName", "MouseName", "SessionType", "SessionNumber")
            )
            destination_counts[destination] = destination_counts.get(destination, 0) + 1
        preview.append({
            "rel_path":   "/".join(s["rel_parts"]) if s["rel_parts"] else "(root)",
            "analysis_dir": s["analysis_dir"].replace("\\", "/"),
            "mapping":    mapping,
            "ok":         mapping is not None,
            "destination": destination,
        })
    for item in preview:
        item["collision"] = bool(
            item["destination"]
            and destination_counts[item["destination"]] > 1
        )
        if item["collision"]:
            item["ok"] = False
    return {
        "sessions": preview,
        "collisions": [
            destination
            for destination, count in destination_counts.items()
            if count > 1
        ],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Build
# ──────────────────────────────────────────────────────────────────────────────

class BuildRequest(BaseModel):
    root_path:     str
    analysis_type: str
    output_path:   str
    depth_rules:   List[Dict[str, Any]]
    global_values: Dict[str, Optional[str]]
    precision: str = "float64"
    compression: str = "gzip"
    compression_level: int = 4
    append_policy: str = "replace"
    stimulus_mode: str = "none"
    stimulus_table_pattern: Optional[str] = None
    stimulus_combined_path: Optional[str] = None
    analysis_pattern: Optional[str] = None


def _run_build(req: BuildRequest):
    global _build_running, _build_done, _build_success

    def cb(msg: str):
        _log_queue.put(msg)

    try:
        build(
            output_path=req.output_path,
            root_path=req.root_path,
            analysis_type=req.analysis_type,
            depth_rules=req.depth_rules,
            global_values=req.global_values,
            progress_callback=cb,
            precision=req.precision,
            compression=req.compression,
            compression_level=req.compression_level,
            append_policy=req.append_policy,
            stimulus_mode=req.stimulus_mode,
            stimulus_table_pattern=req.stimulus_table_pattern,
            stimulus_combined_path=req.stimulus_combined_path,
            analysis_pattern=req.analysis_pattern,
        )
        _build_success = True
    except Exception as exc:
        _log_queue.put(f"✗  Fatal error: {exc}")
        _build_success = False
    finally:
        _build_running = False
        _build_done = True


@app.post("/api/build")
def api_build(req: BuildRequest):
    global _build_running, _build_done, _build_success, _log_queue

    if _build_running:
        raise HTTPException(status_code=409, detail="A build is already in progress.")

    # Reset state
    _log_queue = queue.Queue()
    _build_running = True
    _build_done = False
    _build_success = None

    thread = threading.Thread(target=_run_build, args=(req,), daemon=True)
    thread.start()
    return {"status": "started"}


@app.get("/api/build-status")
def api_build_status():
    lines = []
    while not _log_queue.empty():
        try:
            lines.append(_log_queue.get_nowait())
        except queue.Empty:
            break
    return {
        "running": _build_running,
        "done":    _build_done,
        "success": _build_success,
        "lines":   lines,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Config save / load
# ──────────────────────────────────────────────────────────────────────────────

class SaveConfigRequest(BaseModel):
    path:          str
    analysis_type: str
    root_depth:    int
    depth_rules:   List[Dict[str, Any]]
    global_values: Dict[str, Optional[str]]
    stimulus_mode: str = "none"
    stimulus_table_pattern: Optional[str] = None
    stimulus_combined_path: Optional[str] = None
    analysis_pattern: Optional[str] = None


@app.post("/api/save-config")
def api_save_config(req: SaveConfigRequest):
    try:
        payload = {
            "analysis_type": req.analysis_type,
            "root_depth":    req.root_depth,
            "depth_rules":   req.depth_rules,
            "global_values": req.global_values,
            "stimulus_mode": req.stimulus_mode,
            "stimulus_table_pattern": req.stimulus_table_pattern,
            "stimulus_combined_path": req.stimulus_combined_path,
            "analysis_pattern": req.analysis_pattern,
        }
        save_config(req.path, payload)
        return {"status": "success"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


class LoadConfigRequest(BaseModel):
    path: str


@app.post("/api/load-config")
def api_load_config(req: LoadConfigRequest):
    try:
        cfg = load_config(req.path)
        return {"status": "success", "config": cfg}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def _port_is_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def run() -> None:
    parser = argparse.ArgumentParser(
        prog="db-builder",
        description="Build a portable calcium-imaging HDF5 database.",
    )
    parser.add_argument("--port", type=int, default=8001, help="Local web port (default: 8001).")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args()

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
