"""
config.py — Save and load depth-mapping configurations as JSON files.
"""

import json
import os
from typing import Any, Dict


def save_config(path: str, payload: Dict[str, Any]) -> None:
    """
    Writes a mapping config to disk as JSON.

    Expected payload keys:
        analysis_type   : str
        root_depth      : int   (how many levels up from the analysis folder is the root)
        depth_rules     : list of dicts, one per depth level:
            {
                "label":      str   ("CohortName" | "MouseName" | "SessionType" |
                                     "SessionNumber" | "Ignore" | "Split"),
                "split_regex":   str | null,
                "split_fields":  [str, str] | null   e.g. ["SessionType","SessionNumber"]
            }
        global_values   : dict  { "CohortName": str|null, "MouseName": str|null,
                                   "SessionType": str|null, "SessionNumber": str|null }
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_config(path: str) -> Dict[str, Any]:
    """Loads and returns a config dict from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
