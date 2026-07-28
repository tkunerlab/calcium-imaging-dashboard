"""In-memory editing workspace shared by all dashboard tabs.

The workspace keeps stable cell records and command snapshots made of references
to those records. Discard/undo therefore does not duplicate full footprint and
trace arrays. Only newly merged cells allocate new arrays.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from uuid import uuid4

import numpy as np


SessionKey = Tuple[str, str, str, str]


@dataclass(frozen=True)
class CellRecord:
    cell_id: str
    lineage: Tuple[str, ...]
    spatial: np.ndarray
    temporal: np.ndarray


@dataclass
class SessionState:
    key: SessionKey
    cells: List[CellRecord]
    spatial_shape: Tuple[int, int]
    temporal_length: int
    revision: int = 0
    saved_revision: int = 0
    saved_cells: List[CellRecord] = field(default_factory=list)

    @classmethod
    def from_arrays(
        cls, key: SessionKey, spatial: np.ndarray, temporal: np.ndarray
    ) -> "SessionState":
        records = []
        for index in range(spatial.shape[0]):
            cell_id = f"original:{index}"
            records.append(
                CellRecord(
                    cell_id=cell_id,
                    lineage=(cell_id,),
                    spatial=spatial[index],
                    temporal=temporal[index],
                )
            )
        return cls(
            key=key,
            cells=records,
            spatial_shape=(int(spatial.shape[1]), int(spatial.shape[2])),
            temporal_length=int(temporal.shape[1]),
            saved_cells=list(records),
        )

    def is_dirty(self) -> bool:
        return [cell.cell_id for cell in self.cells] != [cell.cell_id for cell in self.saved_cells]

    def arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.cells:
            # Preserve dimensions for an intentionally empty session.
            return (
                np.empty((0, *self.spatial_shape), dtype=np.float64),
                np.empty((0, self.temporal_length), dtype=np.float64),
            )
        return (
            np.stack([cell.spatial for cell in self.cells], axis=0),
            np.stack([cell.temporal for cell in self.cells], axis=0),
        )


@dataclass
class WorkspaceCommand:
    label: str
    before_cells: Dict[SessionKey, List[CellRecord]] = field(default_factory=dict)
    after_cells: Dict[SessionKey, List[CellRecord]] = field(default_factory=dict)
    before_deleted: Dict[SessionKey, bool] = field(default_factory=dict)
    after_deleted: Dict[SessionKey, bool] = field(default_factory=dict)
    before_alignment: Optional[Dict[str, Any]] = None
    after_alignment: Optional[Dict[str, Any]] = None
    before_matching: Optional[Dict[str, Any]] = None
    after_matching: Optional[Dict[str, Any]] = None

    def domains(self) -> set[str]:
        domains: set[str] = set()
        if self.before_cells or self.before_deleted:
            domains.add("cells")
        if self.before_alignment is not None:
            domains.add("alignment")
        if self.before_matching is not None:
            domains.add("matching")
        return domains


@dataclass(frozen=True)
class WorkspaceSaveSnapshot:
    revision: int
    cells: Dict[SessionKey, Tuple[np.ndarray, np.ndarray]]
    deleted: set[SessionKey]
    alignment: Dict[str, Any]
    matching: Dict[str, Any]


class EditWorkspace:
    """Thread-safe, process-local edit state for the currently loaded database."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._states: Dict[SessionKey, SessionState] = {}
        self._deleted: set[SessionKey] = set()
        self._undo: List[WorkspaceCommand] = []
        self._redo: List[WorkspaceCommand] = []
        self.alignment_state: Dict[str, Any] = {}
        self.matching_state: Dict[str, Any] = {}
        self._transaction_command: Optional[WorkspaceCommand] = None
        self.revision = 0

    @staticmethod
    def key(cohort: str, animal: str, session_type: str, session: str) -> SessionKey:
        return cohort, animal, session_type, session

    def clear(self) -> None:
        with self._lock:
            self._states.clear()
            self._deleted.clear()
            self._undo.clear()
            self._redo.clear()
            self.alignment_state.clear()
            self.matching_state.clear()
            self._transaction_command = None
            self.revision = 0

    @staticmethod
    def _values_equal(left: Any, right: Any) -> bool:
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            try:
                return bool(np.array_equal(left, right, equal_nan=True))
            except TypeError:
                return bool(np.array_equal(left, right))
        if isinstance(left, dict) and isinstance(right, dict):
            return left.keys() == right.keys() and all(
                EditWorkspace._values_equal(left[key], right[key]) for key in left
            )
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            return len(left) == len(right) and all(
                EditWorkspace._values_equal(a, b) for a, b in zip(left, right)
            )
        try:
            result = left == right
            return bool(np.all(result)) if isinstance(result, np.ndarray) else bool(result)
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _replace_mapping(target: Dict[str, Any], value: Dict[str, Any]) -> None:
        target.clear()
        target.update(deepcopy(value))

    def _apply_before(self, command: WorkspaceCommand) -> None:
        for key, cells in command.before_cells.items():
            self._states[key].cells = list(cells)
            self._states[key].revision += 1
        for key, deleted in command.before_deleted.items():
            if deleted:
                self._deleted.add(key)
            else:
                self._deleted.discard(key)
        if command.before_alignment is not None:
            self._replace_mapping(self.alignment_state, command.before_alignment)
        if command.before_matching is not None:
            self._replace_mapping(self.matching_state, command.before_matching)

    def _apply_after(self, command: WorkspaceCommand) -> None:
        for key, cells in command.after_cells.items():
            self._states[key].cells = list(cells)
            self._states[key].revision += 1
        for key, deleted in command.after_deleted.items():
            if deleted:
                self._deleted.add(key)
            else:
                self._deleted.discard(key)
        if command.after_alignment is not None:
            self._replace_mapping(self.alignment_state, command.after_alignment)
        if command.after_matching is not None:
            self._replace_mapping(self.matching_state, command.after_matching)

    @contextmanager
    def transaction(
        self,
        label: str,
        *,
        alignment: bool = False,
        matching: bool = False,
    ) -> Iterator[None]:
        """Group cell and derived-domain mutations into one undoable command."""
        with self._lock:
            if self._transaction_command is not None:
                yield
                return
            command = WorkspaceCommand(label=label)
            if alignment:
                command.before_alignment = deepcopy(self.alignment_state)
            if matching:
                command.before_matching = deepcopy(self.matching_state)
            self._transaction_command = command

        try:
            yield
        except Exception:
            with self._lock:
                self._apply_before(command)
                self._transaction_command = None
            raise
        else:
            with self._lock:
                if command.before_alignment is not None:
                    command.after_alignment = deepcopy(self.alignment_state)
                if command.before_matching is not None:
                    command.after_matching = deepcopy(self.matching_state)
                changed = bool(command.before_cells or command.before_deleted)
                changed = changed or (
                    command.before_alignment is not None
                    and not self._values_equal(
                        command.before_alignment, command.after_alignment
                    )
                )
                changed = changed or (
                    command.before_matching is not None
                    and not self._values_equal(command.before_matching, command.after_matching)
                )
                self._transaction_command = None
                if changed:
                    self._undo.append(command)
                    self._redo.clear()
                    self.revision += 1

    def _state(
        self,
        key: SessionKey,
        loader: Callable[[], Tuple[np.ndarray, np.ndarray]],
    ) -> SessionState:
        state = self._states.get(key)
        if state is None:
            spatial, temporal = loader()
            state = SessionState.from_arrays(key, spatial, temporal)
            self._states[key] = state
        return state

    def overlay(
        self,
        key: SessionKey,
        spatial: np.ndarray,
        temporal: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        with self._lock:
            state = self._states.get(key)
            return state.arrays() if state is not None else (spatial, temporal)

    def arrays_if_loaded(self, key: SessionKey) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        with self._lock:
            state = self._states.get(key)
            return state.arrays() if state is not None else None

    def is_deleted(self, key: SessionKey) -> bool:
        with self._lock:
            return key in self._deleted

    def _commit(self, command: WorkspaceCommand) -> None:
        # Cell edits invalidate matching. Keep that invalidation in the same
        # command so Undo restores the exact prior matching result.
        if (command.before_cells or command.before_deleted) and self.matching_state:
            command.before_matching = deepcopy(self.matching_state)
            self.matching_state.clear()
            command.after_matching = {}
        for key, cells in command.after_cells.items():
            state = self._states[key]
            state.cells = list(cells)
            state.revision += 1
        for key, deleted in command.after_deleted.items():
            if deleted:
                self._deleted.add(key)
            else:
                self._deleted.discard(key)
        if self._transaction_command is not None:
            transaction = self._transaction_command
            for key, cells in command.before_cells.items():
                transaction.before_cells.setdefault(key, list(cells))
            transaction.after_cells.update(
                {key: list(cells) for key, cells in command.after_cells.items()}
            )
            for key, deleted in command.before_deleted.items():
                transaction.before_deleted.setdefault(key, deleted)
            transaction.after_deleted.update(command.after_deleted)
            if command.before_matching is not None and transaction.before_matching is None:
                transaction.before_matching = command.before_matching
            if command.after_matching is not None:
                transaction.after_matching = command.after_matching
            return
        self._undo.append(command)
        self._redo.clear()
        self.revision += 1

    def update_alignments(
        self,
        mouse: str,
        cohort: Optional[str],
        alignments: Dict[str, Dict[str, Any]],
        *,
        label: str = "Adjust alignment",
    ) -> None:
        with self.transaction(label, alignment=True, matching=True):
            mouse_state = self.alignment_state.setdefault(mouse, {})
            if cohort:
                mouse_state["__cohort__"] = cohort
            for session_name, values in alignments.items():
                current = mouse_state.setdefault(session_name, {})
                current.update(deepcopy(values))
                current["accumulated_transform"] = {
                    "dx": float(current.get("dx", 0.0)),
                    "dy": float(current.get("dy", 0.0)),
                    "rotation": float(current.get("rotation", 0.0)),
                    "scale": float(current.get("scale", 1.0)),
                }
            self.matching_state.pop(mouse, None)

    def replace_matching(
        self,
        mouse: str,
        value: Dict[str, Any],
        *,
        label: str = "Update cell matching",
    ) -> None:
        with self.transaction(label, matching=True):
            self.matching_state[mouse] = deepcopy(value)

    def replace_cells(
        self,
        changes: Dict[SessionKey, Sequence[CellRecord]],
        loaders: Dict[SessionKey, Callable[[], Tuple[np.ndarray, np.ndarray]]],
        label: str,
    ) -> None:
        with self._lock:
            command = WorkspaceCommand(label=label)
            for key, new_cells in changes.items():
                state = self._state(key, loaders[key])
                command.before_cells[key] = list(state.cells)
                command.after_cells[key] = list(new_cells)
            self._commit(command)

    def discard_indices(
        self,
        requests: Dict[SessionKey, Iterable[int]],
        loaders: Dict[SessionKey, Callable[[], Tuple[np.ndarray, np.ndarray]]],
        keep_selected: bool = False,
        label: str = "Discard cells",
    ) -> Dict[SessionKey, int]:
        with self._lock:
            changes: Dict[SessionKey, List[CellRecord]] = {}
            counts: Dict[SessionKey, int] = {}
            for key, indices in requests.items():
                state = self._state(key, loaders[key])
                selected = {int(index) for index in indices if 0 <= int(index) < len(state.cells)}
                if keep_selected:
                    cells = [cell for index, cell in enumerate(state.cells) if index in selected]
                else:
                    cells = [cell for index, cell in enumerate(state.cells) if index not in selected]
                counts[key] = len(state.cells) - len(cells)
                if len(cells) != len(state.cells):
                    changes[key] = cells
            if changes:
                self.replace_cells(changes, loaders, label)
            return counts

    def merge_indices(
        self,
        key: SessionKey,
        indices: Sequence[int],
        loader: Callable[[], Tuple[np.ndarray, np.ndarray]],
        label: str = "Merge cells",
    ) -> int:
        with self._lock:
            state = self._state(key, loader)
            unique = sorted({int(index) for index in indices})
            if len(unique) < 2 or unique[0] < 0 or unique[-1] >= len(state.cells):
                return len(state.cells)

            selected = [state.cells[index] for index in unique]
            merged_spatial = np.mean(
                np.stack([cell.spatial for cell in selected], axis=0), axis=0
            )
            max_value = float(np.max(merged_spatial)) if merged_spatial.size else 0.0
            if max_value > 0:
                merged_spatial = merged_spatial / max_value
            merged_temporal = np.max(
                np.stack([cell.temporal for cell in selected], axis=0), axis=0
            )
            lineage = tuple(item for cell in selected for item in cell.lineage)
            merged = CellRecord(
                cell_id=f"merged:{uuid4().hex}",
                lineage=lineage,
                spatial=merged_spatial,
                temporal=merged_temporal,
            )

            selected_set = set(unique)
            first = unique[0]
            new_cells: List[CellRecord] = []
            for index, cell in enumerate(state.cells):
                if index == first:
                    new_cells.append(merged)
                elif index not in selected_set:
                    new_cells.append(cell)
            self.replace_cells({key: new_cells}, {key: loader}, label)
            return len(new_cells)

    @staticmethod
    def _merge_groups_in_cells(
        cells: Sequence[CellRecord], groups: Sequence[Sequence[int]]
    ) -> List[CellRecord]:
        result = list(cells)
        # Components are disjoint and refer to the current state. Process from
        # high to low so lower indices are unaffected by removals.
        normalized = [sorted({int(index) for index in group}) for group in groups]
        for unique in sorted(normalized, key=lambda group: group[0], reverse=True):
            if len(unique) < 2 or unique[0] < 0 or unique[-1] >= len(result):
                continue
            selected = [result[index] for index in unique]
            spatial = np.mean(np.stack([cell.spatial for cell in selected]), axis=0)
            maximum = float(np.max(spatial)) if spatial.size else 0.0
            if maximum > 0:
                spatial = spatial / maximum
            temporal = np.max(np.stack([cell.temporal for cell in selected]), axis=0)
            merged = CellRecord(
                cell_id=f"merged:{uuid4().hex}",
                lineage=tuple(item for cell in selected for item in cell.lineage),
                spatial=spatial,
                temporal=temporal,
            )
            selected_set = set(unique)
            result = [
                merged if index == unique[0] else cell
                for index, cell in enumerate(result)
                if index == unique[0] or index not in selected_set
            ]
        return result

    def merge_groups(
        self,
        requests: Dict[SessionKey, Sequence[Sequence[int]]],
        loaders: Dict[SessionKey, Callable[[], Tuple[np.ndarray, np.ndarray]]],
        label: str = "Auto-merge cells",
    ) -> Dict[SessionKey, int]:
        with self._lock:
            changes: Dict[SessionKey, List[CellRecord]] = {}
            counts: Dict[SessionKey, int] = {}
            for key, groups in requests.items():
                state = self._state(key, loaders[key])
                cells = self._merge_groups_in_cells(state.cells, groups)
                counts[key] = len(state.cells) - len(cells)
                if len(cells) != len(state.cells):
                    changes[key] = cells
            if changes:
                self.replace_cells(changes, loaders, label)
            return counts

    def delete_session(
        self,
        key: SessionKey,
        loader: Callable[[], Tuple[np.ndarray, np.ndarray]],
    ) -> None:
        with self._lock:
            self._state(key, loader)
            was_deleted = key in self._deleted
            command = WorkspaceCommand(
                label="Delete session",
                before_deleted={key: was_deleted},
                after_deleted={key: True},
            )
            self._commit(command)

    def undo(self) -> Optional[str]:
        with self._lock:
            if not self._undo:
                return None
            command = self._undo.pop()
            self._apply_before(command)
            self._redo.append(command)
            self.revision += 1
            return command.label

    def redo(self) -> Optional[str]:
        with self._lock:
            if not self._redo:
                return None
            command = self._redo.pop()
            self._apply_after(command)
            self._undo.append(command)
            self.revision += 1
            return command.label

    def status(self) -> dict:
        with self._lock:
            dirty_states = [state for state in self._states.values() if state.is_dirty()]
            dirty_domains = sorted(
                {domain for command in self._undo for domain in command.domains()}
            )
            return {
                "dirty": bool(self._undo),
                "dirty_sessions": len(dirty_states) + len(self._deleted),
                "dirty_domains": dirty_domains,
                "undo_count": len(self._undo),
                "redo_count": len(self._redo),
                "revision": self.revision,
            }

    def save_payload(self) -> Tuple[Dict[SessionKey, Tuple[np.ndarray, np.ndarray]], set[SessionKey]]:
        with self._lock:
            changed = {
                key: state.arrays()
                for key, state in self._states.items()
                if state.is_dirty() and key not in self._deleted
            }
            return changed, set(self._deleted)

    def save_snapshot(self) -> WorkspaceSaveSnapshot:
        """Freeze all persistable domains under one workspace revision."""
        with self._lock:
            cells, deleted = self.save_payload()
            return WorkspaceSaveSnapshot(
                revision=self.revision,
                cells=cells,
                deleted=deleted,
                alignment=deepcopy(self.alignment_state),
                matching=deepcopy(self.matching_state),
            )

    def mark_saved(self, revision: Optional[int] = None) -> bool:
        with self._lock:
            if revision is not None and revision != self.revision:
                return False
            for state in self._states.values():
                state.saved_revision = state.revision
                state.saved_cells = list(state.cells)
            self._deleted.clear()
            self._undo.clear()
            self._redo.clear()
            return True
