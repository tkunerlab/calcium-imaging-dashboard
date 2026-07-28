"""Pure alignment state and session-reference rules."""

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class AlignmentTransform:
    """Accumulated transform parameters in the common FOV coordinate space."""

    dx: float = 0.0
    dy: float = 0.0
    rotation: float = 0.0
    scale: float = 1.0


@dataclass(frozen=True)
class AlignmentReference:
    active_index: int
    reference_index: int
    root_index: int
    strategy: str

    @property
    def is_root(self) -> bool:
        return self.active_index == self.root_index


def resolve_alignment_reference(
    session_order: Sequence[object],
    active_index: int,
    root_index: int,
    strategy: str,
) -> AlignmentReference:
    """Resolve Direct root or the next Sequential neighbor toward that root."""
    size = len(session_order)
    if size == 0:
        raise ValueError("At least one session is required.")
    if not 0 <= active_index < size or not 0 <= root_index < size:
        raise IndexError("Active and root indices must be in the session order.")
    normalized = strategy.strip().lower()
    if normalized == "direct":
        reference_index = root_index
    elif normalized == "sequential":
        if active_index < root_index:
            reference_index = active_index + 1
        elif active_index > root_index:
            reference_index = active_index - 1
        else:
            reference_index = root_index
    else:
        raise ValueError(f"Unknown alignment strategy: {strategy}")
    return AlignmentReference(
        active_index=active_index,
        reference_index=reference_index,
        root_index=root_index,
        strategy="Sequential" if normalized == "sequential" else "Direct",
    )
