"""
3D Click tracking for iterative refinement.

Tracks the history of prompts (points and bounding boxes) across turns
in the agent loop. Modeled after MedSAM-Agent's Clicker class but for
3D coordinates.

References:
    MedSAM-Agent utils/clicker.py -- Click, Clicker classes
"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Click3D:
    """A 3D prompt point with its label.

    Follows the MedSAM-Agent Click pattern, extended to 3D.

    Attributes:
        coords: (z, y, x) absolute voxel coordinates.
        is_positive: True for foreground, False for background.
        turn: Zero-based turn index when this click was added.
        indx: Zero-based global click index.
    """
    coords: Tuple[int, int, int]
    is_positive: bool
    turn: int = 0
    indx: Optional[int] = None


@dataclass
class Clicker3D:
    """Tracks 3D prompt history for iterative refinement.

    Manages the sequence of positive/negative 3D points and bounding
    boxes across turns. Provides the last-click and all-clicks access
    patterns needed by the SAM-Med3D segmentation engine.

    References:
        MedSAM-Agent Clicker -- same state management pattern.
    """
    output_dir: str = "./intermediate_results"
    dataset_name: str = "default"
    file_name: str = "unknown"

    positives: List[Click3D] = field(default_factory=list)
    negatives: List[Click3D] = field(default_factory=list)
    bboxes: List[Tuple[int, int, int, int, int, int]] = field(default_factory=list)
    max_points: int = 20

    # --- State ---
    _current_bbox: Optional[Tuple[int, int, int, int, int, int]] = None

    # --- Properties ---

    @property
    def all_clicks(self) -> List[Click3D]:
        """All clicks in insertion order, independent of their label."""
        return sorted(self.positives + self.negatives, key=lambda c: c.indx if c.indx is not None else -1)

    @property
    def num_positives(self) -> int:
        return len(self.positives)

    @property
    def num_negatives(self) -> int:
        return len(self.negatives)

    @property
    def has_bbox(self) -> bool:
        return self._current_bbox is not None or len(self.bboxes) > 0

    @property
    def current_bbox(self) -> Optional[Tuple[int, int, int, int, int, int]]:
        """The most recent bounding box, if any."""
        return self._current_bbox

    # --- Mutation ---

    def add_positive(self, coords: Tuple[int, int, int], turn: int = 0):
        """Add a positive (foreground) 3D point."""
        if self.total_clicks >= self.max_points:
            return
        click = Click3D(coords=coords, is_positive=True, turn=turn)
        click.indx = self.total_clicks
        self.positives.append(click)

    def add_negative(self, coords: Tuple[int, int, int], turn: int = 0):
        """Add a negative (background) 3D point."""
        if self.total_clicks >= self.max_points:
            return
        click = Click3D(coords=coords, is_positive=False, turn=turn)
        click.indx = self.total_clicks
        self.negatives.append(click)

    def set_bbox(self, bbox: Tuple[int, int, int, int, int, int]):
        """Set the current bounding box and record it in history."""
        self._current_bbox = bbox
        self.bboxes.append(bbox)

    def reset(self):
        """Clear all prompts for a new volume."""
        self.positives.clear()
        self.negatives.clear()
        self.bboxes.clear()
        self._current_bbox = None

    @property
    def total_clicks(self) -> int:
        return len(self.positives) + len(self.negatives)

    # --- Query ---

    def get_last_click(self) -> Optional[Click3D]:
        """Get the most recently added click (for iterative refinement)."""
        clicks = self.all_clicks
        return clicks[-1] if clicks else None

    def get_all_points(self) -> List[Tuple[int, int, int]]:
        """Get all 3D point coordinates in order."""
        return [c.coords for c in self.all_clicks]

    def get_all_labels(self) -> List[int]:
        """Get all point labels (1=positive, 0=negative) in order."""
        return [1 if c.is_positive else 0 for c in self.all_clicks]

    def get_state(self) -> dict:
        """Get serializable state for saving."""
        return {
            "positives": [c.coords for c in self.positives],
            "negatives": [c.coords for c in self.negatives],
            "bboxes": self.bboxes,
            "clicks": [{"coords": c.coords, "is_positive": c.is_positive, "turn": c.turn} for c in self.all_clicks],
        }

    def set_state(self, state: dict):
        """Restore state from a dict."""
        self.reset()
        if "clicks" in state:
            for click in state["clicks"]:
                add = self.add_positive if click["is_positive"] else self.add_negative
                add(tuple(click["coords"]), click.get("turn", 0))
        else:
            for coords in state.get("positives", []):
                self.add_positive(coords)
            for coords in state.get("negatives", []):
                self.add_negative(coords)
        for bbox in state.get("bboxes", []):
            self.set_bbox(bbox)

    def copy(self) -> "Clicker3D":
        return deepcopy(self)
