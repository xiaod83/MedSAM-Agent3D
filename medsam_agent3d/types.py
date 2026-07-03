"""
Shared data types for the MedSAM-Agent3D pipeline.

References:
    MedSAM-Agent: InferenceArgs pattern for config objects
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class InferenceArgs:
    """Configuration for a single agent run.

    Pattern borrowed from MedSAM-Agent's InferenceArgs -- cleanly separates
    config from orchestration logic.
    """
    # --- Required ---
    target_name: str = ""
    volume_id: str = ""
    output_dir: str = "eval_outputs"

    # --- Agent loop ---
    max_turns: int = 5
    temperature: float = 0.1
    seed: int = 42

    # --- MLLM ---
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    load_in_4bit: bool = False
    device_map: str = "auto"

    # --- SAM-Med3D ---
    sam_checkpoint: Optional[str] = None

    # --- Visualization ---
    save_trajectories: bool = False
    compute_per_turn_metrics: bool = True


@dataclass
class TurnRecord:
    """Metrics and state for a single turn in the agent loop."""
    turn: int
    image_path: str
    action: dict
    raw_output: str
    iou: float
    dice: float
    mask_voxels: int
    center: Tuple[int, int, int]


@dataclass
class AgentResult:
    """Result of running the agent loop on one volume/target pair."""
    volume_id: str
    target_name: str
    turns: List[TurnRecord] = field(default_factory=list)
    final_iou: float = 0.0
    final_dice: float = 0.0
    success: bool = False
    error: str = ""


# --- Coordinate types ---

# A 3D point in original voxel space: (z, y, x)
VoxelPoint = Tuple[int, int, int]

# A 3D bounding box in original voxel space: (z1, y1, x1, z2, y2, x2)
VoxelBBox = Tuple[int, int, int, int, int, int]

# Normalized coordinates [0, 999] -- MedSAM-Agent convention
NormCoord = int
NormPoint = Tuple[NormCoord, NormCoord, NormCoord]
NormBBox = Tuple[NormCoord, NormCoord, NormCoord, NormCoord, NormCoord, NormCoord]


def compute_iou_3d(pred: "np.ndarray", gt: "np.ndarray") -> float:
    """3D Intersection-over-Union."""
    import numpy as np
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter / union) if union > 0 else 0.0


def compute_dice_3d(pred: "np.ndarray", gt: "np.ndarray") -> float:
    """3D Dice coefficient."""
    import numpy as np
    inter = np.logical_and(pred, gt).sum()
    return float(2 * inter / (pred.sum() + gt.sum() + 1e-8))


def compute_metrics(pred: "np.ndarray", gt: "np.ndarray") -> Tuple[float, float]:
    """Compute (IoU, Dice) for 3D binary masks."""
    return compute_iou_3d(pred, gt), compute_dice_3d(pred, gt)
