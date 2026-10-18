"""
Shared data types for the MedSAM-Agent3D pipeline.

References:
    MedSAM-Agent: InferenceArgs pattern for config objects
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class InferenceArgs:
    """Configuration for a single agent run.

    Pattern borrowed from MedSAM-Agent's InferenceArgs -- cleanly separates
    config from orchestration logic.
    """
    # --- Required ---
    target_name: str = ""
    volume_id: str = ""
    output_dir: str = "outputs/inference"

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
    sam_device: str = "auto"

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
    iou: Optional[float]
    dice: Optional[float]
    mask_voxels: int
    center: Tuple[int, int, int]
    error: str = ""
    next_center: Optional[Tuple[int, int, int]] = None


@dataclass
class AgentResult:
    """Result of running the agent loop on one volume/target pair."""
    volume_id: str
    target_name: str
    turns: List[TurnRecord] = field(default_factory=list)
    final_iou: Optional[float] = None
    final_dice: Optional[float] = None
    success: bool = False
    error: str = ""
    predicted_mask: Optional[Any] = field(default=None, repr=False)
    mask_path: Optional[str] = None
    metrics_available: bool = False
    termination_reason: str = ""
    config: Dict[str, Any] = field(default_factory=dict)


# --- Coordinate types ---

# A 3D point in original voxel space: (z, y, x)
VoxelPoint = Tuple[int, int, int]

# A 3D bounding box in original voxel space: (z1, y1, x1, z2, y2, x2)
VoxelBBox = Tuple[int, int, int, int, int, int]

# Normalized coordinates [0, 999] -- MedSAM-Agent convention
NormCoord = int
NormPoint = Tuple[NormCoord, NormCoord, NormCoord]
NormBBox = Tuple[NormCoord, NormCoord, NormCoord, NormCoord, NormCoord, NormCoord]


def _binary_masks(pred, gt):
    import numpy as np
    pred, gt = np.asarray(pred), np.asarray(gt)
    if pred.ndim != 3 or pred.shape != gt.shape:
        raise ValueError("Metrics require matching 3D mask shapes")
    if not np.isfinite(pred).all() or not np.isfinite(gt).all():
        raise ValueError("Metrics require finite mask values")
    return pred.astype(bool), gt.astype(bool)


def compute_iou_3d(pred: "np.ndarray", gt: "np.ndarray") -> float:
    """Binary IoU; two empty masks score one."""
    import numpy as np
    pred, gt = _binary_masks(pred, gt)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(intersection / union) if union else 1.0


def compute_dice_3d(pred: "np.ndarray", gt: "np.ndarray") -> float:
    """Binary Dice; two empty masks score one."""
    import numpy as np
    pred, gt = _binary_masks(pred, gt)
    denominator = pred.sum() + gt.sum()
    return float(2 * np.logical_and(pred, gt).sum() / denominator) if denominator else 1.0


def compute_metrics(pred: "np.ndarray", gt: "np.ndarray") -> Tuple[float, float]:
    """Compute binary IoU and Dice on matching 3D grids."""
    return compute_iou_3d(pred, gt), compute_dice_3d(pred, gt)
