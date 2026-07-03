"""
Action parsing: extract structured actions from MLLM text output.

Handles JSON extraction from Qwen3-VL chat responses, including the
MedSAM-Agent-style <tool_call> XML wrapper format.

Coordinate denormalization: [0, 999] -> absolute voxel coordinates.

References:
    MedSAM-Agent process_response() -- same parsing pattern.
"""

import json
import re
from typing import Any, Dict, Optional, Tuple


# Sentinel value: a well-formed but empty action (no valid tool call found)
STOP_ACTION: Dict[str, Any] = {"name": "stop_action"}


def extract_balanced_json(text: str) -> Optional[str]:
    """Extract the first balanced { ... } JSON object from a text string.

    Args:
        text: Raw MLLM output, possibly containing non-JSON text.

    Returns:
        The JSON string if found, else None.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_action(text: str) -> Dict[str, Any]:
    """Parse MLLM output text into a structured action dictionary.

    Supports these output formats (in order of priority):
    1. MedSAM-Agent style: ``<tool_call>{"name": "...", "arguments": {...}}</tool_call>``
    2. Raw JSON: ``{"name": "...", "arguments": {...}}``
    3. Flat JSON (legacy): ``{"name": "add_3d_bbox", "bbox_3d": [...]}``

    If no valid action can be extracted, returns ``{"name": "stop_action"}``.

    Args:
        text: Raw MLLM response text.

    Returns:
        Action dict with keys:
        - ``name``: one of "add_3d_bbox", "add_3d_point", "stop_action"
        - ``arguments``: tool-call arguments
        - ``_raw_output``: the original raw text
    """
    # Primary: MedSAM-Agent style <tool_call>{"name": ...}</tool_call>
    tool_match = re.search(
        r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL
    )
    search_text = tool_match.group(1).strip() if tool_match else text

    json_str = extract_balanced_json(search_text)
    if json_str:
        try:
            obj = json.loads(json_str)
            if "name" in obj:
                obj["_raw_output"] = text
                return obj
        except json.JSONDecodeError:
            pass

    return STOP_ACTION | {"_raw_output": text}


def denormalize_coord(norm_val: int, dim_size: int) -> int:
    """Convert a [0, 999] normalized coordinate to absolute voxel index.

    Following the MedSAM-Agent convention: normalized coordinates make
    the MLLM's output distribution volume-size-invariant. This function
    maps them back into the actual volume's voxel space.

    Args:
        norm_val: Coordinate in [0, 999].
        dim_size: Size of the volume dimension (voxels).

    Returns:
        Absolute voxel index in [0, dim_size - 1].
    """
    return round(norm_val / 999.0 * (dim_size - 1))


def denormalize_point(
    point_3d: Tuple[int, int, int],
    shape: Tuple[int, int, int],
) -> Tuple[int, int, int]:
    """Denormalize a 3D point from [0, 999] to absolute voxel coordinates.

    Args:
        point_3d: Normalized (z, y, x) in [0, 999].
        shape: Volume shape (dz, dy, dx).

    Returns:
        Absolute (z, y, x) in voxel space.
    """
    z = denormalize_coord(point_3d[0], shape[0])
    y = denormalize_coord(point_3d[1], shape[1])
    x = denormalize_coord(point_3d[2], shape[2])
    return (z, y, x)


def denormalize_bbox(
    bbox_3d: Tuple[int, int, int, int, int, int],
    shape: Tuple[int, int, int],
) -> Tuple[int, int, int, int, int, int]:
    """Denormalize a 3D bounding box from [0, 999] to absolute voxel coordinates.

    Sorts corners to ensure z1 ≤ z2, y1 ≤ y2, x1 ≤ x2, and clamps to volume bounds.

    Args:
        bbox_3d: Normalized (z1, y1, x1, z2, y2, x2) in [0, 999].
        shape: Volume shape (dz, dy, dx).

    Returns:
        Absolute (z1, y1, x1, z2, y2, x2) in voxel space, sorted and clamped.
    """
    dz, dy, dx = shape
    z1 = denormalize_coord(bbox_3d[0], dz)
    y1 = denormalize_coord(bbox_3d[1], dy)
    x1 = denormalize_coord(bbox_3d[2], dx)
    z2 = denormalize_coord(bbox_3d[3], dz)
    y2 = denormalize_coord(bbox_3d[4], dy)
    x2 = denormalize_coord(bbox_3d[5], dx)

    z1, z2 = sorted([max(0, min(z1, dz - 1)), max(0, min(z2, dz - 1))])
    y1, y2 = sorted([max(0, min(y1, dy - 1)), max(0, min(y2, dy - 1))])
    x1, x2 = sorted([max(0, min(x1, dx - 1)), max(0, min(x2, dx - 1))])

    return (z1, y1, x1, z2, y2, x2)


def is_stop_action(action: Dict[str, Any]) -> bool:
    """Check if an action is a stop signal."""
    return action.get("name") == "stop_action"


def is_bbox_action(action: Dict[str, Any]) -> bool:
    """Check if an action is a bounding box prompt."""
    return action.get("name") == "add_3d_bbox"


def is_point_action(action: Dict[str, Any]) -> bool:
    """Check if an action is a point prompt."""
    return action.get("name") == "add_3d_point"
