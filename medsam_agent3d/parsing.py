"""
Action parsing: extract structured actions from MLLM text output.

Handles JSON extraction from Qwen3-VL chat responses, including the
MedSAM-Agent-style <tool_call> XML wrapper format.

Coordinate denormalization: [0, 999] -> absolute voxel coordinates.

References:
    MedSAM-Agent process_response() -- same parsing pattern.
"""

import json
import math
import re
from typing import Any, Dict, Optional, Tuple


# Explicit stop action; never used as a parsing fallback.
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
    in_string = False
    escaped = False
    for i, ch in enumerate(text[start:], start):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _parse_function_xml(body):
    """Read Qwen native function tags; never evaluate parameter text as code."""
    body = re.sub(r'<parameter=(view|point_type)="(axial|coronal|sagittal|positive|negative)</parameter>',
                  lambda m: '<parameter=' + m[1] + '>' + m[2] + '</parameter>', body)
    names = re.findall(r"<function=([A-Za-z_][A-Za-z_0-9]*)>", body)
    if not names:
        raise ValueError("Missing function tag")
    # Observed model output sometimes repeats the same opening tag before args.
    opening = re.match(r"\s*(?:<function=[A-Za-z_][A-Za-z_0-9]*>\s*)+", body)
    if len(set(names)) != 1 or not opening or body.count('</function>') != 1:
        raise ValueError("Expected exactly one function call")
    content = body[opening.end():]
    if '<function=' in content or not content.rstrip().endswith('</function>'):
        raise ValueError("Ambiguous or incomplete function call")
    content = content.rstrip()[:-len('</function>')]
    arguments = {}
    pattern = re.compile(r'<parameter=([A-Za-z_][A-Za-z_0-9]*)([^>]*)>(.*?)</parameter>', re.DOTALL)
    end = 0
    for match in pattern.finditer(content):
        if content[end:match.start()].strip():
            raise ValueError("Unexpected text between parameters")
        key, suffix, value = match.groups()
        if key in arguments:
            raise ValueError("Duplicate parameter")
        if suffix:
            # Narrow recovery of the reported broken enum syntax:
            # <parameter=view="axial</parameter>. Coordinates never use this.
            if value.strip() or key not in {'view', 'point_type'} or not suffix.startswith('="'):
                raise ValueError("Malformed parameter tag")
            value = suffix[2:]
        value = value.strip()
        if key in {'view', 'point_type'}:
            if value.startswith('"') and value.endswith('"'):
                value = json.loads(value)
        else:
            value = json.loads(value)
        arguments[key] = value
        end = match.end()
    if content[end:].strip():
        raise ValueError("Incomplete or malformed parameter")
    return {'name': names[0], 'arguments': arguments}


def parse_action(text: str) -> Dict[str, Any]:
    """Parse MLLM output text into a structured action dictionary.

    Supports these output formats (in order of priority):
    1. MedSAM-Agent style: ``<tool_call>{"name": "...", "arguments": {...}}</tool_call>``
    2. Raw JSON: ``{"name": "...", "arguments": {...}}``
    3. Flat JSON (legacy): ``{"name": "add_3d_bbox", "bbox_3d": [...]}``

    Invalid output returns invalid_action with a diagnostic; only explicit stop calls stop.

    Args:
        text: Raw MLLM response text.

    Returns:
        Action dict with keys:
        - ``name``: one of "add_3d_bbox", "add_3d_point", "stop_action"
        - ``arguments``: tool-call arguments
        - ``_raw_output``: the original raw text
    """
    # Primary: MedSAM-Agent style <tool_call>{"name": ...}</tool_call>
    if not isinstance(text, str):
        return {"name": "invalid_action", "_error": "Response must be text", "_raw_output": repr(text)}
    tool_match = re.search(
        r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL
    )
    if len(re.findall(r'<tool_call>', text)) > 1:
        return {'name': 'invalid_action', '_error': 'Return exactly one tool call', '_raw_output': text}
    search_text = tool_match.group(1).strip() if tool_match else text
    if '<function=' in search_text:
        try:
            return validate_action(_parse_function_xml(search_text), raw_output=text)
        except (ValueError, TypeError) as exc:
            return {'name': 'invalid_action', '_error': f'Invalid function tool call: {exc}', '_raw_output': text}

    json_str = extract_balanced_json(search_text)
    if json_str:
        try:
            obj = json.loads(json_str)
            return validate_action(obj, raw_output=text)
        except json.JSONDecodeError:
            pass

    return {"name": "invalid_action", "_error": "No valid JSON or function tool call found", "_raw_output": text}


def validate_action(action: Any, raw_output: Optional[str] = None) -> Dict[str, Any]:
    """Validate tool arguments before any model state is changed."""
    raw = raw_output if raw_output is not None else (action.get("_raw_output", "") if isinstance(action, dict) else "")
    def invalid(message):
        return {"name": "invalid_action", "_error": str(message), "_raw_output": str(raw)}
    if not isinstance(action, dict):
        return invalid("Tool call must be an object")
    name = action.get("name")
    if not isinstance(name, str) or name not in {"add_3d_point", "add_3d_bbox", "stop_action", "navigate", "add_view_point"}:
        return invalid(action.get("_error", "Unknown tool name"))
    args = action.get("arguments", {})
    if not isinstance(args, dict):
        return invalid("arguments must be an object")
    if name == "stop_action":
        return {"name": name, "arguments": {}, "_raw_output": raw}
    key, count = (("point_2d", 2) if name == "add_view_point" else
                  ("bbox_3d", 6) if name == "add_3d_bbox" else ("point_3d", 3))
    coords = args.get(key, action.get(key))
    if not isinstance(coords, (list, tuple)) or len(coords) != count:
        return invalid(f"{key} must contain exactly {count} integers")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 999 or not math.isfinite(v) or int(v) != v for v in coords):
        return invalid("Coordinates must be integers in [0, 999]")
    normalized = {key: [int(v) for v in coords]}
    if name == "add_view_point":
        view = args.get("view")
        if not isinstance(view, str) or view not in {"axial", "coronal", "sagittal"}:
            return invalid("view must be axial, coronal, or sagittal")
        normalized["view"] = view
    if name in {"add_3d_point", "add_view_point"}:
        label = args.get("point_type", action.get("point_type"))
        if not isinstance(label, str) or label not in {"positive", "negative"}:
            return invalid("point_type must be positive or negative")
        normalized["point_type"] = label
    return {"name": name, "arguments": normalized, "_raw_output": raw}


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

    Sorts corners to ensure z1 Ã¢â€°Â¤ z2, y1 Ã¢â€°Â¤ y2, x1 Ã¢â€°Â¤ x2, and clamps to volume bounds.

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
    return action.get("name") in {"add_3d_point", "add_view_point"}
