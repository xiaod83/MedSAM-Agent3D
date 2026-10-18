"""
System prompt builder for the MedSAM-Agent3D pipeline.

Constructs the tool-calling system prompt with normalized [0, 999]
coordinate conventions, tool definitions, and tri-view geometry
instructions for the MLLM.

References:
    MedSAM-Agent unibiomed_inference_toolusing.py -- build_prompt pattern
    Qwen3-VL chat template with <tool_call> XML tags
"""

import json
from typing import List, Optional, Tuple


# Tool function definitions (JSON schema for Qwen3-VL tool calling)
TOOL_DEFINITIONS = [
    {"type": "function", "function": {
        "name": "add_view_point",
        "description": "Click visible anatomy in one displayed panel; fixed slice coordinate is supplied automatically.",
        "parameters": {"type": "object", "properties": {
            "view": {"type": "string", "enum": ["axial", "coronal", "sagittal"]},
            "point_2d": {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 999}, "minItems": 2, "maxItems": 2},
            "point_type": {"type": "string", "enum": ["positive", "negative"]}},
            "required": ["view", "point_2d", "point_type"]}}},

    {"type": "function", "function": {
        "name": "navigate", "description": "Inspect slices through a different 3D intersection without changing segmentation.",
        "parameters": {"type": "object", "properties": {"point_3d": {
            "type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 999},
            "minItems": 3, "maxItems": 3}}, "required": ["point_3d"]}}},
    {
        "type": "function",
        "function": {
            "name": "add_3d_bbox",
            "description": (
                "Add a 3D bounding box to initialize the segmentation of a "
                "target organ. This uses a positive center point if the checkpoint has no trained box embeddings. Define the rough "
                "location and extent of the target."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "bbox_3d": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 6,
                        "maxItems": 6,
                        "description": (
                            "3D bounding box [z1, y1, x1, z2, y2, x2] in range "
                            "[0, 999]. (z1,y1,x1) is one corner, (z2,y2,x2) "
                            "is the opposite corner."
                        ),
                    }
                },
                "required": ["bbox_3d"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_3d_point",
            "description": (
                "Add a 3D point to initialize or refine the 3D segmentation. "
                "Use positive points to include tissue, negative points to "
                "exclude tissue."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "point_3d": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 3,
                        "maxItems": 3,
                        "description": (
                            "3D coordinate [z, y, x] in range [0, 999]. "
                            "These are normalized coordinates where 0 is the "
                            "start and 999 is the end of the volume in each "
                            "dimension."
                        ),
                    },
                    "point_type": {
                        "type": "string",
                        "enum": ["positive", "negative"],
                        "description": (
                            "Type of point: 'positive' to expand mask, "
                            "'negative' to refine/exclude tissue"
                        ),
                    },
                },
                "required": ["point_3d", "point_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop_action",
            "description": (
                "Stop the refinement process when the mask accurately covers "
                "the target organ."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


def build_tools_xml() -> str:
    """Build the <tools> XML block for the system prompt."""
    return "\n".join([json.dumps(t) for t in TOOL_DEFINITIONS])


def build_system_prompt(volume_shape: Tuple[int, int, int],
                        organ_names: Optional[List[str]] = None) -> str:
    """Describe visible-panel grounding without suggesting a target coordinate."""
    dz, dy, dx = volume_shape
    targets = f" Requested structures: {', '.join(organ_names)}." if organ_names else ""
    return f"""You annotate a 3D medical volume using three full-slice panels, left to right:
Axial | Coronal | Sagittal. Volume dimensions Z={dz}, Y={dy}, X={dx}.{targets}
Green overlay is the current prediction; it may cover the wrong tissue.

Choose a point INSIDE the requested anatomical structure visible in a panel using
add_view_point. Its point_2d is [u,v], relative to that panel's ANATOMY rectangle:
u increases left to right, v top to bottom; endpoints are 0 and 999.
Exclude headers, tick labels, margins and the other panels when estimating u,v.
Axial holds Z fixed, u=X,v=Y. Coronal holds Y fixed, u=X,v=Z.
Sagittal holds X fixed, u=Y,v=Z. The tool supplies the fixed slice coordinate.
Use the patient direction labels to distinguish anatomical left/right; these are
patient directions, not the viewer's left/right. If labels are absent, orientation is unknown.
Margin focus ticks indicate the displayed slices, not a suggested target point.
Do not choose a point just because it is at the focus or near an earlier click.
If the structure is not visible, navigate to inspect other slices rather than guessing.
Initialize with a positive point in target tissue. Later add positive points to missing
target tissue or negative points to incorrect green tissue. Repeated clicks on incorrect
foreground will not locate a different organ. Stop only when the nonempty overlay
visibly covers the requested structure. Never claim accuracy from a nonempty mask alone.

Legacy add_3d_point and navigate coordinates are normalized [z,y,x], each 0..999.
Boxes use [z1,y1,x1,z2,y2,x2]; stock SAM approximates boxes with their center point.
Return exactly one JSON tool call inside <tool_call>...</tool_call>, using the
function name and arguments from these schemas:
<tools>
{build_tools_xml()}
</tools>"""
