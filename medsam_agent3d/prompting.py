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
    {
        "type": "function",
        "function": {
            "name": "add_3d_bbox",
            "description": (
                "Add a 3D bounding box to initialize the segmentation of a "
                "target organ. Use this on the first turn to define the rough "
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


def _format_examples() -> str:
    """Format the few-shot examples that teach the MLLM the expected format."""
    return """
Examples:
- To place a positive point at the center:
  <tool_call>
  {"name": "add_3d_point", "arguments": {"point_3d": [500, 500, 500], "point_type": "positive"}}
  </tool_call>
- To draw a bounding box around the liver:
  <tool_call>
  {"name": "add_3d_bbox", "arguments": {"bbox_3d": [200, 300, 250, 700, 650, 750]}}
  </tool_call>
- To stop when done:
  <tool_call>
  {"name": "stop_action"}
  </tool_call>"""


def build_system_prompt(
    volume_shape: Tuple[int, int, int],
    organ_names: Optional[List[str]] = None,
) -> str:
    """Build the full system prompt with tool definitions and instructions.

    Uses normalized [0, 999] coordinate range (following MedSAM-Agent
    convention) so the MLLM's output distribution is volume-size invariant.
    Coordinates are denormalized to voxel space inside the agent loop.

    Args:
        volume_shape: (dz, dy, dx) voxel dimensions of the 3D volume.
        organ_names: Optional list of organ/structure names to segment.

    Returns:
        Full system prompt string ready for the MLLM's system role.
    """
    dz, dy, dx = volume_shape
    tools_str = build_tools_xml()
    examples_str = _format_examples()

    organ_str = ""
    if organ_names:
        organ_str = (
            f" You will be asked to segment specific organs: "
            f"{', '.join(organ_names)}."
        )

    return f"""You are a professional 3D medical segmentation annotator. You analyze a 3D CT/MRI volume through three orthogonal 2D slice views stitched side-by-side: [Axial | Coronal | Sagittal].

The 3D volume dimensions are: Z={dz}, Y={dy}, X={dx} (voxel coordinates).{organ_str}

The current segmentation mask is shown as a semi-transparent green overlay on each view. Your goal is to ensure the mask exactly covers the entire target organ and excludes all non-target tissue.

IMPORTANT COORDINATE RULES:
- All coordinates use a normalized range of [0, 999] in each dimension.
- 0 represents the start of the volume, 999 represents the end.
- So the center of the volume in every dimension is approximately 500.
- For the bbox, [z1, y1, x1, z2, y2, x2] specifies opposite corners.

# Tools

You must call one function to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools_str}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>
{examples_str}

Only use the provided functions to complete your task. Do not invent or assume any other functions. Carefully consider the current mask state and the tri-view slices before each action.

Remember: the three views are [Axial | Coronal | Sagittal] from left to right. The center of each view corresponds to the focus point in 3D. Use this spatial understanding to place accurate 3D coordinates."""
