"""
medsam_agent3d: MLLM-driven agentic loop for 3D medical segmentation.

Modules:
    types       -- Shared dataclasses (TurnRecord, AgentResult, InferenceArgs)
    rendering   -- Tri-view visualization (3D volume -> 2D image)
    parsing     -- Action parsing (MLLM text -> structured JSON)
    prompting   -- System prompt builder with tool definitions
    clicker     -- 3D click history tracking
    agent       -- MLLM wrapper (Qwen3-VL)
    loop        -- Agentic loop orchestration
    evaluation  -- Batch and single-case evaluation runners
    cli         -- CLI entry points
"""

from .types import TurnRecord, AgentResult, InferenceArgs
from .rendering import render_triview
from .parsing import parse_action, denormalize_coord
from .prompting import build_system_prompt
from .clicker import Click3D, Clicker3D
from .agent import MLLMAgent, QwenVLAgent
from .loop import run_agent_loop
from .evaluation import evaluate_dataset, evaluate_single

__all__ = [
    "TurnRecord", "AgentResult", "InferenceArgs",
    "render_triview",
    "parse_action", "denormalize_coord",
    "build_system_prompt",
    "Click3D", "Clicker3D",
    "MLLMAgent", "QwenVLAgent",
    "run_agent_loop",
    "evaluate_dataset", "evaluate_single",
]
