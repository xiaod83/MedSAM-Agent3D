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

from importlib import import_module

# Keep navigation/help usable before the scientific dependencies are installed.
_EXPORTS = {
    "TurnRecord": "types", "AgentResult": "types", "InferenceArgs": "types",
    "render_triview": "rendering", "parse_action": "parsing", "denormalize_coord": "parsing",
    "build_system_prompt": "prompting", "Click3D": "clicker", "Clicker3D": "clicker",
    "MLLMAgent": "agent", "QwenVLAgent": "agent", "run_agent_loop": "loop",
    "evaluate_dataset": "evaluation", "evaluate_single": "evaluation",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value
