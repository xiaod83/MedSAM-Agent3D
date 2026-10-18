"""Agent orchestration with validated actions and explicit termination state."""
import inspect
import json
import os
import random
from typing import Optional, Tuple

import numpy as np
import torch

from .types import AgentResult, InferenceArgs, TurnRecord, compute_metrics
from .rendering import render_triview
from .parsing import denormalize_bbox, denormalize_point, validate_action
from .prompting import build_system_prompt
from .agent import MLLMAgent


def run_agent_loop(
    mllm_agent: MLLMAgent,
    sam_model,
    volume: np.ndarray,
    gt_mask: Optional[np.ndarray],
    target_name: str,
    volume_id: str,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    max_turns: int = 5,
    output_dir: str = "outputs/inference",
    temperature: float = 0.1,
    seed: int = 42,
    affine=None,
) -> AgentResult:
    """Run inference; ground truth is optional and only used for reporting.

    A successful run means at least one nonempty prediction and a normal exit,
    not that segmentation quality passed a threshold. Invalid outputs receive
    corrective feedback and one retry without consuming a valid action turn.
    """
    if max_turns < 1:
        raise ValueError("max_turns must be positive")
    if not isinstance(volume, np.ndarray) or volume.ndim != 3 or not np.isfinite(volume).all():
        raise ValueError("volume must be a finite 3D array")
    if gt_mask is not None and gt_mask.shape != volume.shape:
        raise ValueError("Ground truth and volume shapes must match")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    result = AgentResult(volume_id=volume_id, target_name=target_name)
    result.metrics_available = gt_mask is not None
    result.config = {"model_id": getattr(mllm_agent, "model_id", type(mllm_agent).__name__),
                     "sam_checkpoint": getattr(sam_model, "checkpoint_path", None), "seed": seed, "max_turns": max_turns, "temperature": temperature,
                     "shape_zyx": list(volume.shape), "spacing_zyx": list(spacing),
                     "coordinate_system": "normalized_0_999_zyx",
                     "affine_xyz": None if affine is None else np.asarray(affine).tolist()}
    pred_mask = np.zeros(volume.shape, dtype=np.uint8)
    low_res_logits = None
    initialized = False
    center = tuple(s // 2 for s in volume.shape)
    history = []
    system_prompt = build_system_prompt(volume.shape)
    instruction = f"Segment the {target_name}"
    failures = 0
    actions = 0
    attempts = 0
    reason = "max_turns"

    def metrics():
        return compute_metrics(pred_mask, gt_mask) if gt_mask is not None else (None, None)

    try:
        # Keep lightweight third-party agents implementing set_image(volume) usable.
        params = inspect.signature(sam_model.set_image).parameters
        kwargs = {}
        if "spacing" in params:
            kwargs["spacing"] = spacing
        if "affine" in params:
            kwargs["affine"] = affine
        sam_model.set_image(volume, **kwargs)
        while actions < max_turns:
            attempt = attempts
            attempts += 1
            render_center = center
            image_path = os.path.join(output_dir, f"turn_{attempt:03d}.png")
            triview = render_triview(volume, center, spacing, mask=pred_mask)
            triview.save(image_path)
            focus = tuple(round(c / max(s - 1, 1) * 999) for c, s in zip(center, volume.shape))
            request = (f"{instruction}. Current slice intersection (z,y,x), normalized: {focus}. "
                       "Cyan crosshairs mark this intersection; panel axes increase right/down. "
                       "Initialize with a positive point inside visible target tissue. "
                       "Use navigate to inspect a different slice intersection if needed.")
            raw_action = mllm_agent.generate_action(
                image=triview, instruction=request, system_prompt=system_prompt,
                conversation_history=history, temperature=temperature,
            )
            action = validate_action(raw_action)
            raw = action.get("_raw_output", "")
            if not raw:
                raw = json.dumps({k: v for k, v in action.items() if not k.startswith("_")})
            name = action["name"]
            failure = action.get("_error", "")
            if name == "stop_action" and not initialized:
                failure = "No nonempty segmentation exists yet; initialize with a positive point."
            if name == "add_3d_point" and not initialized and action["arguments"]["point_type"] == "negative":
                failure = "Initialize with a positive foreground point before negative refinement."
            iou, dice = metrics()
            if failure:
                result.turns.append(TurnRecord(
                    turn=attempt, image_path=image_path, action=action, raw_output=raw,
                    iou=iou, dice=dice, mask_voxels=int(pred_mask.sum()), center=center,
                    error=failure,
                ))
                history.extend([{"role": "user", "content": request},
                                {"role": "assistant", "content": raw},
                                {"role": "user", "content": f"Tool error: {failure} Return one corrected tool call."}])
                failures += 1
                if failures > 1:
                    result.error = failure
                    reason = "invalid_action"
                    break
                continue
            failures = 0
            if name == "stop_action":
                reason = "agent_stop"
            elif name == "navigate":
                center = denormalize_point(tuple(action["arguments"]["point_3d"]), volume.shape)
            else:
                args = action["arguments"]
                if name == "add_3d_bbox":
                    bbox = denormalize_bbox(tuple(args["bbox_3d"]), volume.shape)
                    fallback = tuple((bbox[i] + bbox[i + 3]) // 2 for i in range(3))
                    mask, logits = sam_model.predict_mask(points=None, labels=None, box_3d=bbox, mask_logits=None)
                else:
                    point = denormalize_point(tuple(args["point_3d"]), volume.shape)
                    fallback = point
                    mask, logits = sam_model.predict_mask(
                        points=[point], labels=[int(args["point_type"] == "positive")],
                        box_3d=None, mask_logits=low_res_logits,
                    )
                if np.asarray(mask).shape != volume.shape:
                    raise ValueError("SAM returned a mask with the wrong shape")
                pred_mask = (np.asarray(mask) > 0).astype(np.uint8)
                low_res_logits = logits
                initialized = bool(pred_mask.any())
                center = _update_center(pred_mask, fallback)
                iou, dice = metrics()
            result.turns.append(TurnRecord(
                turn=attempt, image_path=image_path,
                action={k: v for k, v in action.items() if not k.startswith("_")},
                raw_output=raw, iou=iou, dice=dice,
                mask_voxels=int(pred_mask.sum()), center=render_center, next_center=center,
            ))
            history.extend([{"role": "user", "content": request},
                            {"role": "assistant", "content": raw}])
            actions += 1
            if name == "stop_action":
                break
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        reason = "execution_error"
    if reason != "execution_error":
        try:
            render_triview(volume, center, spacing, mask=pred_mask).save(os.path.join(output_dir, "final.png"))
        except Exception as exc:
            result.error = f"Saving final view failed: {exc}"
            reason = "execution_error"
    result.predicted_mask = pred_mask.copy()
    result.final_iou, result.final_dice = metrics()
    result.termination_reason = reason
    result.success = initialized and not result.error
    if not initialized and not result.error:
        result.error = "No nonempty segmentation produced"
    result.config["attempts"] = attempts
    result.config["valid_actions"] = actions
    return result


def _update_center(mask, fallback):
    """Return a three-coordinate focus even for empty predictions."""
    if mask.any():
        from scipy.ndimage import center_of_mass
        return tuple(int(c) for c in center_of_mass(mask))
    if len(fallback) != 3:
        raise ValueError("Focus fallback must have three coordinates")
    return tuple(int(c) for c in fallback)


def run_with_args(mllm_agent, sam_model, volume, gt_mask, spacing, args: InferenceArgs):
    return run_agent_loop(
        mllm_agent, sam_model, volume, gt_mask, args.target_name, args.volume_id,
        spacing=spacing, max_turns=args.max_turns, output_dir=args.output_dir,
        temperature=args.temperature, seed=args.seed,
    )
