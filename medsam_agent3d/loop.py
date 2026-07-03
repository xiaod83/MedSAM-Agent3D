"""
Agentic loop orchestration: MLLM -> SAM-Med3D -> render -> repeat.

This is the core of MedSAM-Agent3D. It runs the iterative refinement
loop where the MLLM reasons about tri-view slices and calls SAM-Med3D
with 3D prompts (bounding boxes or points), refining the segmentation
until the MLLM issues a ``stop_action`` or max turns are reached.

References:
    MedSAM-Agent forward_single_image() -- same loop pattern.
    Original agent_inference.py run_agent_loop().
"""

import os
from typing import Optional, Tuple

import numpy as np
import torch

from .types import AgentResult, InferenceArgs, TurnRecord, compute_metrics
from .rendering import render_triview
from .parsing import (
    denormalize_bbox,
    denormalize_point,
    is_bbox_action,
    is_point_action,
    is_stop_action,
)
from .prompting import build_system_prompt
from .clicker import Clicker3D
from .agent import MLLMAgent


def run_agent_loop(
    mllm_agent: MLLMAgent,
    sam_model: "FrozenSAMMed3D",
    volume: np.ndarray,
    gt_mask: np.ndarray,
    target_name: str,
    volume_id: str,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    max_turns: int = 5,
    output_dir: str = "eval_outputs",
    temperature: float = 0.1,
    seed: int = 42,
) -> AgentResult:
    """Run the full agent loop: MLLM -> SAM-Med3D -> render -> repeat.

    Args:
        mllm_agent: Vision-language agent (e.g., QwenVLAgent).
        sam_model: Frozen SAM-Med3D instance with ``set_image`` already called.
        volume: 3D volume array (z, y, x).
        gt_mask: Ground truth binary mask (z, y, x) -- used for metrics only.
        target_name: Organ/structure name (e.g., "liver").
        volume_id: Case identifier for output naming.
        spacing: Voxel spacing (dz, dy, dx).
        max_turns: Maximum interaction turns.
        output_dir: Directory to save tri-view images.
        temperature: Sampling temperature for MLLM.
        seed: Random seed for reproducibility.

    Returns:
        AgentResult with turn-by-turn details and final metrics.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs(output_dir, exist_ok=True)
    result = AgentResult(volume_id=volume_id, target_name=target_name)

    if gt_mask.sum() == 0:
        result.error = "Empty ground truth mask"
        return result

    # --- Setup ---
    dz, dy, dx = volume.shape
    system_prompt = build_system_prompt(volume.shape)
    instruction = f"Segment the {target_name}"

    # Initialize SAM-Med3D with this volume
    print(f"Setting up SAM-Med3D for {volume_id}/{target_name}...")
    sam_model.set_image(volume)

    # State
    center: Tuple[int, int, int] = (dz // 2, dy // 2, dx // 2)
    pred_mask = np.zeros(volume.shape, dtype=np.uint8)
    low_res_logits = None  # No prior logits on first call
    has_initialized = False  # SAM has produced at least one mask?
    clicker = Clicker3D()

    conversation_history = []

    prev_iou = 0.0
    MAX_RETRIES = 1
    retry_count = 0

    # ===================================================================
    # Main loop
    # ===================================================================
    for turn in range(max_turns):
        # --- Render current state ---
        img_name = f"{volume_id}_{target_name}_t{turn}.png"
        img_path = os.path.join(output_dir, img_name)
        triview = render_triview(
            volume, center, spacing,
            mask=pred_mask if pred_mask.sum() > 0 else None,
        )
        triview.save(img_path)
        print(f"  Turn {turn}: tri-view saved to {img_path}")

        # --- MLLM generates action ---
        action = mllm_agent.generate_action(
            image=triview,
            instruction=instruction,
            system_prompt=system_prompt,
            conversation_history=conversation_history,
            temperature=temperature,
        )
        raw_out = action.get("_raw_output", "")
        action_name = action.get("name", "unknown")
        print(f"  Turn {turn}: MLLM -> {action_name}")
        print(f"    Raw output: {raw_out[:300]}")

        # --- Handle stop_action ---
        if is_stop_action(action):
            current_iou, current_dice = compute_metrics(pred_mask, gt_mask)
            rec = TurnRecord(
                turn=turn, image_path=img_path,
                action={"name": "stop_action"}, raw_output=raw_out,
                iou=current_iou, dice=current_dice,
                mask_voxels=int(pred_mask.sum()), center=center,
            )
            result.turns.append(rec)
            result.final_iou = current_iou
            result.final_dice = current_dice
            result.success = True
            print(f"  -> Agent stopped at turn {turn}. "
                  f"Final IoU={current_iou:.4f}, Dice={current_dice:.4f}")
            break

        # --- Parse arguments (supports both nested and flat JSON) ---
        args = action.get("arguments", {})

        # ========== BBOX ACTION ==========
        if is_bbox_action(action):
            bbox = args.get("bbox_3d") or action.get("bbox_3d")
            if bbox is None or len(bbox) != 6:
                print(f"  -> Invalid bbox_3d: {bbox}, retrying...")
                retry_count += 1
                if retry_count > MAX_RETRIES:
                    print("  -> Max retries reached, stopping")
                    break
                continue

            bbox_voxel = denormalize_bbox(tuple(bbox), (dz, dy, dx))
            print(f"  Turn {turn}: bbox={bbox_voxel}")

            # SAM-Med3D inference: bbox + no prior logits (first initialization)
            pred_mask, low_res_logits = sam_model.predict_mask(
                points=None, labels=None,
                box_3d=bbox_voxel,
                mask_logits=None,  # no prior -> no_mask_embed
            )
            has_initialized = True
            clicker.set_bbox(bbox_voxel)

            current_iou, current_dice = compute_metrics(pred_mask, gt_mask)
            delta_iou = current_iou - prev_iou
            prev_iou = current_iou
            print(f"    IoU={current_iou:.4f} (d={delta_iou:+.4f}), "
                  f"Dice={current_dice:.4f}")

            # Update focus center
            center = _update_center(pred_mask, bbox_voxel)

            rec = TurnRecord(
                turn=turn, image_path=img_path,
                action={"name": "add_3d_bbox",
                        "arguments": {"bbox_3d": list(bbox_voxel)}},
                raw_output=raw_out,
                iou=current_iou, dice=current_dice,
                mask_voxels=int(pred_mask.sum()), center=center,
            )
            result.turns.append(rec)

        # ========== POINT ACTION ==========
        elif is_point_action(action):
            pt = args.get("point_3d") or action.get("point_3d")
            pt_type = args.get("point_type") or action.get("point_type", "positive")

            if pt is None or len(pt) < 3:
                print(f"  -> No valid point_3d in action: {action}")
                retry_count += 1
                if retry_count > MAX_RETRIES:
                    print("  -> Max retries reached, stopping")
                    break
                continue

            voxel_pt = denormalize_point(tuple(pt), (dz, dy, dx))
            z, y, x = voxel_pt
            z = max(0, min(z, dz - 1))
            y = max(0, min(y, dy - 1))
            x = max(0, min(x, dx - 1))
            voxel_pt = (z, y, x)
            label = 1 if pt_type == "positive" else 0

            # Track click
            if label == 1:
                clicker.add_positive(voxel_pt, turn)
            else:
                clicker.add_negative(voxel_pt, turn)

            # MedSAM-Agent refinement pattern:
            # - First call (no prior mask): pass all points, mask_logits=None
            # - Subsequent calls: pass only latest point + prior logits
            if not has_initialized:
                pred_mask, low_res_logits = sam_model.predict_mask(
                    points=[voxel_pt], labels=[label],
                    box_3d=None,
                    mask_logits=None,
                )
                has_initialized = True
            else:
                pred_mask, low_res_logits = sam_model.predict_mask(
                    points=[voxel_pt], labels=[label],
                    box_3d=None,
                    mask_logits=low_res_logits,
                )

            current_iou, current_dice = compute_metrics(pred_mask, gt_mask)
            delta_iou = current_iou - prev_iou
            prev_iou = current_iou
            print(f"  Turn {turn}: point={voxel_pt} ({pt_type}), "
                  f"IoU={current_iou:.4f} (d={delta_iou:+.4f}), "
                  f"Dice={current_dice:.4f}")

            center = _update_center(pred_mask, voxel_pt)

            rec = TurnRecord(
                turn=turn, image_path=img_path,
                action={
                    "name": "add_3d_point",
                    "arguments": {
                        "point_3d": list(voxel_pt),
                        "point_type": pt_type,
                    },
                },
                raw_output=raw_out,
                iou=current_iou, dice=current_dice,
                mask_voxels=int(pred_mask.sum()), center=center,
            )
            result.turns.append(rec)

        # ========== UNKNOWN ACTION ==========
        else:
            print(f"  -> Unknown action '{action_name}', "
                  f"raw: {raw_out[:200]}")
            retry_count += 1
            if retry_count > MAX_RETRIES:
                print(f"  -> Max retries ({MAX_RETRIES}) reached, ending loop")
                break
            continue

        # Reset retry count on successful action
        retry_count = 0

        # --- Append to conversation history (text-only to avoid VRAM blowup) ---
        conversation_history.append({
            "role": "user",
            "content": (
                f"[Tri-view image -- mask shown in green overlay] "
                f"{instruction}"
            ),
        })
        conversation_history.append({
            "role": "assistant",
            "content": raw_out,
        })

    else:
        # Max turns reached without stop_action
        result.final_iou, result.final_dice = compute_metrics(pred_mask, gt_mask)
        result.success = True
        print(f"  -> Max turns ({max_turns}) reached. "
              f"Final IoU={result.final_iou:.4f}")

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _update_center(
    mask: np.ndarray,
    fallback: Tuple[int, int, int],
) -> Tuple[int, int, int]:
    """Compute the new focus center: centroid of the mask if non-empty,
    otherwise the fallback position."""
    if mask.sum() > 0:
        from scipy.ndimage import center_of_mass
        return tuple(int(c) for c in center_of_mass(mask))
    return fallback


def run_with_args(
    mllm_agent: MLLMAgent,
    sam_model: "FrozenSAMMed3D",
    volume: np.ndarray,
    gt_mask: np.ndarray,
    spacing: Tuple[float, float, float],
    args: InferenceArgs,
) -> AgentResult:
    """Convenience wrapper: run the agent loop using an InferenceArgs object.

    Args:
        mllm_agent: Vision-language agent.
        sam_model: Frozen SAM-Med3D.
        volume: 3D volume array.
        gt_mask: Ground truth mask.
        spacing: Voxel spacing.
        args: InferenceArgs configuration.

    Returns:
        AgentResult.
    """
    return run_agent_loop(
        mllm_agent=mllm_agent,
        sam_model=sam_model,
        volume=volume,
        gt_mask=gt_mask,
        target_name=args.target_name,
        volume_id=args.volume_id,
        spacing=spacing,
        max_turns=args.max_turns,
        output_dir=args.output_dir,
        temperature=args.temperature,
        seed=args.seed,
    )
