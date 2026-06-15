"""
MedSAM-Agent3D — Agentic Inference (Non-Trained Qwen3-VL)
==========================================================
Runs the interactive segmentation agent loop using a vanilla (non-fine-tuned)
Qwen3-VL model paired with a frozen SAM-Med3D engine.

Usage:
    python agent_inference.py \
        --volume path/to/ct.nii.gz \
        --mask path/to/liver.nii.gz \
        --target liver \
        --model Qwen/Qwen3-VL-8B-Instruct \
        --max-turns 5

Outputs:
    - Console log of each turn (action, IoU, Dice)
    - Final metrics summary
    - Tri-view images saved to --output-dir
"""

import os
import sys
import json
import re
import argparse
import time
from io import BytesIO
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info

# ---------------------------------------------------------------------------
# Helpers from existing codebase
# ---------------------------------------------------------------------------

def _normalize_slice(slice_2d):
    if slice_2d.max() == slice_2d.min():
        return np.zeros_like(slice_2d, dtype=np.uint8)
    norm = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min())
    return (norm * 255).astype(np.uint8)


def _resample_slice(slice_2d, spacing_2d, target_spacing=(1.0, 1.0)):
    import cv2
    scale_y = spacing_2d[0] / target_spacing[0]
    scale_x = spacing_2d[1] / target_spacing[1]
    if scale_y == 1.0 and scale_x == 1.0:
        return slice_2d
    new_shape = (int(slice_2d.shape[0] * scale_y), int(slice_2d.shape[1] * scale_x))
    if new_shape[0] == 0 or new_shape[1] == 0:
        return slice_2d
    return cv2.resize(slice_2d, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_CUBIC)


def _overlay_mask(slice_2d, mask_2d, alpha=0.3, color=(0, 255, 0)):
    # Always return (H, W, 3) RGB to avoid dimension mismatch in np.hstack
    if mask_2d is None or mask_2d.sum() == 0:
        return np.stack([slice_2d] * 3, axis=-1).astype(np.uint8)
    rgb = np.stack([slice_2d] * 3, axis=-1).astype(np.float32)
    mask_bool = mask_2d.astype(bool)
    for c in range(3):
        rgb[..., c] = np.where(mask_bool,
                               rgb[..., c] * (1 - alpha) + color[c] * alpha,
                               rgb[..., c])
    return rgb.astype(np.uint8)


def _compute_iou_3d(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter / union) if union > 0 else 0.0


def _compute_dice_3d(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    return float(2 * inter / (pred.sum() + gt.sum() + 1e-8))


def _extract_triview(volume, center, mask=None):
    z, y, x = [max(0, min(int(c), s - 1)) for c, s in zip(center, volume.shape)]
    axial = volume[z, :, :]
    coronal = volume[:, y, :]
    sagittal = volume[:, :, x]
    if mask is not None:
        return (axial, coronal, sagittal), (mask[z, :, :], mask[:, y, :], mask[:, :, x])
    return axial, coronal, sagittal


def render_triview(volume, center, spacing, mask=None):
    """Render a tri-view (Axial|Coronal|Sagittal) as a PIL Image."""
    if mask is not None:
        (ax, cor, sag), (m_ax, m_cor, m_sag) = _extract_triview(volume, center, mask=mask)
    else:
        ax, cor, sag = _extract_triview(volume, center)
        m_ax = m_cor = m_sag = None

    ax_res = _resample_slice(ax, (spacing[1], spacing[2]))
    cor_res = _resample_slice(cor, (spacing[0], spacing[2]))
    sag_res = _resample_slice(sag, (spacing[0], spacing[1]))

    if m_ax is not None:
        m_ax_r = _resample_slice(m_ax.astype(np.float32), (spacing[1], spacing[2])) > 0.5
        m_cor_r = _resample_slice(m_cor.astype(np.float32), (spacing[0], spacing[2])) > 0.5
        m_sag_r = _resample_slice(m_sag.astype(np.float32), (spacing[0], spacing[1])) > 0.5
    else:
        m_ax_r = m_cor_r = m_sag_r = None

    ax_norm = _normalize_slice(ax_res)
    cor_norm = _normalize_slice(cor_res)
    sag_norm = _normalize_slice(sag_res)

    ax_norm = _overlay_mask(ax_norm, m_ax_r)
    cor_norm = _overlay_mask(cor_norm, m_cor_r)
    sag_norm = _overlay_mask(sag_norm, m_sag_r)

    h = max(ax_norm.shape[0], cor_norm.shape[0], sag_norm.shape[0])

    def _pad(img, th):
        if img.shape[0] < th:
            top = (th - img.shape[0]) // 2
            bot = th - img.shape[0] - top
            pw = [(top, bot), (0, 0)] + ([(0, 0)] if img.ndim == 3 else [])
            return np.pad(img, pw, mode='constant')
        return img

    triview = np.hstack([_pad(ax_norm, h), _pad(cor_norm, h), _pad(sag_norm, h)])
    return Image.fromarray(triview)


def _extract_balanced_json(text: str):
    """Extract the first balanced { ... } JSON object from text."""
    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _parse_action(text: str):
    """Extract JSON action from model output, supporting <tool_call> XML wrapper."""
    # Primary: MedSAM-Agent style <tool_call>{"name": ...}</tool_call>
    tool_match = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL)
    search_text = tool_match.group(1).strip() if tool_match else text

    json_str = _extract_balanced_json(search_text)
    if json_str:
        try:
            obj = json.loads(json_str)
            if "name" in obj:
                return obj
        except json.JSONDecodeError:
            pass

    return {"name": "stop_action"}


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

def build_system_prompt(volume_shape, organ_names=None):
    """Build a system prompt aligned with MedSAM-Agent's tool-calling format."""
    dz, dy, dx = volume_shape
    organ_str = ""
    if organ_names:
        organ_str = f" You will be asked to segment specific organs: {', '.join(organ_names)}."

    tools = [
        {
            "type": "function",
            "function": {
                "name": "add_3d_point",
                "description": "Add a 3D point to initialize or refine the 3D segmentation. Use positive points to include tissue, negative points to exclude tissue.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "point_3d": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 3,
                            "maxItems": 3,
                            "description": f"3D coordinate in voxel space [z, y, x]. Valid ranges: z in [0, {dz-1}], y in [0, {dy-1}], x in [0, {dx-1}]"
                        },
                        "point_type": {
                            "type": "string",
                            "enum": ["positive", "negative"],
                            "description": "Type of point: 'positive' to expand mask, 'negative' to refine mask"
                        }
                    },
                    "required": ["point_3d", "point_type"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "stop_action",
                "description": "Stop the refinement process when the mask accurately covers the target organ.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
    ]
    tools_str = "\n".join([json.dumps(t) for t in tools])

    return f"""You are a professional 3D medical segmentation annotator. You analyze a 3D CT/MRI volume through three orthogonal 2D slice views stitched side-by-side: [Axial | Coronal | Sagittal].

The 3D volume dimensions are: Z={dz}, Y={dy}, X={dx} (voxel coordinates).{organ_str}

The current segmentation mask is shown as a semi-transparent green overlay on each view. Your goal is to ensure the mask exactly covers the entire target organ and excludes all non-target tissue.

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

Example: to place a positive point at z=100, y=200, x=150:
<tool_call>
{{"name": "add_3d_point", "arguments": {{"point_3d": [100, 200, 150], "point_type": "positive"}}}}
</tool_call>

Only use the provided functions to complete your task. Do not invent or assume any other functions. Carefully consider the current mask state and the tri-view slices before each action.

Remember: the three views are [Axial | Coronal | Sagittal] from left to right. The center of each view corresponds to the focus point in 3D. Use this spatial understanding to place accurate 3D coordinates."""


# ---------------------------------------------------------------------------
# Qwen3-VL Agent wrapper
# ---------------------------------------------------------------------------

class QwenVLAgent:
    """Wraps Qwen3-VL for tool-call generation (no training needed)."""

    def __init__(self, model_id: str, device_map="auto", **load_kwargs):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        print(f"Loading Qwen3-VL from {model_id}...")
        defaults = dict(
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
        )
        defaults.update(load_kwargs)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **defaults)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model.eval()
        print(f"Qwen3-VL loaded on {self.model.device}")

    def generate_action(self, image: Image.Image, instruction: str, system_prompt: str,
                        conversation_history: list = None, max_new_tokens: int = 256,
                        temperature: float = 0.1, do_sample: bool = False) -> dict:
        """
        Generate the next action given the current tri-view image and instruction.

        Returns: parsed action dict, e.g. {"name": "add_3d_point", "arguments": {...}}
        """
        # Build messages
        messages = [{"role": "system", "content": system_prompt}]

        if conversation_history:
            messages.extend(conversation_history)

        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction}
            ]
        })

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process vision inputs (proper Qwen3-VL image handling)
        image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)

        # Process inputs
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        # Generate
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.processor.tokenizer.pad_token_id,
        }
        if do_sample:
            generation_kwargs.update({"temperature": temperature, "top_p": 0.9})

        with torch.no_grad():
            generated = self.model.generate(**inputs, **generation_kwargs)

        # Decode only new tokens
        prompt_len = inputs["input_ids"].shape[1]
        response_ids = generated[0, prompt_len:]
        response_text = self.processor.decode(response_ids, skip_special_tokens=True).strip()

        action = _parse_action(response_text)
        action["_raw_output"] = response_text
        return action


# ---------------------------------------------------------------------------
# Agent Loop
# ---------------------------------------------------------------------------

@dataclass
class TurnRecord:
    turn: int
    image_path: str
    action: dict
    raw_output: str
    iou: float
    dice: float
    mask_voxels: int
    center: tuple


@dataclass
class AgentResult:
    volume_id: str
    target_name: str
    turns: list = field(default_factory=list)
    final_iou: float = 0.0
    final_dice: float = 0.0
    success: bool = False
    error: str = ""


def run_agent_loop(
    mllm_agent: QwenVLAgent,
    sam_model,
    volume: np.ndarray,
    gt_mask: np.ndarray,
    target_name: str,
    volume_id: str,
    spacing: tuple = (1.0, 1.0, 1.0),
    max_turns: int = 5,
    output_dir: str = "eval_outputs",
    temperature: float = 0.1,
    seed: int = 42,
) -> AgentResult:
    """
    Run the full agent loop: MLLM → SAM-Med3D → render → repeat.

    Args:
        mllm_agent: Wrapped Qwen3-VL agent
        sam_model: FrozenSAMMed3D instance
        volume: 3D volume array (z, y, x)
        gt_mask: Ground truth binary mask (z, y, x)
        target_name: Organ/structure name (e.g., "liver")
        volume_id: Case identifier
        spacing: Voxel spacing (dz, dy, dx)
        max_turns: Maximum interaction turns
        output_dir: Directory to save tri-view images
        temperature: Sampling temperature for MLLM
        seed: Random seed

    Returns:
        AgentResult with turn-by-turn details and final metrics
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs(output_dir, exist_ok=True)
    result = AgentResult(volume_id=volume_id, target_name=target_name)

    if gt_mask.sum() == 0:
        result.error = "Empty ground truth mask"
        return result

    # Build system prompt with volume info
    system_prompt = build_system_prompt(volume.shape)
    instruction = f"Segment the {target_name}"

    # Initialize SAM-Med3D with this volume (NO gt_mask — avoids bias in crop centering)
    print(f"Setting up SAM-Med3D for {volume_id}/{target_name}...")
    sam_model.set_image(volume)

    # Start at volume center
    center = (volume.shape[0] // 2, volume.shape[1] // 2, volume.shape[2] // 2)
    points = []
    labels = []
    pred_mask = np.zeros(volume.shape, dtype=np.uint8)

    conversation_history = []
    prev_iou = 0.0

    for turn in range(max_turns):
        # --- Render current state ---
        img_name = f"{volume_id}_{target_name}_t{turn}.png"
        img_path = os.path.join(output_dir, img_name)
        triview = render_triview(volume, center, spacing, mask=pred_mask if turn > 0 else None)
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
        print(f"  Turn {turn}: MLLM → {action_name}")
        print(f"    Raw output: {raw_out[:200]}")

        if action_name == "stop_action":
            # Record final turn
            current_iou = _compute_iou_3d(pred_mask, gt_mask)
            current_dice = _compute_dice_3d(pred_mask, gt_mask)
            rec = TurnRecord(
                turn=turn, image_path=img_path,
                action={"name": "stop_action"}, raw_output=raw_out,
                iou=current_iou, dice=current_dice,
                mask_voxels=int(pred_mask.sum()), center=center
            )
            result.turns.append(rec)
            result.final_iou = current_iou
            result.final_dice = current_dice
            result.success = True
            print(f"  → Agent stopped at turn {turn}. Final IoU={current_iou:.4f}, Dice={current_dice:.4f}")
            break

        # --- Parse action coordinates (tolerant of both flat and nested format) ---
        try:
            # Qwen3 tool-calling can emit either:
            #   nested: {"name": "add_3d_point", "arguments": {"point_3d": [...], ...}}
            #   flat:   {"name": "add_3d_point", "point_3d": [...], "point_type": "..."}
            args = action.get("arguments", {})
            pt = args.get("point_3d") or action.get("point_3d")
            pt_type = args.get("point_type") or action.get("point_type", "positive")
            if pt is None:
                print(f"  → No point_3d in action, treating as stop")
                action_name = "stop_action"
                current_iou = _compute_iou_3d(pred_mask, gt_mask)
                current_dice = _compute_dice_3d(pred_mask, gt_mask)
                result.final_iou = current_iou
                result.final_dice = current_dice
                break

            z, y, x = [int(c) for c in pt]
            # Clamp to volume bounds
            z = max(0, min(z, volume.shape[0] - 1))
            y = max(0, min(y, volume.shape[1] - 1))
            x = max(0, min(x, volume.shape[2] - 1))
            action_point = (z, y, x)
        except (ValueError, TypeError, KeyError) as e:
            print(f"  → Invalid action format: {e}, treating as stop")
            current_iou = _compute_iou_3d(pred_mask, gt_mask)
            result.final_iou = current_iou
            result.final_dice = _compute_dice_3d(pred_mask, gt_mask)
            break

        label = 1 if pt_type == "positive" else 0
        points.append(action_point)
        labels.append(label)

        # --- SAM-Med3D inference ---
        pred_mask = sam_model.predict_mask(points, labels)
        current_iou = _compute_iou_3d(pred_mask, gt_mask)
        current_dice = _compute_dice_3d(pred_mask, gt_mask)

        delta_iou = current_iou - prev_iou
        prev_iou = current_iou

        print(f"  Turn {turn}: point={action_point} ({pt_type}), IoU={current_iou:.4f} (Δ={delta_iou:+.4f}), Dice={current_dice:.4f}")

        # Update focus center to mask centroid
        if pred_mask.sum() > 0:
            from scipy.ndimage import center_of_mass
            center = tuple(int(c) for c in center_of_mass(pred_mask))
        else:
            center = action_point

        # Record turn
        rec = TurnRecord(
            turn=turn, image_path=img_path,
            action={"name": action_name, "arguments": {
                "point_3d": list(action_point), "point_type": pt_type
            }},
            raw_output=raw_out,
            iou=current_iou, dice=current_dice,
            mask_voxels=int(pred_mask.sum()), center=center
        )
        result.turns.append(rec)

        # Append to conversation history (text-only — current tri-view encodes full state)
        # Storing PIL images in history causes VRAM exhaustion after 3+ turns.
        conversation_history.append({
            "role": "user",
            "content": f"[Tri-view image — mask shown in green overlay] {instruction}"
        })
        conversation_history.append({
            "role": "assistant",
            "content": raw_out
        })

    else:
        # Max turns reached
        result.final_iou = _compute_iou_3d(pred_mask, gt_mask)
        result.final_dice = _compute_dice_3d(pred_mask, gt_mask)
        result.success = True
        print(f"  → Max turns ({max_turns}) reached. Final IoU={result.final_iou:.4f}")

    return result


# ---------------------------------------------------------------------------
# Batch Evaluation Runner
# ---------------------------------------------------------------------------

def evaluate_dataset(
    mllm_agent: QwenVLAgent,
    sam_model,
    dataset_entries: list,
    output_dir: str = "eval_outputs",
    max_turns: int = 5,
    temperature: float = 0.1,
    seed: int = 42,
):
    """Run agent evaluation over a list of {volume, mask, target_name, volume_id, spacing} entries."""
    results = []
    total = len(dataset_entries)

    for i, entry in enumerate(dataset_entries):
        vol_id = entry["volume_id"]
        tgt = entry["target_name"]
        print(f"\n{'='*60}")
        print(f"[{i+1}/{total}] {vol_id} / {tgt}")
        print(f"{'='*60}")

        try:
            res = run_agent_loop(
                mllm_agent=mllm_agent,
                sam_model=sam_model,
                volume=entry["volume"],
                gt_mask=entry["mask"],
                target_name=tgt,
                volume_id=vol_id,
                spacing=entry.get("spacing", (1.0, 1.0, 1.0)),
                max_turns=max_turns,
                output_dir=os.path.join(output_dir, vol_id, tgt),
                temperature=temperature,
                seed=seed,
            )
            results.append(res)
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results.append(AgentResult(
                volume_id=vol_id, target_name=tgt,
                error=str(e)
            ))

    # Summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    successful = [r for r in results if r.success]
    print(f"Total cases: {len(results)}")
    print(f"Successful:  {len(successful)}")
    print(f"Failed:      {len(results) - len(successful)}")

    if successful:
        ious = [r.final_iou for r in successful]
        dices = [r.final_dice for r in successful]
        turns_list = [len(r.turns) for r in successful]
        print(f"\nMetrics (mean ± std):")
        print(f"  IoU:   {np.mean(ious):.4f} ± {np.std(ious):.4f}")
        print(f"  Dice:  {np.mean(dices):.4f} ± {np.std(dices):.4f}")
        print(f"  Turns: {np.mean(turns_list):.2f} ± {np.std(turns_list):.2f}")

    # Save detailed results
    results_json = []
    for r in results:
        results_json.append({
            "volume_id": r.volume_id,
            "target_name": r.target_name,
            "success": r.success,
            "error": r.error,
            "final_iou": r.final_iou,
            "final_dice": r.final_dice,
            "num_turns": len(r.turns),
            "turns": [
                {
                    "turn": t.turn,
                    "action": t.action,
                    "iou": t.iou,
                    "dice": t.dice,
                    "mask_voxels": t.mask_voxels,
                }
                for t in r.turns
            ]
        })

    summary_path = os.path.join(output_dir, "evaluation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\nDetailed results saved to {summary_path}")

    return results


# ---------------------------------------------------------------------------
# Single-volume evaluation (convenience wrapper)
# ---------------------------------------------------------------------------

def evaluate_single(
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
    volume_path: str = None,
    mask_path: str = None,
    target_name: str = "organ",
    sam_ckpt: str = None,
    output_dir: str = "eval_outputs",
    max_turns: int = 5,
    temperature: float = 0.1,
    **model_kwargs,
):
    """Evaluate on a single volume. Loads data, runs agent, prints results."""
    from data_generation.totalsegmentator_loader import load_nifti
    from models.sam_med3d_wrapper import FrozenSAMMed3D

    vol, spacing = load_nifti(volume_path)
    mask, _ = load_nifti(mask_path)
    mask = (mask > 0).astype(np.uint8)
    vol_id = os.path.splitext(os.path.basename(volume_path))[0]

    print(f"Volume: {vol.shape}, spacing={spacing}, mask voxels={int(mask.sum()):,}")

    mllm = QwenVLAgent(model_id, **model_kwargs)
    sam = FrozenSAMMed3D(checkpoint_path=sam_ckpt) if sam_ckpt else FrozenSAMMed3D()

    result = run_agent_loop(
        mllm_agent=mllm,
        sam_model=sam,
        volume=vol,
        gt_mask=mask,
        target_name=target_name,
        volume_id=vol_id,
        spacing=spacing,
        max_turns=max_turns,
        output_dir=output_dir,
        temperature=temperature,
    )

    print(f"\n{'='*40}")
    print(f"FINAL: IoU={result.final_iou:.4f}, Dice={result.final_dice:.4f}, Turns={len(result.turns)}")
    print(f"{'='*40}")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MedSAM-Agent3D: Agentic Inference (Non-Trained MLLM)")

    # Model args
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct",
                        help="Qwen3-VL model ID (HuggingFace) or local path")
    parser.add_argument("--sam-ckpt", default=None,
                        help="Path to SAM-Med3D checkpoint (default: sam_med3d_turbo.pth in repo root)")

    # Data args
    parser.add_argument("--volume", required=True,
                        help="Path to NIfTI volume (.nii.gz)")
    parser.add_argument("--mask", default=None,
                        help="Path to NIfTI ground truth mask (.nii.gz)")
    parser.add_argument("--target", default="organ",
                        help="Target organ/structure name (e.g., 'liver')")

    # Loop config
    parser.add_argument("--max-turns", type=int, default=5,
                        help="Maximum interaction turns")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="MLLM sampling temperature")
    parser.add_argument("--output-dir", default="eval_outputs",
                        help="Output directory for tri-view images and results")

    # Model loading
    parser.add_argument("--device", default="auto",
                        help="Device map ('auto', 'cuda', 'cpu')")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load MLLM in 4-bit quantization")

    args = parser.parse_args()

    model_kwargs = {}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    model_kwargs["device_map"] = args.device

    evaluate_single(
        model_id=args.model,
        volume_path=args.volume,
        mask_path=args.mask,
        target_name=args.target,
        sam_ckpt=args.sam_ckpt,
        output_dir=args.output_dir,
        max_turns=args.max_turns,
        temperature=args.temperature,
        **model_kwargs,
    )


if __name__ == "__main__":
    main()
