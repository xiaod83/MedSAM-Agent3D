"""
Evaluation runners: single-case and batch evaluation.

Provides ``evaluate_single()`` for one volume and ``evaluate_dataset()``
for batch evaluation over multiple volume/target pairs. Handles data
loading, metrics aggregation, and JSON result serialization.

References:
    MedSAM-Agent run_single_image_inference() and save_metrics_json().
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .types import AgentResult, compute_metrics
from .agent import MLLMAgent, QwenVLAgent
from .loop import run_agent_loop


def evaluate_single(
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
    volume_path: Optional[str] = None,
    mask_path: Optional[str] = None,
    target_name: str = "organ",
    sam_ckpt: Optional[str] = None,
    output_dir: str = "eval_outputs",
    max_turns: int = 5,
    temperature: float = 0.1,
    **model_kwargs,
) -> AgentResult:
    """Evaluate the agent loop on a single NIfTI volume.

    Loads data, initializes models, runs the agent loop, and prints results.

    Args:
        model_id: HuggingFace model ID for Qwen3-VL.
        volume_path: Path to NIfTI volume (.nii.gz).
        mask_path: Path to NIfTI ground truth mask (.nii.gz).
        target_name: Organ/structure name.
        sam_ckpt: Path to SAM-Med3D checkpoint.
        output_dir: Output directory for images and results.
        max_turns: Max agent interaction turns.
        temperature: MLLM sampling temperature.
        **model_kwargs: Passed to QwenVLAgent (device_map, quantization_config, etc.).

    Returns:
        AgentResult with final metrics.
    """
    from models.sam_med3d_wrapper import FrozenSAMMed3D
    from utils.io_utils import load_nifti

    vol, spacing = load_nifti(volume_path)
    mask, _ = load_nifti(mask_path)
    mask = (mask > 0).astype(np.uint8)
    vol_id = os.path.splitext(os.path.basename(volume_path))[0]
    vol_id = vol_id.replace(".nii", "").replace(".gz", "")

    print(f"Volume: {vol.shape}, spacing={spacing}, "
          f"mask voxels={int(mask.sum()):,}")

    mllm = QwenVLAgent(model_id, **model_kwargs)
    sam = FrozenSAMMed3D(
        checkpoint_path=sam_ckpt) if sam_ckpt else FrozenSAMMed3D()

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
    print(f"FINAL: IoU={result.final_iou:.4f}, "
          f"Dice={result.final_dice:.4f}, "
          f"Turns={len(result.turns)}")
    print(f"{'='*40}")
    return result


def evaluate_dataset(
    mllm_agent: MLLMAgent,
    sam_model: "FrozenSAMMed3D",
    dataset_entries: List[Dict[str, Any]],
    output_dir: str = "eval_outputs",
    max_turns: int = 5,
    temperature: float = 0.1,
    seed: int = 42,
) -> List[AgentResult]:
    """Run agent evaluation over a list of volume/mask entries.

    Args:
        mllm_agent: Pre-loaded MLLM agent.
        sam_model: Pre-loaded SAM-Med3D model.
        dataset_entries: List of dicts, each with:
            - volume_path (or volume): path or np.ndarray
            - mask_path (or mask): path or np.ndarray
            - target_name: str
            - volume_id: str
            - spacing: optional (dz, dy, dx)
        output_dir: Base output directory.
        max_turns: Max turns per sample.
        temperature: MLLM sampling temperature.
        seed: Random seed.

    Returns:
        List of AgentResult objects.
    """
    results: List[AgentResult] = []
    total = len(dataset_entries)

    for i, entry in enumerate(dataset_entries):
        vol_id = entry["volume_id"]
        tgt = entry["target_name"]
        print(f"\n{'='*60}")
        print(f"[{i+1}/{total}] {vol_id} / {tgt}")
        print(f"{'='*60}")

        try:
            # Load volume if path provided, or use pre-loaded array
            volume = entry.get("volume")
            spacing_val = entry.get("spacing", (1.0, 1.0, 1.0))
            if volume is None and "volume_path" in entry:
                from utils.io_utils import load_nifti
                volume, spacing_val = load_nifti(entry["volume_path"])

            # Load mask
            mask = entry.get("mask")
            if mask is None and "mask_path" in entry:
                from utils.io_utils import load_nifti
                mask, _ = load_nifti(entry["mask_path"])
                mask = (mask > 0).astype(np.uint8)

            res = run_agent_loop(
                mllm_agent=mllm_agent,
                sam_model=sam_model,
                volume=volume,
                gt_mask=mask,
                target_name=tgt,
                volume_id=vol_id,
                spacing=spacing_val,
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
                error=str(e),
            ))

    # --- Summary ---
    _print_summary(results)

    # --- Save ---
    _save_results_json(results, output_dir)

    return results


def _print_summary(results: List[AgentResult]):
    """Print aggregate evaluation summary."""
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

        # Per-organ breakdown (if multiple)
        organs: Dict[str, Dict[str, List[float]]] = {}
        for r in successful:
            org = r.target_name
            if org not in organs:
                organs[org] = {"ious": [], "dices": [], "turns": []}
            organs[org]["ious"].append(r.final_iou)
            organs[org]["dices"].append(r.final_dice)
            organs[org]["turns"].append(float(len(r.turns)))

        if len(organs) > 1:
            print(f"\nPer-organ breakdown:")
            for org, metrics in sorted(organs.items()):
                print(
                    f"  {org}: "
                    f"IoU={np.mean(metrics['ious']):.4f}±{np.std(metrics['ious']):.4f}, "
                    f"Dice={np.mean(metrics['dices']):.4f}±{np.std(metrics['dices']):.4f}, "
                    f"Turns={np.mean(metrics['turns']):.2f}±{np.std(metrics['turns']):.2f} "
                    f"(n={len(metrics['ious'])})"
                )


def _save_results_json(results: List[AgentResult], output_dir: str):
    """Serialize results to a JSON file."""
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
            ],
        })

    summary_path = os.path.join(output_dir, "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nDetailed results saved to {summary_path}")
