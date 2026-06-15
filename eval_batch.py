"""
MedSAM-Agent3D — Batch Evaluation Runner
========================================
Runs the agentic inference pipeline on multiple volumes from TotalSegmentator
(or any NIfTI dataset) to evaluate base Qwen3-VL performance.

Usage:
    python eval_batch.py \
        --ts-dir Totalsegmentator_dataset_v201 \
        --model Qwen/Qwen3-VL-8B-Instruct \
        --max-samples 10 \
        --max-turns 5 \
        --output-dir eval_outputs
"""

import os
import sys
import json
import argparse
import random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_generation.totalsegmentator_loader import get_totalsegmentator_dataset, load_nifti
from models.sam_med3d_wrapper import FrozenSAMMed3D
from agent_inference import QwenVLAgent, run_agent_loop, AgentResult


def main():
    parser = argparse.ArgumentParser(description="MedSAM-Agent3D: Batch Evaluation")

    # Data
    parser.add_argument("--ts-dir", default="Totalsegmentator_dataset_v201",
                        help="Path to TotalSegmentator dataset directory")
    parser.add_argument("--max-samples", type=int, default=10,
                        help="Maximum samples to evaluate")
    parser.add_argument("--shuffle", action="store_true", default=True,
                        help="Shuffle dataset before sampling")
    parser.add_argument("--seed", type=int, default=42)

    # Models
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct",
                        help="Qwen3-VL model ID or local path")
    parser.add_argument("--sam-ckpt", default=None,
                        help="Path to SAM-Med3D checkpoint")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load MLLM in 4-bit quantization")
    parser.add_argument("--device", default="auto")

    # Loop
    parser.add_argument("--max-turns", type=int, default=5,
                        help="Maximum interaction turns per sample")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="MLLM sampling temperature (0=deterministic)")

    # Output
    parser.add_argument("--output-dir", default="eval_outputs",
                        help="Output directory for results and images")

    args = parser.parse_args()

    # --- Setup ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load dataset ---
    print(f"Loading TotalSegmentator dataset from {args.ts_dir}...")
    dataset = get_totalsegmentator_dataset(args.ts_dir)

    if args.shuffle:
        random.shuffle(dataset)

    dataset = dataset[:args.max_samples]
    if not dataset:
        print("ERROR: No samples found. Check --ts-dir path.")
        sys.exit(1)

    print(f"Evaluating {len(dataset)} samples")

    # --- Load models ---
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

    mllm = QwenVLAgent(args.model, **model_kwargs)

    ckpt = args.sam_ckpt or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "sam_med3d_turbo.pth"
    )
    sam = FrozenSAMMed3D(checkpoint_path=ckpt)

    # --- Run evaluation ---
    results = []

    for i, entry in enumerate(dataset):
        vol_id = entry["volume_id"]
        tgt = entry["target_name"]
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(dataset)}] {vol_id} / {tgt}")
        print(f"{'='*60}")

        try:
            vol, spacing = load_nifti(entry["ct_path"])
            mask, _ = load_nifti(entry["mask_path"])
            mask = (mask > 0).astype(np.uint8)

            if mask.sum() == 0:
                print(f"  SKIP: empty mask")
                results.append({
                    "volume_id": vol_id, "target_name": tgt,
                    "success": False, "error": "empty_mask"
                })
                continue

            print(f"  Volume shape: {vol.shape}, mask voxels: {int(mask.sum()):,}")

            result = run_agent_loop(
                mllm_agent=mllm,
                sam_model=sam,
                volume=vol,
                gt_mask=mask,
                target_name=tgt,
                volume_id=vol_id,
                spacing=spacing,
                max_turns=args.max_turns,
                output_dir=os.path.join(args.output_dir, vol_id, tgt),
                temperature=args.temperature,
                seed=args.seed,
            )

            results.append({
                "volume_id": result.volume_id,
                "target_name": result.target_name,
                "success": result.success,
                "error": result.error,
                "final_iou": result.final_iou,
                "final_dice": result.final_dice,
                "num_turns": len(result.turns),
                "turns": [
                    {
                        "turn": t.turn,
                        "action": t.action,
                        "iou": t.iou,
                        "dice": t.dice,
                        "mask_voxels": t.mask_voxels,
                    }
                    for t in result.turns
                ]
            })

        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results.append({
                "volume_id": vol_id, "target_name": tgt,
                "success": False, "error": str(e)
            })

    # --- Summary ---
    print(f"\n{'='*60}")
    print("BATCH EVALUATION SUMMARY")
    print(f"{'='*60}")
    successful = [r for r in results if r.get("success")]
    print(f"Total samples: {len(results)}")
    print(f"Successful:    {len(successful)}")
    print(f"Failed:        {len(results) - len(successful)}")

    if successful:
        ious = [r["final_iou"] for r in successful]
        dices = [r["final_dice"] for r in successful]
        turns = [r["num_turns"] for r in successful]
        print(f"\nMetrics (mean ± std):")
        print(f"  IoU:   {np.mean(ious):.4f} ± {np.std(ious):.4f}")
        print(f"  Dice:  {np.mean(dices):.4f} ± {np.std(dices):.4f}")
        print(f"  Turns: {np.mean(turns):.2f} ± {np.std(turns):.2f}")

        # Per-organ breakdown (if multiple)
        organs = {}
        for r in successful:
            org = r["target_name"]
            if org not in organs:
                organs[org] = {"ious": [], "dices": [], "turns": []}
            organs[org]["ious"].append(r["final_iou"])
            organs[org]["dices"].append(r["final_dice"])
            organs[org]["turns"].append(r["num_turns"])

        if len(organs) > 1:
            print(f"\nPer-organ breakdown:")
            for org, metrics in sorted(organs.items()):
                print(f"  {org}: IoU={np.mean(metrics['ious']):.4f}±{np.std(metrics['ious']):.4f}, "
                      f"Dice={np.mean(metrics['dices']):.4f}±{np.std(metrics['dices']):.4f}, "
                      f"Turns={np.mean(metrics['turns']):.2f}±{np.std(metrics['turns']):.2f} "
                      f"(n={len(metrics['ious'])})")

    # Save results
    summary_path = os.path.join(args.output_dir, "batch_evaluation.json")
    with open(summary_path, 'w') as f:
        json.dump({
            "args": vars(args),
            "summary": {
                "total": len(results),
                "successful": len(successful),
                "failed": len(results) - len(successful),
                "mean_iou": float(np.mean(ious)) if successful else 0,
                "mean_dice": float(np.mean(dices)) if successful else 0,
                "mean_turns": float(np.mean(turns)) if successful else 0,
            },
            "results": results
        }, f, indent=2)

    print(f"\nFull results saved to {summary_path}")


if __name__ == "__main__":
    main()
