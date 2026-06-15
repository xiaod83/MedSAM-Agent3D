"""
MedSAM-Agent3D — Data Pipeline
===============================
Processes TotalSegmentator NIfTI volumes through the SAM-Med3D model
to generate interactive segmentation trajectories for MLLM training.
"""
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.sam_med3d_wrapper import FrozenSAMMed3D
from data_generation.trajectory_generator import generate_trajectory
from data_generation.totalsegmentator_loader import get_totalsegmentator_dataset, load_nifti
from training.sft_formatter import format_trajectories_for_sft


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def process_totalsegmentator(ts_dir: str, max_samples: int = 10):
    print(f"Loading TotalSegmentator dataset from {ts_dir}...")
    dataset = get_totalsegmentator_dataset(ts_dir)

    if not dataset:
        print("No TotalSegmentator data found. Expected structure:")
        print(f"  {ts_dir}/sXXXX/ct.nii.gz + segmentations/*.nii.gz")
        return

    print(f"Found {len(dataset)} volume-mask pairs.")

    sam_model = FrozenSAMMed3D()
    processed_count = 0
    scanned = 0
    max_scan = max_samples * 50  # abort after scanning too many empties

    for sample in dataset:
        if processed_count >= max_samples:
            break
        if scanned >= max_scan:
            print(f"Reached scan limit ({max_scan}) with only {processed_count} valid samples.")
            break

        scanned += 1
        print(f"\n[{scanned}] {sample['volume_id']} / {sample['target_name']}...", end=' ')

        try:
            vol, spacing = load_nifti(sample['ct_path'])
            mask, _ = load_nifti(sample['mask_path'])
            mask = (mask > 0).astype(np.uint8)

            if mask.sum() == 0:
                print("empty mask, skip")
                continue

            print(f"mask={int(mask.sum()):,d}")

            generate_trajectory(
                sam_model=sam_model,
                volume=vol,
                gt_mask=mask,
                target_name=sample['target_name'],
                volume_id=sample['volume_id'],
                spacing=spacing,
            )
            processed_count += 1

        except Exception as e:
            import traceback
            print(f"ERROR: {e}")
            traceback.print_exc()

    print(f"\nDone. Scanned {scanned}, processed {processed_count} samples.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MedSAM-Agent3D data pipeline")
    parser.add_argument("--ts_dir",
                        default=os.path.join("Totalsegmentator_dataset_v201"),
                        help="Path to TotalSegmentator v2.01 CT data directory")
    parser.add_argument("--max_samples", type=int, default=10,
                        help="Max volume-mask pairs to process")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    print("MedSAM-Agent3D Data Pipeline (TotalSegmentator mode)")
    print(f"  Data dir:     {args.ts_dir}")
    print(f"  Max samples:  {args.max_samples}")
    print(f"  Seed:         {args.seed}")
    print()

    process_totalsegmentator(ts_dir=args.ts_dir, max_samples=args.max_samples)

    print("\nFormatting trajectories for SFT...")
    format_trajectories_for_sft("data/trajectories", "data/sft_dataset.jsonl")


if __name__ == "__main__":
    main()
