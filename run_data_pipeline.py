import numpy as np
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sam_med3d_wrapper import FrozenSAMMed3D
from data_generation.trajectory_generator import generate_trajectory
from data_generation.totalsegmentator_loader import get_totalsegmentator_dataset, load_nifti
from training.sft_formatter import format_trajectories_for_sft

def process_totalsegmentator(ts_dir="data/totalsegmentator", max_samples=10):
    print(f"Loading TotalSegmentator dataset from {ts_dir}...")
    dataset = get_totalsegmentator_dataset(ts_dir)
    
    if not dataset:
        print("No TotalSegmentator data found. Please ensure it is placed in the correct directory structure.")
        return
        
    print(f"Found {len(dataset)} volume-mask pairs.")
    
    sam_model = FrozenSAMMed3D()
    
    # Process dataset
    processed_count = 0
    for sample in dataset:
        if processed_count >= max_samples:
            break
            
        print(f"\nProcessing {sample['volume_id']} - {sample['target_name']}...")
        
        try:
            vol, spacing = load_nifti(sample['ct_path'])
            mask, _ = load_nifti(sample['mask_path'])
            mask = (mask > 0).astype(np.uint8) # Binarize
            
            generate_trajectory(
                sam_model=sam_model,
                volume=vol,
                gt_mask=mask,
                target_name=sample['target_name'],
                volume_id=sample['volume_id'],
                spacing=spacing
            )
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {sample['volume_id']}: {e}")

def create_dummy_dataset(num_volumes=3):
    """Creates a few dummy 3D numpy arrays to simulate a dataset."""
    os.makedirs("data/dummy_dataset", exist_ok=True)
    
    for i in range(num_volumes):
        # Create a 64x128x128 volume
        vol = np.random.randn(64, 128, 128) * 50 + 100
        
        # Create a mock ground truth mask (a sphere)
        z, y, x = np.ogrid[-32:32, -64:64, -64:64]
        # Random center offset
        z_c, y_c, x_c = np.random.randint(-10, 10, 3)
        mask = (x-x_c)**2 + (y-y_c)**2 + (z-z_c)**2 <= 15**2
        
        np.save(f"data/dummy_dataset/vol_{i}.npy", vol)
        np.save(f"data/dummy_dataset/mask_{i}.npy", mask.astype(np.uint8))
        print(f"Saved dummy volume {i}")

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

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run the MedSAM-Agent3D data pipeline")
    parser.add_argument("--mode", choices=["dummy", "totalsegmentator"], default="dummy", 
                        help="Data source mode")
    parser.add_argument("--ts_dir", default="data/totalsegmentator", help="Path to TotalSegmentator data")
    parser.add_argument("--max_samples", type=int, default=10, help="Max samples to process")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    # Set seeds for reproducibility
    set_seed(args.seed if hasattr(args, 'seed') else 42)
    
    print(f"Initializing Data Pipeline in {args.mode} mode...")
    
    if args.mode == "dummy":
        create_dummy_dataset()
        sam_model = FrozenSAMMed3D()
        
        # Process dataset
        for i in range(3):
            print(f"\nProcessing volume {i}...")
            vol = np.load(f"data/dummy_dataset/vol_{i}.npy")
            mask = np.load(f"data/dummy_dataset/mask_{i}.npy")
            
            generate_trajectory(
                sam_model=sam_model,
                volume=vol,
                gt_mask=mask,
                target_name="liver",
                volume_id=f"dummy_{i}"
            )
    elif args.mode == "totalsegmentator":
        process_totalsegmentator(ts_dir=args.ts_dir, max_samples=args.max_samples)
        
    print("\nFormatting generated trajectories for SFT...")
    format_trajectories_for_sft("data/trajectories", "data/sft_dataset.jsonl")

if __name__ == "__main__":
    main()