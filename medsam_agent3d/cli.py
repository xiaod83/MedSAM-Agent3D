"""
CLI entry points for single-case and batch evaluation.

Provides ``main_single`` and ``main_batch`` functions that wire up
argument parsing with the agent loop and evaluation runners.
"""

import argparse
import os
import random
import sys

import numpy as np
import torch


def _add_common_args(parser: argparse.ArgumentParser):
    """Add arguments shared between single and batch evaluation."""
    # Model args
    parser.add_argument(
        "--model", default="Qwen/Qwen3-VL-8B-Instruct",
        help="Qwen3-VL model ID (HuggingFace) or local path",
    )
    parser.add_argument(
        "--sam-ckpt", default=None,
        help="Path to SAM-Med3D checkpoint (default: sam_med3d_turbo.pth)",
    )

    # Loop config
    parser.add_argument(
        "--max-turns", type=int, default=5,
        help="Maximum interaction turns",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1,
        help="MLLM sampling temperature",
    )
    parser.add_argument(
        "--output-dir", default="eval_outputs",
        help="Output directory for tri-view images and results",
    )

    # Model loading
    parser.add_argument(
        "--device", default="auto",
        help="Device map ('auto', 'cuda', 'cpu')",
    )
    parser.add_argument(
        "--load-in-4bit", action="store_true",
        help="Load MLLM in 4-bit quantization",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )


def _build_model_kwargs(args) -> dict:
    """Build model kwargs from CLI args."""
    model_kwargs = {"device_map": args.device}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    return model_kwargs


# ---------------------------------------------------------------------------
# Single-case CLI
# ---------------------------------------------------------------------------

def main_single(args=None):
    """CLI entry point for single-volume evaluation.

    Usage:
        python -m medsam_agent3d.cli single --volume path/to/ct.nii.gz \\
            --mask path/to/liver.nii.gz --target liver
    """
    parser = argparse.ArgumentParser(
        description="MedSAM-Agent3D: Single-case agentic inference",
    )

    _add_common_args(parser)

    parser.add_argument(
        "--volume", required=True,
        help="Path to NIfTI volume (.nii.gz)",
    )
    parser.add_argument(
        "--mask", default=None,
        help="Path to NIfTI ground truth mask (.nii.gz)",
    )
    parser.add_argument(
        "--target", default="organ",
        help="Target organ/structure name (e.g., 'liver')",
    )

    parsed = parser.parse_args(args)

    from .evaluation import evaluate_single

    evaluate_single(
        model_id=parsed.model,
        volume_path=parsed.volume,
        mask_path=parsed.mask,
        target_name=parsed.target,
        sam_ckpt=parsed.sam_ckpt,
        output_dir=parsed.output_dir,
        max_turns=parsed.max_turns,
        temperature=parsed.temperature,
        **_build_model_kwargs(parsed),
    )


# ---------------------------------------------------------------------------
# Batch CLI
# ---------------------------------------------------------------------------

def main_batch(args=None):
    """CLI entry point for batch evaluation on TotalSegmentator.

    Usage:
        python -m medsam_agent3d.cli batch --ts-dir Totalsegmentator_dataset_v201 \\
            --max-samples 10
    """
    parser = argparse.ArgumentParser(
        description="MedSAM-Agent3D: Batch evaluation on TotalSegmentator",
    )

    _add_common_args(parser)

    # Data
    parser.add_argument(
        "--ts-dir", default="Totalsegmentator_dataset_v201",
        help="Path to TotalSegmentator dataset directory",
    )
    parser.add_argument(
        "--max-samples", type=int, default=10,
        help="Maximum samples to evaluate",
    )
    parser.add_argument(
        "--shuffle", action="store_true", default=True,
        help="Shuffle dataset before sampling",
    )

    parsed = parser.parse_args(args)

    # Add project root to path for imports
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from utils.io_utils import get_totalsegmentator_dataset, load_nifti
    from models.sam_med3d_wrapper import FrozenSAMMed3D
    from .agent import QwenVLAgent
    from .evaluation import evaluate_dataset

    # --- Setup ---
    random.seed(parsed.seed)
    np.random.seed(parsed.seed)
    torch.manual_seed(parsed.seed)

    os.makedirs(parsed.output_dir, exist_ok=True)

    # --- Load dataset ---
    print(f"Loading TotalSegmentator dataset from {parsed.ts_dir}...")
    dataset = get_totalsegmentator_dataset(parsed.ts_dir)

    if parsed.shuffle:
        random.shuffle(dataset)

    dataset = dataset[:parsed.max_samples]
    if not dataset:
        print("ERROR: No samples found. Check --ts-dir path.")
        sys.exit(1)

    print(f"Evaluating {len(dataset)} samples")

    # --- Load models ---
    mllm = QwenVLAgent(parsed.model, **_build_model_kwargs(parsed))

    ckpt = parsed.sam_ckpt or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "sam_med3d_turbo.pth",
    )
    sam = FrozenSAMMed3D(checkpoint_path=ckpt)

    # --- Run evaluation ---
    evaluate_dataset(
        mllm_agent=mllm,
        sam_model=sam,
        dataset_entries=dataset,
        output_dir=parsed.output_dir,
        max_turns=parsed.max_turns,
        temperature=parsed.temperature,
        seed=parsed.seed,
    )


# ---------------------------------------------------------------------------
# Main dispatch (for backward compatibility)
# ---------------------------------------------------------------------------

def main():
    """Legacy dispatch: if --ts-dir is provided, run batch; else single."""
    # Parse just enough to detect mode
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--ts-dir", default=None)
    pre_parser.add_argument("--volume", default=None)
    pre_args, remaining = pre_parser.parse_known_args()

    if pre_args.ts_dir:
        main_batch(remaining)
    else:
        main_single(remaining)
