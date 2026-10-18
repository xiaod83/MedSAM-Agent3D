"""
CLI entry points for single-case and batch evaluation.

Provides ``main_single`` and ``main_batch`` functions that wire up
argument parsing with the agent loop and evaluation runners.
"""

import argparse
import os
import random
import sys



def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

    parser.add_argument("--sam-device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="SAM hardware; auto uses CPU when available CUDA memory is insufficient")

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
        "--output-dir", default="outputs/inference",
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


def _validate_args(parser, args):
    import math
    import torch
    if getattr(args, 'sam_device', 'auto') == 'cuda' and not torch.cuda.is_available():
        parser.error('--sam-device cuda requires an available CUDA GPU; use --sam-device cpu or auto')
    if args.max_turns < 1:
        parser.error("--max-turns must be positive")
    if hasattr(args, "max_samples") and args.max_samples < 1:
        parser.error("--max-samples must be positive")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error("--temperature must be finite and nonnegative")

def _build_model_kwargs(args) -> dict:
    """Build model kwargs from CLI args."""
    import importlib.util
    import torch
    device = getattr(args, "device", "auto")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requires a CUDA-capable PyTorch installation and available GPU; use --device cpu or auto")
    model_kwargs = {"device_map": device}
    if getattr(args, "load_in_4bit", False):
        if importlib.util.find_spec("bitsandbytes") is None:
            raise RuntimeError("--load-in-4bit requires bitsandbytes; install it with python -m pip install bitsandbytes or omit --load-in-4bit")
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=(torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16 if torch.cuda.is_available() else torch.float32),
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
    _validate_args(parser, parsed)

    from .evaluation import evaluate_single

    result = evaluate_single(
        model_id=parsed.model,
        volume_path=parsed.volume,
        mask_path=parsed.mask,
        target_name=parsed.target,
        sam_ckpt=parsed.sam_ckpt,
        sam_device=parsed.sam_device,
        output_dir=parsed.output_dir,
        max_turns=parsed.max_turns,
        temperature=parsed.temperature,
        seed=parsed.seed,
        **_build_model_kwargs(parsed),
    )
    if not result.success:
        print(f'Inference failed: {result.error or result.termination_reason}', file=sys.stderr)
        raise SystemExit(1)
    if result.quality_status == "no_overlap" or result.target_overlap is False:
        print("Segmentation quality failure: prediction has zero overlap with the target ground truth "
              "(quality_status=no_overlap). Execution completed; inspect trajectory.json.", file=sys.stderr)
        raise SystemExit(2)
    return result


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

    parser.add_argument('--split', choices=['train', 'val', 'test', 'all'], default='test', help='Dataset metadata split; default held-out test')
    # Data
    parser.add_argument(
        "--ts-dir", default="data/Totalsegmentator_dataset_v201",
        help="Path to TotalSegmentator dataset directory",
    )
    parser.add_argument(
        "--max-samples", type=int, default=10,
        help="Maximum samples to evaluate",
    )
    parser.add_argument(
        "--shuffle", action=argparse.BooleanOptionalAction, default=True,
        help="Shuffle dataset before sampling",
    )

    parsed = parser.parse_args(args)
    _validate_args(parser, parsed)

    # Add project root to path for imports
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from medsam_agent3d.io import get_totalsegmentator_dataset, load_nifti
    from medsam_agent3d.sam import FrozenSAMMed3D
    from .agent import QwenVLAgent
    from .evaluation import evaluate_dataset

    # --- Setup ---
    seed_everything(parsed.seed)

    os.makedirs(parsed.output_dir, exist_ok=True)

    # --- Load dataset ---
    print(f"Loading TotalSegmentator dataset from {parsed.ts_dir}...")
    dataset = get_totalsegmentator_dataset(parsed.ts_dir, split=None if parsed.split == "all" else parsed.split)

    if parsed.shuffle:
        random.shuffle(dataset)

    dataset = dataset[:parsed.max_samples]
    if not dataset:
        print("ERROR: No samples found. Check --ts-dir path.")
        sys.exit(1)

    print(f"Evaluating {len(dataset)} samples")

    # --- Load models ---
    mllm = QwenVLAgent(parsed.model, **_build_model_kwargs(parsed))

    sam = (FrozenSAMMed3D(checkpoint_path=parsed.sam_ckpt, device=parsed.sam_device)
           if parsed.sam_ckpt else FrozenSAMMed3D(device=parsed.sam_device))

    # --- Run evaluation ---
    results = evaluate_dataset(
        mllm_agent=mllm,
        sam_model=sam,
        dataset_entries=dataset,
        output_dir=parsed.output_dir,
        max_turns=parsed.max_turns,
        temperature=parsed.temperature,
        seed=parsed.seed,
    )
    if any(not result.success for result in results):
        print('Batch finished with failed cases; see evaluation_summary.json', file=sys.stderr)
        raise SystemExit(1)
    if any(result.quality_status == "no_overlap" or result.target_overlap is False for result in results):
        print("Batch segmentation quality failure: at least one prediction has zero target overlap "
              "(quality_status=no_overlap); see evaluation_summary.json", file=sys.stderr)
        raise SystemExit(2)
    return results


# ---------------------------------------------------------------------------
# Main dispatch (for backward compatibility)
# ---------------------------------------------------------------------------

def main(argv=None):
    """Legacy dispatch: if --ts-dir is provided, run batch; else single."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in {"single", "batch"}:
        mode, argv = argv[0], argv[1:]
    else:
        mode = "batch" if any(arg == "--ts-dir" or arg.startswith("--ts-dir=") for arg in argv) else "single"
    return main_batch(argv) if mode == "batch" else main_single(argv)


if __name__ == "__main__":
    main()
