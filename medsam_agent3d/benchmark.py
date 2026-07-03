"""
Evaluation benchmark: compare MedSAM-Agent3D against baselines.

Supports:
1. MedSAM-Agent3D (agentic, MLLM-driven)
2. SAM-Med3D direct (single bbox or prompt, no agent)
3. Integration points for other 3D frameworks (nnUNet, TotalSegmentator, etc.)

Output: Per-method metrics table, per-organ breakdown, significance tests.

Usage:
    python -m medsam_agent3d.benchmark \
        --method agent \
        --ts-dir Totalsegmentator_dataset_v201 \
        --model Qwen/Qwen3-VL-8B-Instruct \
        --max-samples 10 \
        --output-dir benchmark_results
"""

import json
import os
import sys
import time
import argparse
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np


@dataclass
class BenchmarkEntry:
    """A single evaluation case in the benchmark."""
    volume_path: str
    mask_path: str
    target_name: str
    volume_id: str
    volume: Optional[np.ndarray] = None
    gt_mask: Optional[np.ndarray] = None
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass
class MethodResult:
    """Result of running one method on one case."""
    method: str
    volume_id: str
    target_name: str
    iou: float = 0.0
    dice: float = 0.0
    runtime_sec: float = 0.0
    num_turns: int = 0
    success: bool = False
    error: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    """Aggregated benchmark results."""
    methods: List[str]
    total_cases: int
    results: List[MethodResult] = field(default_factory=list)
    per_method: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    per_organ: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)

    def add_result(self, result: MethodResult):
        self.results.append(result)

    def compute(self):
        """Aggregate results into per-method and per-organ statistics."""
        self.per_method = {}
        self.per_organ = {}

        # Group by method
        by_method: Dict[str, List[MethodResult]] = defaultdict(list)
        for r in self.results:
            if r.success:
                by_method[r.method].append(r)

        for method, results in by_method.items():
            ious = [r.iou for r in results]
            dices = [r.dice for r in results]
            times = [r.runtime_sec for r in results]
            turns = [r.num_turns for r in results if r.num_turns > 0]

            self.per_method[method] = {
                "num_cases": len(results),
                "mean_iou": float(np.mean(ious)),
                "std_iou": float(np.std(ious)),
                "mean_dice": float(np.mean(dices)),
                "std_dice": float(np.std(dices)),
                "mean_runtime_sec": float(np.mean(times)),
                "mean_turns": float(np.mean(turns)) if turns else 0.0,
            }

        # Group by organ + method
        by_organ_method: Dict[str, Dict[str, List[MethodResult]]] = \
            defaultdict(lambda: defaultdict(list))
        for r in self.results:
            if r.success:
                by_organ_method[r.target_name][r.method].append(r)

        for organ, methods in by_organ_method.items():
            self.per_organ[organ] = {}
            for method, results in methods.items():
                ious = [r.iou for r in results]
                self.per_organ[organ][method] = {
                    "num_cases": len(results),
                    "mean_iou": float(np.mean(ious)),
                    "std_iou": float(np.std(ious)),
                }

    def summary_table(self) -> str:
        """Generate a formatted summary table."""
        if not self.per_method:
            self.compute()

        lines = []
        lines.append("=" * 90)
        lines.append("BENCHMARK SUMMARY")
        lines.append("=" * 90)
        lines.append(f"Total cases: {self.total_cases}")
        lines.append(f"Methods: {', '.join(self.methods)}")
        lines.append("")

        # Header
        lines.append(f"{'Method':<25} {'Cases':>6} {'IoU':>8} {'IoU_std':>8} "
                     f"{'Dice':>8} {'Dice_std':>8} {'Time(s)':>8} {'Turns':>6}")
        lines.append("-" * 90)

        # One row per method
        for method in self.methods:
            m = self.per_method.get(method, {})
            if m:
                lines.append(
                    f"{method:<25} {m.get('num_cases', 0):>6} "
                    f"{m.get('mean_iou', 0):>8.4f} {m.get('std_iou', 0):>8.4f} "
                    f"{m.get('mean_dice', 0):>8.4f} {m.get('std_dice', 0):>8.4f} "
                    f"{m.get('mean_runtime_sec', 0):>8.1f} {m.get('mean_turns', 0):>6.1f}"
                )

        lines.append("-" * 90)
        lines.append("")

        # Per-organ breakdown
        if self.per_organ:
            organs = sorted(self.per_organ.keys())
            lines.append("Per-Organ IoU Breakdown:")
            lines.append("-" * 60)

            # Header
            header = f"{'Organ':<20}"
            for method in self.methods:
                header += f" {method:<12}"
            lines.append(header)

            for organ in organs:
                row = f"{organ:<20}"
                for method in self.methods:
                    om = self.per_organ[organ].get(method, {})
                    row += f" {om.get('mean_iou', 0):>11.4f}" if om else f" {'N/A':>12}"
                lines.append(row)

            lines.append("-" * 60)

        return "\n".join(lines)

    def to_json(self, path: str):
        """Save benchmark results as JSON."""
        data = {
            "methods": self.methods,
            "total_cases": self.total_cases,
            "per_method": self.per_method,
            "per_organ": self.per_organ,
            "results": [
                {
                    "method": r.method,
                    "volume_id": r.volume_id,
                    "target_name": r.target_name,
                    "iou": r.iou,
                    "dice": r.dice,
                    "runtime_sec": r.runtime_sec,
                    "num_turns": r.num_turns,
                    "success": r.success,
                    "error": r.error,
                }
                for r in self.results
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Benchmark report saved to {path}")


# =========================================================================
# Method runners
# =========================================================================

def run_agent_method(entry: BenchmarkEntry, args) -> MethodResult:
    """Run MedSAM-Agent3D (our agentic method)."""
    from medsam_agent3d.agent import QwenVLAgent
    from medsam_agent3d.loop import run_agent_loop
    from models.sam_med3d_wrapper import FrozenSAMMed3D

    t0 = time.time()

    try:
        # Lazy-load models
        if not hasattr(run_agent_method, "_mllm"):
            run_agent_method._mllm = QwenVLAgent(
                args.model,
                device_map=getattr(args, "device", "auto"),
            )
        if not hasattr(run_agent_method, "_sam"):
            ckpt = getattr(args, "sam_ckpt", None)
            run_agent_method._sam = FrozenSAMMed3D(
                checkpoint_path=ckpt
            ) if ckpt else FrozenSAMMed3D()

        result = run_agent_loop(
            mllm_agent=run_agent_method._mllm,
            sam_model=run_agent_method._sam,
            volume=entry.volume,
            gt_mask=entry.gt_mask,
            target_name=entry.target_name,
            volume_id=entry.volume_id,
            spacing=entry.spacing,
            max_turns=args.max_turns,
            output_dir=os.path.join(args.output_dir, "agent", entry.volume_id,
                                    entry.target_name),
            temperature=args.temperature,
            seed=args.seed,
        )

        runtime = time.time() - t0
        return MethodResult(
            method="MedSAM-Agent3D",
            volume_id=entry.volume_id,
            target_name=entry.target_name,
            iou=result.final_iou,
            dice=result.final_dice,
            runtime_sec=runtime,
            num_turns=len(result.turns),
            success=result.success,
            error=result.error,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return MethodResult(
            method="MedSAM-Agent3D",
            volume_id=entry.volume_id,
            target_name=entry.target_name,
            error=str(e),
        )


def run_sam_direct_bbox(entry: BenchmarkEntry, args) -> MethodResult:
    """Run SAM-Med3D with a single center bounding box (no agent)."""
    from models.sam_med3d_wrapper import FrozenSAMMed3D

    t0 = time.time()

    try:
        if not hasattr(run_sam_direct_bbox, "_sam"):
            ckpt = getattr(args, "sam_ckpt", None)
            run_sam_direct_bbox._sam = FrozenSAMMed3D(
                checkpoint_path=ckpt
            ) if ckpt else FrozenSAMMed3D()

        vol = entry.volume
        dz, dy, dx = vol.shape

        # Simple heuristic: bbox covers middle 60% of volume
        z1, z2 = int(0.2 * dz), int(0.8 * dz)
        y1, y2 = int(0.2 * dy), int(0.8 * dy)
        x1, x2 = int(0.2 * dx), int(0.8 * dx)
        bbox = (z1, y1, x1, z2, y2, x2)

        run_sam_direct_bbox._sam.set_image(vol)
        pred_mask, _ = run_sam_direct_bbox._sam.predict_mask(
            points=None, labels=None,
            box_3d=bbox,
            mask_logits=None,
        )

        from medsam_agent3d.types import compute_metrics
        iou, dice = compute_metrics(pred_mask, entry.gt_mask)
        runtime = time.time() - t0

        return MethodResult(
            method="SAM-Med3D (bbox)",
            volume_id=entry.volume_id,
            target_name=entry.target_name,
            iou=iou,
            dice=dice,
            runtime_sec=runtime,
            num_turns=1,
            success=True,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return MethodResult(
            method="SAM-Med3D (bbox)",
            volume_id=entry.volume_id,
            target_name=entry.target_name,
            error=str(e),
        )


def run_sam_direct_point(entry: BenchmarkEntry, args) -> MethodResult:
    """Run SAM-Med3D with a single center point (no agent)."""
    from models.sam_med3d_wrapper import FrozenSAMMed3D

    t0 = time.time()

    try:
        if not hasattr(run_sam_direct_point, "_sam"):
            ckpt = getattr(args, "sam_ckpt", None)
            run_sam_direct_point._sam = FrozenSAMMed3D(
                checkpoint_path=ckpt
            ) if ckpt else FrozenSAMMed3D()

        vol = entry.volume
        center = (vol.shape[0] // 2, vol.shape[1] // 2, vol.shape[2] // 2)

        run_sam_direct_point._sam.set_image(vol)
        pred_mask, _ = run_sam_direct_point._sam.predict_mask(
            points=[center], labels=[1],
            box_3d=None,
            mask_logits=None,
        )

        from medsam_agent3d.types import compute_metrics
        iou, dice = compute_metrics(pred_mask, entry.gt_mask)
        runtime = time.time() - t0

        return MethodResult(
            method="SAM-Med3D (point)",
            volume_id=entry.volume_id,
            target_name=entry.target_name,
            iou=iou,
            dice=dice,
            runtime_sec=runtime,
            num_turns=1,
            success=True,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return MethodResult(
            method="SAM-Med3D (point)",
            volume_id=entry.volume_id,
            target_name=entry.target_name,
            error=str(e),
        )


# =========================================================================
# Method registry
# =========================================================================

METHOD_REGISTRY = {
    "agent": run_agent_method,
    "sam-bbox": run_sam_direct_bbox,
    "sam-point": run_sam_direct_point,
    # Add more methods here:
    # "nnunet": run_nnunet,
    # "totalsegmentator": run_total_segmentator,
    # "sam-med3d-medsam-agent": run_medsam_agent_3d,
}


# =========================================================================
# Benchmark orchestrator
# =========================================================================

def run_benchmark(
    methods: List[str],
    entries: List[BenchmarkEntry],
    args,
) -> BenchmarkReport:
    """Run multiple methods against the same set of evaluation cases.

    Args:
        methods: List of method keys from METHOD_REGISTRY.
        entries: List of BenchmarkEntry objects (volumes + masks).
        args: argparse namespace with model/config options.

    Returns:
        BenchmarkReport with per-method and per-case results.
    """
    # Check method validity
    unknown = [m for m in methods if m not in METHOD_REGISTRY]
    if unknown:
        print(f"Unknown methods: {unknown}")
        print(f"Available: {list(METHOD_REGISTRY.keys())}")
        sys.exit(1)

    report = BenchmarkReport(
        methods=methods,
        total_cases=len(entries),
    )

    for i, entry in enumerate(entries):
        print(f"\n{'='*60}")
        print(f"Case [{i+1}/{len(entries)}]: {entry.volume_id}/{entry.target_name}")
        print(f"{'='*60}")

        for method_key in methods:
            method_fn = METHOD_REGISTRY[method_key]
            print(f"\n  Running {method_key}...")
            result = method_fn(entry, args)
            report.add_result(result)

            if result.success:
                print(f"    IoU={result.iou:.4f}, Dice={result.dice:.4f}, "
                      f"Time={result.runtime_sec:.1f}s, Turns={result.num_turns}")
            else:
                print(f"    FAILED: {result.error}")

    # Compute aggregate stats
    report.compute()
    return report


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MedSAM-Agent3D Benchmark: compare methods on 3D segmentation"
    )

    # Method selection
    parser.add_argument(
        "--method", "-m", action="append", default=[],
        choices=list(METHOD_REGISTRY.keys()) + ["all"],
        help="Methods to benchmark (repeatable). Use 'all' for all available.",
    )

    # Data
    parser.add_argument(
        "--ts-dir", default=None,
        help="Path to TotalSegmentator dataset directory",
    )
    parser.add_argument(
        "--volume", action="append", default=[],
        help="Single volume path (repeatable for multiple)",
    )
    parser.add_argument(
        "--mask", action="append", default=[],
        help="Single mask path (repeatable, paired with --volume)",
    )
    parser.add_argument(
        "--target", action="append", default=[],
        help="Target name per volume (repeatable, paired with --volume)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=10,
        help="Maximum samples when using --ts-dir",
    )
    parser.add_argument("--shuffle", action="store_true", default=True)

    # Model config
    parser.add_argument(
        "--model", default="Qwen/Qwen3-VL-8B-Instruct",
        help="MLLM model ID (for agent method)",
    )
    parser.add_argument(
        "--sam-ckpt", default=None,
        help="Path to SAM-Med3D checkpoint",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--load-in-4bit", action="store_true")

    # Loop config
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="benchmark_results")

    args = parser.parse_args()

    # --- Resolve methods ---
    if "all" in args.method or not args.method:
        methods = list(METHOD_REGISTRY.keys())
    else:
        methods = args.method

    print(f"Benchmark methods: {methods}")

    # --- Resolve dataset ---
    entries: List[BenchmarkEntry] = []

    if args.ts_dir:
        # TotalSegmentator dataset
        from utils.io_utils import get_totalsegmentator_dataset, load_nifti

        dataset = get_totalsegmentator_dataset(args.ts_dir)
        if args.shuffle:
            import random
            random.shuffle(dataset)
        dataset = dataset[:args.max_samples]

        for d in dataset:
            vol, spacing = load_nifti(d["ct_path"])
            mask, _ = load_nifti(d["mask_path"])
            mask = (mask > 0).astype(np.uint8)
            if mask.sum() == 0:
                continue
            entries.append(BenchmarkEntry(
                volume_path=d["ct_path"],
                mask_path=d["mask_path"],
                target_name=d["target_name"],
                volume_id=d["volume_id"],
                volume=vol,
                gt_mask=mask,
                spacing=spacing,
            ))

    elif args.volume:
        # User-provided volumes
        from utils.io_utils import load_nifti

        masks = args.mask or [None] * len(args.volume)
        targets = args.target or ["organ"] * len(args.volume)

        for i, vol_path in enumerate(args.volume):
            vol, spacing = load_nifti(vol_path)
            mask, _ = load_nifti(masks[i]) if masks[i] else (None, None)
            if mask is not None:
                mask = (mask > 0).astype(np.uint8)
            vol_id = os.path.splitext(os.path.basename(vol_path))[0]
            vol_id = vol_id.replace(".nii", "").replace(".gz", "")
            entries.append(BenchmarkEntry(
                volume_path=vol_path,
                mask_path=masks[i] if masks else "",
                target_name=targets[i],
                volume_id=vol_id,
                volume=vol,
                gt_mask=mask,
                spacing=spacing,
            ))

    if not entries:
        print("ERROR: No entries found. Provide --ts-dir or --volume/--mask.")
        sys.exit(1)

    print(f"Evaluation cases: {len(entries)}")
    for e in entries:
        print(f"  {e.volume_id}/{e.target_name}  vol={e.volume.shape}  "
              f"mask_voxels={int(e.gt_mask.sum()):,}" if e.gt_mask is not None else "")

    # --- Run benchmark ---
    os.makedirs(args.output_dir, exist_ok=True)
    report = run_benchmark(methods, entries, args)

    # --- Print results ---
    print("\n" + report.summary_table())

    # --- Save ---
    report.to_json(os.path.join(args.output_dir, "benchmark_report.json"))

    return report


if __name__ == "__main__":
    main()
