"""
Evaluation benchmark: compare MedSAM-Agent3D against baselines.

Supports:
1. MedSAM-Agent3D (agentic, MLLM-driven)
2. SAM-Med3D direct (single bbox or prompt, no agent)
3. Integration points for other 3D frameworks (nnUNet, TotalSegmentator, etc.)

Output: failure-aware metrics, matched-case summaries, and per-case trajectories.

Usage:
    python -m medsam_agent3d.benchmark \
        --method agent \
        --ts-dir Totalsegmentator_dataset_v201 \
        --model Qwen/Qwen3-VL-8B-Instruct \
        --max-samples 10 \
        --output-dir outputs/benchmarks
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
    affine: Optional[np.ndarray] = None


@dataclass
class MethodResult:
    """Result of running one method on one case."""
    method: str
    volume_id: str
    target_name: str
    iou: Optional[float] = None
    dice: Optional[float] = None
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
    config: Dict[str, Any] = field(default_factory=dict)

    def add_result(self, result: MethodResult):
        self.results.append(result)

    @staticmethod
    def _stats(results):
        scored = [r for r in results if r.iou is not None and r.dice is not None]
        ious = [r.iou if r.success else 0.0 for r in scored]
        dices = [r.dice if r.success else 0.0 for r in scored]
        return {
            "num_cases": len(results), "num_successful": sum(r.success for r in results),
            "num_failed": sum(not r.success for r in results), "num_scored": len(scored),
            "failure_rate": sum(not r.success for r in results) / len(results) if results else 0.,
            "mean_iou": float(np.mean(ious)) if ious else 0.,
            "std_iou": float(np.std(ious)) if ious else 0.,
            "mean_dice": float(np.mean(dices)) if dices else 0.,
            "std_dice": float(np.std(dices)) if dices else 0.,
            "mean_runtime_sec": float(np.mean([r.runtime_sec for r in results])) if results else 0.,
            "mean_turns": float(np.mean([r.num_turns for r in results])) if results else 0.,
        }

    def compute(self):
        """Primary metrics count failed scored cases as zero; retain all denominators."""
        self.per_method = {method: self._stats([r for r in self.results if r.method == method])
                           for method in self.methods}
        self.per_organ = {
            organ: {method: self._stats([r for r in self.results
                                        if r.method == method and r.target_name == organ])
                    for method in self.methods}
            for organ in sorted({r.target_name for r in self.results})}
        matched = set.intersection(*[
            {(r.volume_id, r.target_name) for r in self.results
             if r.method == method and r.success and r.iou is not None}
            for method in self.methods]) if self.methods else set()
        for method in self.methods:
            values = [r.iou for r in self.results if r.method == method
                      and (r.volume_id, r.target_name) in matched]
            self.per_method[method]["matched_successful_cases"] = len(values)
            self.per_method[method]["matched_mean_iou"] = float(np.mean(values)) if values else None
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
        lines.append(f"{'Method':<25} {'Cases':>6} {'Fail':>6} {'Scored':>6} {'IoU':>8} {'IoU_std':>8} "
                     f"{'Dice':>8} {'Dice_std':>8} {'Time(s)':>8} {'Turns':>6}")
        lines.append("-" * 90)

        # One row per method
        for method in self.methods:
            m = self.per_method.get(method, {})
            if m:
                lines.append(
                    f"{method:<25} {m.get('num_cases', 0):>6} {m.get('num_failed', 0):>6} {m.get('num_scored', 0):>6} "
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
            "config": self.config,
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
                    "extra": r.extra,
                }
                for r in self.results
            ],
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Benchmark report saved to {path}")


# =========================================================================
# Method runners
# =========================================================================

def _save_prediction(prediction, entry, args, method):
    from medsam_agent3d.io import save_nifti
    path = os.path.join(args.output_dir, method, entry.volume_id, entry.target_name, "prediction.nii.gz")
    save_nifti(prediction.astype(np.uint8), path, affine=entry.affine, reference_path=entry.volume_path)
    return path

def _model_kwargs(args):
    from .cli import _build_model_kwargs
    return _build_model_kwargs(args)


def _shared_sam(args):
    if not hasattr(args, "_sam"):
        from medsam_agent3d.sam import FrozenSAMMed3D
        ckpt = getattr(args, "sam_ckpt", None)
        args._sam = FrozenSAMMed3D(checkpoint_path=ckpt, device=getattr(args, "sam_device", "auto")) if ckpt else FrozenSAMMed3D(device=getattr(args, "sam_device", "auto"))
    return args._sam

def run_agent_method(entry: BenchmarkEntry, args) -> MethodResult:
    """Run MedSAM-Agent3D (our agentic method)."""
    from medsam_agent3d.agent import QwenVLAgent
    from medsam_agent3d.loop import run_agent_loop
    from medsam_agent3d.sam import FrozenSAMMed3D

    t0 = time.time()

    try:
        # Model initialization is excluded from per-case runtime.
        if not hasattr(args, "_mllm"):
            args._mllm = QwenVLAgent(
                args.model,
                **_model_kwargs(args),
            )
        sam = _shared_sam(args)
        t0 = time.time()

        result = run_agent_loop(
            mllm_agent=args._mllm,
            sam_model=sam,
            volume=entry.volume,
            gt_mask=entry.gt_mask,
            target_name=entry.target_name,
            volume_id=entry.volume_id,
            spacing=entry.spacing,
            affine=entry.affine,
            max_turns=args.max_turns,
            output_dir=os.path.join(args.output_dir, "agent", entry.volume_id,
                                    entry.target_name),
            temperature=args.temperature,
            seed=args.seed,
        )

        runtime = time.time() - t0
        result.config.update(model_id=args.model, sam_checkpoint=getattr(args, "sam_ckpt", None),
                             load_in_4bit=getattr(args, 'load_in_4bit', False))
        from .evaluation import save_case_result
        save_case_result(result, os.path.join(args.output_dir, 'agent', entry.volume_id, entry.target_name), entry.volume_path)
        return MethodResult(
            method="agent",
            volume_id=entry.volume_id,
            target_name=entry.target_name,
            iou=result.final_iou,
            dice=result.final_dice,
            runtime_sec=runtime,
            num_turns=len(result.turns),
            success=result.success,
            error=result.error,
            extra={'trajectory_path': os.path.join(args.output_dir, 'agent', entry.volume_id, entry.target_name, 'trajectory.json'), 'mask_path': result.mask_path},
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return MethodResult(
            method="agent",
            volume_id=entry.volume_id,
            target_name=entry.target_name,
            error=str(e),
        )


def run_sam_direct_bbox(entry: BenchmarkEntry, args) -> MethodResult:
    """Central-volume bbox heuristic; wrapper uses a trained center-point fallback."""
    from medsam_agent3d.sam import FrozenSAMMed3D

    t0 = time.time()

    try:
        sam = _shared_sam(args)
        t0 = time.time()

        vol = entry.volume
        dz, dy, dx = vol.shape

        # Simple heuristic: bbox covers middle 60% of volume
        z1, z2 = int(0.2 * dz), int(0.8 * dz)
        y1, y2 = int(0.2 * dy), int(0.8 * dy)
        x1, x2 = int(0.2 * dx), int(0.8 * dx)
        bbox = (z1, y1, x1, z2, y2, x2)

        sam.set_image(vol, spacing=entry.spacing, affine=entry.affine)
        pred_mask, _ = sam.predict_mask(
            points=None, labels=None,
            box_3d=bbox,
            mask_logits=None,
        )

        from medsam_agent3d.types import compute_metrics
        iou, dice = compute_metrics(pred_mask, entry.gt_mask) if entry.gt_mask is not None else (None, None)
        runtime = time.time() - t0

        return MethodResult(
            method="sam-bbox",
            volume_id=entry.volume_id,
            target_name=entry.target_name,
            iou=iou,
            dice=dice,
            runtime_sec=runtime,
            num_turns=1,
            success=bool(np.any(pred_mask)),
            error='' if np.any(pred_mask) else 'Empty segmentation',
            extra={'mask_path': _save_prediction(pred_mask, entry, args, 'sam-bbox'), 'prompt_policy': 'central_60_percent_box_with_center_point_fallback', 'uses_ground_truth_prompts': False},
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return MethodResult(
            method="sam-bbox",
            volume_id=entry.volume_id,
            target_name=entry.target_name,
            error=str(e),
        )


def run_sam_direct_point(entry: BenchmarkEntry, args) -> MethodResult:
    """Run SAM-Med3D with a single center point (no agent)."""
    from medsam_agent3d.sam import FrozenSAMMed3D

    t0 = time.time()

    try:
        sam = _shared_sam(args)
        t0 = time.time()

        vol = entry.volume
        center = (vol.shape[0] // 2, vol.shape[1] // 2, vol.shape[2] // 2)

        sam.set_image(vol, spacing=entry.spacing, affine=entry.affine)
        pred_mask, _ = sam.predict_mask(
            points=[center], labels=[1],
            box_3d=None,
            mask_logits=None,
        )

        from medsam_agent3d.types import compute_metrics
        iou, dice = compute_metrics(pred_mask, entry.gt_mask) if entry.gt_mask is not None else (None, None)
        runtime = time.time() - t0

        return MethodResult(
            method="sam-point",
            volume_id=entry.volume_id,
            target_name=entry.target_name,
            iou=iou,
            dice=dice,
            runtime_sec=runtime,
            num_turns=1,
            success=bool(np.any(pred_mask)),
            error='' if np.any(pred_mask) else 'Empty segmentation',
            extra={'mask_path': _save_prediction(pred_mask, entry, args, 'sam-point'), 'prompt_policy': 'volume_center_point', 'uses_ground_truth_prompts': False},
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return MethodResult(
            method="sam-point",
            volume_id=entry.volume_id,
            target_name=entry.target_name,
            error=str(e),
        )


# =========================================================================
# Method registry
# =========================================================================

def run_sam_oracle(entry, args):
    """Ground-truth error clicks: an explicitly privileged refinement baseline."""
    from scipy.ndimage import distance_transform_edt
    from .types import compute_metrics
    start = time.time()
    try:
        if entry.gt_mask is None or not np.any(entry.gt_mask):
            raise ValueError("Oracle clicks require a nonempty ground truth mask")
        sam = _shared_sam(args)
        start = time.time()
        sam.set_image(entry.volume, spacing=entry.spacing, affine=entry.affine)
        prediction = np.zeros(entry.volume.shape, dtype=bool)
        points, labels, trajectory, logits = [], [], [], None
        for turn in range(args.max_turns):
            missing = np.logical_and(entry.gt_mask, ~prediction.astype(bool))
            excess = np.logical_and(~entry.gt_mask.astype(bool), prediction)
            # Padding ensures distance is measured to background at every edge.
            positive = distance_transform_edt(np.pad(missing, 1), sampling=entry.spacing)[1:-1, 1:-1, 1:-1]
            negative = distance_transform_edt(np.pad(excess, 1), sampling=entry.spacing)[1:-1, 1:-1, 1:-1]
            label = int(positive.max() >= negative.max())
            distance = positive if label else negative
            if not distance.max():
                break
            point = tuple(int(v) for v in np.unravel_index(distance.argmax(), distance.shape))
            points.append(point)
            labels.append(label)
            prediction, logits = sam.predict_mask(points=points, labels=labels, box_3d=None, mask_logits=logits)
            iou, dice = compute_metrics(prediction, entry.gt_mask)
            trajectory.append({"turn": turn + 1, "point": point, "label": label, "iou": iou, "dice": dice})
        iou, dice = compute_metrics(prediction, entry.gt_mask)
        return MethodResult(method="sam-oracle", volume_id=entry.volume_id, target_name=entry.target_name,
            iou=iou, dice=dice, runtime_sec=time.time()-start, num_turns=len(trajectory), success=True,
            extra={"uses_ground_truth_prompts": True, "turns": trajectory,
                   "mask_path": _save_prediction(prediction, entry, args, "sam-oracle")})
    except Exception as exc:
        return MethodResult(method="sam-oracle", volume_id=entry.volume_id,
                            target_name=entry.target_name, error=str(exc), runtime_sec=time.time()-start)

METHOD_REGISTRY = {
    "agent": run_agent_method,
    "sam-bbox": run_sam_direct_bbox,
    "sam-point": run_sam_direct_point,
    "sam-oracle": run_sam_oracle,
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

    from .evaluation import runtime_provenance, checkpoint_provenance
    report = BenchmarkReport(
        methods=methods,
        total_cases=len(entries),
        config={key: value for key, value in vars(args).items() if not key.startswith('_')},
    )

    report.config["environment"] = runtime_provenance()
    report.config['sam_weights'] = checkpoint_provenance(getattr(args, 'sam_ckpt', None))
    for i, entry in enumerate(entries):
        print(f"\n{'='*60}")
        print(f"Case [{i+1}/{len(entries)}]: {entry.volume_id}/{entry.target_name}")
        print(f"{'='*60}")

        loaded_here = entry.volume is None
        load_error = ""
        if loaded_here:
            try:
                from medsam_agent3d.io import load_nifti
                entry.volume, entry.spacing, entry.affine = load_nifti(entry.volume_path, return_affine=True)
                if entry.mask_path:
                    entry.gt_mask, _, _ = load_nifti(entry.mask_path, return_affine=True,
                        reference_path=entry.volume_path, is_label=True)
                    entry.gt_mask = (entry.gt_mask > 0).astype(np.uint8)
            except Exception as exc:
                load_error = str(exc)
        for method_key in methods:
            method_fn = METHOD_REGISTRY[method_key]
            print(f"\n  Running {method_key}...")
            from .cli import seed_everything
            seed_everything(args.seed)
            try:
                if load_error:
                    raise ValueError(load_error)
                result = method_fn(entry, args)
            except Exception as exc:
                result = MethodResult(method=method_key, volume_id=entry.volume_id,
                                      target_name=entry.target_name, error=str(exc))
            result.method = method_key
            if (entry.gt_mask is not None or entry.mask_path) and not result.success:
                result.iou, result.dice = 0.0, 0.0
            if hasattr(args, "_sam"):
                result.extra['sam_device'] = str(args._sam.device)
            report.add_result(result)

            if result.success and result.iou is not None:
                print(f"    IoU={result.iou:.4f}, Dice={result.dice:.4f}, "
                      f"Time={result.runtime_sec:.1f}s, Turns={result.num_turns}")
            elif not result.success:
                print(f"    FAILED: {result.error}")

        if loaded_here:
            entry.volume, entry.gt_mask = None, None

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
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--split', choices=['train', 'val', 'test', 'all'], default='test')

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
    parser.add_argument('--sam-device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument("--load-in-4bit", action="store_true")

    # Loop config
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/benchmarks")

    args = parser.parse_args()
    from .cli import seed_everything, _validate_args
    _validate_args(parser, args)
    seed_everything(args.seed)

    # --- Resolve methods ---
    if "all" in args.method or not args.method:
        methods = list(METHOD_REGISTRY.keys())
    else:
        methods = list(dict.fromkeys(args.method))

    print(f"Benchmark methods: {methods}")

    # --- Resolve dataset ---
    entries: List[BenchmarkEntry] = []

    if args.ts_dir:
        # TotalSegmentator dataset
        from medsam_agent3d.io import get_totalsegmentator_dataset, load_nifti

        dataset = get_totalsegmentator_dataset(args.ts_dir, split=None if args.split == "all" else args.split)
        if args.shuffle:
            import random
            random.shuffle(dataset)
        dataset = dataset[:args.max_samples]

        for d in dataset:
            entries.append(BenchmarkEntry(volume_path=d["volume_path"], mask_path=d["mask_path"],
                target_name=d["target_name"], volume_id=d["volume_id"]))
    elif args.volume:
        if args.mask and len(args.mask) != len(args.volume):
            parser.error("--mask count must match --volume count")
        if args.target and len(args.target) != len(args.volume):
            parser.error("--target count must match --volume count")
        masks = args.mask or [None] * len(args.volume)
        targets = args.target or ["organ"] * len(args.volume)
        for path, mask, target in zip(args.volume, masks, targets):
            entries.append(BenchmarkEntry(volume_path=path, mask_path=mask, target_name=target,
                volume_id=os.path.basename(path).replace(".nii.gz", "").replace(".nii", "")))
    if not entries:
        print("ERROR: No entries found. Provide --ts-dir or --volume/--mask.")
        sys.exit(1)

    print(f"Evaluation cases: {len(entries)}")
    for e in entries:
        print(f"  {e.volume_id}/{e.target_name}")

    # --- Run benchmark ---
    os.makedirs(args.output_dir, exist_ok=True)
    report = run_benchmark(methods, entries, args)

    # --- Print results ---
    print("\n" + report.summary_table())

    # --- Save ---
    report.to_json(os.path.join(args.output_dir, "benchmark_report.json"))

    if any(not result.success for result in report.results):
        raise SystemExit(1)
    return report


if __name__ == "__main__":
    main()
