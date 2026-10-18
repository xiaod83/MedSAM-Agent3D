"""Evaluation, complete trajectory persistence, and failure-aware summaries."""
import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import numpy as np
from .types import AgentResult
from .agent import MLLMAgent, QwenVLAgent
from .loop import run_agent_loop


def checkpoint_provenance(path=None):
    """Fingerprint the actual SAM weights, caching by file size and modification time."""
    root = os.path.dirname(os.path.dirname(__file__))
    default = os.path.join(root, "checkpoints", "sam_med3d_turbo.pth")
    if not os.path.isfile(default):
        default = os.path.join(root, "sam_med3d_turbo.pth")
    path = os.path.abspath(path or default)
    if not os.path.isfile(path):
        return {"path": path, "exists": False}
    try:
        stat = os.stat(path)
        return _checkpoint_fingerprint(path, stat.st_size, stat.st_mtime_ns)
    except OSError as exc:
        # A provenance failure must not prevent saving the original model error.
        return {"path": path, "exists": True, "readable": False,
                "error": f"{type(exc).__name__}: {exc}"}


from functools import lru_cache


@lru_cache(maxsize=8)
def _checkpoint_fingerprint(path, size, mtime):
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": path, "size_bytes": size, "sha256": digest.hexdigest()}

def runtime_provenance():
    """Capture reproducibility metadata without importing optional dependencies."""
    import importlib.metadata
    import platform
    import subprocess
    versions = {}
    for name in ("numpy", "torch", "torchio", "nibabel", "scipy", "pillow", "transformers", "qwen-vl-utils"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        git_revision = subprocess.check_output(["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stderr=subprocess.DEVNULL, text=True, timeout=5).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stderr=subprocess.DEVNULL, text=True, timeout=5).strip())
    except (OSError, subprocess.SubprocessError):
        git_revision, dirty = None, None
    return {"python": platform.python_version(), "package_versions": versions,
            "git_revision": git_revision, "git_dirty": dirty}

def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def result_dict(result):
    # Avoid asdict(result): it deep-copies the potentially large voxel array.
    return {key: ( [asdict(t) for t in value] if key == "turns" else value)
            for key, value in vars(result).items() if key != "predicted_mask"}


def save_case_result(result, output_dir, reference_path=None):
    """Save the full trajectory and a mask in the source NIfTI geometry."""
    os.makedirs(output_dir, exist_ok=True)
    result.config.setdefault('environment', runtime_provenance())
    result.config.setdefault('sam_weights', checkpoint_provenance(result.config.get('sam_checkpoint')))
    if result.predicted_mask is not None:
        if reference_path:
            from medsam_agent3d.io import save_nifti
            result.mask_path = os.path.join(output_dir, "prediction.nii.gz")
            save_nifti(result.predicted_mask, result.mask_path, reference_path=reference_path)
        else:
            result.mask_path = os.path.join(output_dir, "prediction.npy")
            np.save(result.mask_path, result.predicted_mask.astype(np.uint8))
    with open(os.path.join(output_dir, "trajectory.json"), "w", encoding="utf-8") as stream:
        json.dump(result_dict(result), stream, indent=2, default=_json_default, allow_nan=False)


def evaluate_single(model_id="Qwen/Qwen3-VL-8B-Instruct", volume_path=None,
                    mask_path=None, target_name="organ", sam_ckpt=None,
                    output_dir="outputs/inference", max_turns=5, temperature=0.1,
                    seed=42, sam_device="auto", **model_kwargs):
    from medsam_agent3d.sam import FrozenSAMMed3D
    from medsam_agent3d.io import load_nifti
    if not volume_path:
        raise ValueError("volume_path is required")
    vol, spacing, affine = load_nifti(volume_path, return_affine=True)
    mask = None
    if mask_path:
        mask, mask_spacing, mask_affine = load_nifti(mask_path, return_affine=True, reference_path=volume_path, is_label=True)
        if mask.shape != vol.shape or not np.allclose(mask_affine, affine):
            raise ValueError("Ground truth must have the same voxel grid and affine as the volume")
        mask = (mask > 0).astype(np.uint8)
    from .cli import seed_everything
    seed_everything(seed)
    mllm = QwenVLAgent(model_id, **model_kwargs)
    sam = FrozenSAMMed3D(checkpoint_path=sam_ckpt, device=sam_device) if sam_ckpt else FrozenSAMMed3D(device=sam_device)
    result = run_agent_loop(mllm_agent=mllm, sam_model=sam, volume=vol,
        gt_mask=mask, target_name=target_name,
        volume_id=os.path.basename(volume_path).replace(".nii.gz", "").replace(".nii", ""),
        spacing=spacing, affine=affine, max_turns=max_turns,
        output_dir=output_dir, temperature=temperature, seed=seed)
    result.config.update(model_id=model_id, sam_checkpoint=sam_ckpt, sam_device=str(sam.device),
        volume_path=os.path.abspath(volume_path), mask_path=mask_path,
        max_turns=max_turns, temperature=temperature, seed=seed,
        model_kwargs={k: str(v) for k, v in model_kwargs.items()})
    save_case_result(result, output_dir, volume_path)
    _print_summary([result])
    _save_results_json([result], output_dir)
    return result


def evaluate_dataset(mllm_agent: MLLMAgent, sam_model, dataset_entries: List[Dict[str, Any]],
                     output_dir="outputs/inference", max_turns=5, temperature=0.1, seed=42):
    from medsam_agent3d.io import load_nifti
    results = []
    for entry in dataset_entries:
        vol_id, target = entry["volume_id"], entry["target_name"]
        reference_path = entry.get("volume_path") or entry.get("ct_path")
        case_dir = os.path.join(output_dir, vol_id, target)
        try:
            volume, spacing, affine = entry.get("volume"), entry.get("spacing", (1., 1., 1.)), entry.get("affine")
            if volume is None:
                if not reference_path:
                    raise ValueError("Entry requires volume, volume_path, or ct_path")
                volume, spacing, affine = load_nifti(reference_path, return_affine=True)
            mask = entry.get("mask")
            if mask is None and entry.get("mask_path"):
                mask, _, mask_affine = load_nifti(entry["mask_path"], return_affine=True, reference_path=reference_path, is_label=True)
                if affine is not None and not np.allclose(affine, mask_affine):
                    raise ValueError("Ground truth affine differs from volume")
            if mask is not None:
                if mask.shape != volume.shape:
                    raise ValueError("Ground truth shape differs from volume")
                mask = (mask > 0).astype(np.uint8)
            result = run_agent_loop(mllm_agent=mllm_agent, sam_model=sam_model,
                volume=volume, gt_mask=mask, target_name=target, volume_id=vol_id,
                spacing=spacing, affine=affine, max_turns=max_turns,
                output_dir=case_dir, temperature=temperature, seed=seed)
        except Exception as exc:
            result = AgentResult(volume_id=vol_id, target_name=target, error=str(exc), termination_reason="error")
        if not result.success and result.final_iou is None and (entry.get("mask") is not None or entry.get("mask_path")):
            result.final_iou, result.final_dice = 0.0, 0.0
            result.metrics_available = True
        result.config.update(volume_path=reference_path, mask_path=entry.get("mask_path"),
                             max_turns=max_turns, temperature=temperature, seed=seed,
                             sam_checkpoint=getattr(sam_model, 'checkpoint_path', None),
                             sam_device=str(getattr(sam_model, 'device', 'unknown')),
                             model_id=getattr(mllm_agent, 'model_id', type(mllm_agent).__name__))
        # Persistence failures should be visible rather than losing the entire batch.
        try:
            save_case_result(result, case_dir, reference_path)
        except Exception as exc:
            result.success = False
            result.error = (result.error + "; " if result.error else "") + "Saving failed: " + str(exc)
        results.append(result)
    _print_summary(results)
    _save_results_json(results, output_dir)
    return results


def _print_summary(results):
    successful = [r for r in results if r.success]
    scored = [r for r in results if r.final_iou is not None and r.final_dice is not None]
    print(f"Cases: {len(results)}, successful: {len(successful)}, failed: {len(results)-len(successful)}, scored: {len(scored)}")
    if scored:
        # Failed cases contribute zero to the primary score, preserving the denominator.
        ious = [r.final_iou if r.success else 0. for r in scored]
        dices = [r.final_dice if r.success else 0. for r in scored]
        print(f"IoU (failure=0): {np.mean(ious):.4f}; Dice (failure=0): {np.mean(dices):.4f}")


def _save_results_json(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "evaluation_summary.json"), "w", encoding="utf-8") as stream:
        json.dump([result_dict(r) for r in results], stream, indent=2, default=_json_default, allow_nan=False)
