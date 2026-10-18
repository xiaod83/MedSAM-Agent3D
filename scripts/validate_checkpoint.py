"""Exercise actual SAM weights, refinement, and NIfTI export on one real case.

This is a privileged checkpoint smoke test: ground truth chooses the foreground
click. It is not evidence of autonomous Qwen segmentation performance. Use
--agent to additionally exercise the real vision-language agent.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=(ROOT / 'checkpoints/sam_med3d_turbo.pth' if (ROOT / 'checkpoints/sam_med3d_turbo.pth').is_file() else ROOT / 'sam_med3d_turbo.pth'))
    parser.add_argument('--volume', type=Path, default=ROOT / 'data/Totalsegmentator_dataset_v201/s0000/ct.nii.gz')
    parser.add_argument('--mask', type=Path, default=ROOT / 'data/Totalsegmentator_dataset_v201/s0000/segmentations/femur_left.nii.gz')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/validation')
    parser.add_argument('--sam-device', choices=['auto', 'cpu', 'cuda'], default='cpu')
    parser.add_argument('--agent', action='store_true', help='Also run real Qwen; may download model weights')
    parser.add_argument('--model', default='Qwen/Qwen3-VL-8B-Instruct')
    parser.add_argument('--agent-device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--target', default='femur_left')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {'success': False, 'uses_ground_truth_prompts': True,
              'autonomous_agent_requested': args.agent,
              'checkpoint': str(args.checkpoint), 'volume': str(args.volume),
              'mask': str(args.mask), 'seed': args.seed, 'steps': []}
    started = time.monotonic()
    exit_code = 1
    try:
        # Check real bytes before any expensive import/model allocation.
        for path in (args.checkpoint, args.volume, args.mask):
            with path.open('rb') as stream:
                if not stream.read(1):
                    raise ValueError(f'File is empty: {path}')
        import numpy as np
        import torch
        from scipy.ndimage import distance_transform_edt
        from medsam_agent3d.sam import FrozenSAMMed3D
        from medsam_agent3d.io import load_nifti, save_nifti
        from medsam_agent3d.types import compute_metrics
        from medsam_agent3d.cli import seed_everything, _build_model_kwargs
        from medsam_agent3d.evaluation import checkpoint_provenance, runtime_provenance
        seed_everything(args.seed)
        report['environment'] = runtime_provenance()
        report['weights'] = checkpoint_provenance(str(args.checkpoint))
        volume, spacing, affine = load_nifti(str(args.volume), return_affine=True)
        truth, _ = load_nifti(str(args.mask), reference_path=str(args.volume), is_label=True)
        truth = truth > 0
        if not truth.any():
            raise ValueError('Checkpoint smoke test requires a nonempty reference organ')
        sam = FrozenSAMMed3D(checkpoint_path=str(args.checkpoint), device=args.sam_device)
        sam.set_image(volume, spacing=spacing, affine=affine)
        distance = distance_transform_edt(np.pad(truth, 1), sampling=spacing)[1:-1, 1:-1, 1:-1]
        point = tuple(int(v) for v in np.unravel_index(distance.argmax(), truth.shape))
        prediction, logits = sam.predict_mask(points=[point], labels=[1])
        if prediction.shape != volume.shape or not np.isfinite(prediction).all():
            raise AssertionError('SAM returned invalid original-grid mask')
        if tuple(logits.shape) != (1, 1, 32, 32, 32) or not torch.isfinite(logits).all():
            raise AssertionError('SAM returned invalid refinement logits')
        iou, dice = compute_metrics(prediction, truth)
        report['steps'].append({'operation': 'positive_point', 'point_zyx': point, 'iou': iou, 'dice': dice})
        if not prediction.any() or iou == 0:
            raise AssertionError('Ground-truth interior prompt produced no overlapping segmentation')
        missing, excess = truth & ~prediction.astype(bool), ~truth & prediction.astype(bool)
        positive = distance_transform_edt(np.pad(missing, 1), sampling=spacing)[1:-1, 1:-1, 1:-1]
        negative = distance_transform_edt(np.pad(excess, 1), sampling=spacing)[1:-1, 1:-1, 1:-1]
        label = int(positive.max() >= negative.max())
        distance = positive if label else negative
        if distance.max():
            point = tuple(int(v) for v in np.unravel_index(distance.argmax(), truth.shape))
            prediction, logits = sam.predict_mask(points=[point], labels=[label], mask_logits=logits)
            if prediction.shape != volume.shape or not torch.isfinite(logits).all():
                raise AssertionError('Refinement returned invalid mask or logits')
            iou, dice = compute_metrics(prediction, truth)
            report['steps'].append({'operation': 'refinement', 'point_zyx': point, 'label': label, 'iou': iou, 'dice': dice})
        if not prediction.any() or compute_metrics(prediction, truth)[0] == 0:
            raise AssertionError('Refinement lost all overlap with the target; inspect per-turn metrics')
        output = args.output_dir / 'sam_prediction.nii.gz'
        save_nifti(prediction, str(output), affine=affine, reference_path=str(args.volume))
        restored, _, restored_affine = load_nifti(str(output), return_affine=True)
        np.testing.assert_array_equal(restored, prediction)
        np.testing.assert_allclose(restored_affine, affine)
        report['steps'].append({'operation': 'nifti_round_trip', 'path': str(output), 'success': True})
        if args.agent:
            from medsam_agent3d.agent import QwenVLAgent
            from medsam_agent3d.loop import run_agent_loop
            from medsam_agent3d.evaluation import save_case_result, result_dict
            config = argparse.Namespace(device=args.agent_device, load_in_4bit=args.load_in_4bit)
            agent = QwenVLAgent(args.model, **_build_model_kwargs(config))
            result = run_agent_loop(agent, sam, volume, truth, args.target, 'validation',
                                    spacing=spacing, affine=affine, max_turns=5,
                                    temperature=0, seed=args.seed, output_dir=str(args.output_dir / 'agent'))
            save_case_result(result, str(args.output_dir / 'agent'), str(args.volume))
            report['agent_result'] = result_dict(result)
            if not result.success:
                raise AssertionError(f'Real agent failed: {result.error}')
        report['success'] = True
        exit_code = 0
    except (PermissionError, FileNotFoundError, ModuleNotFoundError) as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        report['blocked_by_environment'] = True
        exit_code = 2
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        report['traceback'] = traceback.format_exc()
    finally:
        report['runtime_sec'] = time.monotonic() - started
        output = args.output_dir / 'validation_report.json'
        output.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps({'success': report['success'], 'error': report.get('error'), 'report': str(output)}, indent=2))
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
