"""Install and check this project with the current Python interpreter.

CPU regression environment: python scripts/bootstrap.py --mode cpu
Existing CUDA environment: python scripts/bootstrap.py --mode cuda
Read-only diagnostics: python scripts/bootstrap.py --doctor
Checks after installation: python scripts/bootstrap.py --check
A matching CUDA torch/torchvision pair can be installed separately or selected
with --torch-index using the URL from pytorch.org/get-started/locally/.
"""
import argparse
import importlib.util
import json
import os
import tempfile
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEPENDENCIES = ('numpy', 'scipy', 'PIL', 'cv2', 'nibabel', 'torch',
                'torchvision', 'torchio', 'transformers', 'accelerate',
                'qwen_vl_utils', 'pytest')


def run(command):
    print('+', subprocess.list2cmdline([str(part) for part in command]), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def doctor(checkpoint=None, output_dir=None):
    default_checkpoint = ROOT / 'checkpoints/sam_med3d_turbo.pth'
    if not default_checkpoint.is_file():
        default_checkpoint = ROOT / 'sam_med3d_turbo.pth'
    checkpoint = Path(checkpoint) if checkpoint else default_checkpoint
    output_dir = Path(output_dir) if output_dir else ROOT / 'outputs/inference'
    report = {'python': sys.executable, 'version': platform.python_version(),
              'platform': platform.platform(), 'project': str(ROOT),
              'dependencies': {name: importlib.util.find_spec(name) is not None
                               for name in DEPENDENCIES},
              'sam_source': (ROOT / 'SAM-Med3D/segment_anything/build_sam3D.py').is_file(),
              'checkpoint': str(checkpoint), 'checkpoint_exists': checkpoint.is_file(),
              'output_dir': str(output_dir.resolve()),
              'output_parent_exists': output_dir.parent.is_dir(),
              'output_parent_writable_hint': os.access(output_dir.parent, os.W_OK)}
    try:
        with checkpoint.open('rb') as handle:
            handle.read(4)
        report['checkpoint_readable'] = True
    except OSError as exc:
        report['checkpoint_readable'] = False
        report['checkpoint_error'] = f'{type(exc).__name__}: {exc}'
    if report['dependencies']['torch']:
        try:
            import torch
            report['torch_version'] = torch.__version__
            report['cuda_available'] = torch.cuda.is_available()
            if torch.cuda.is_available():
                report['gpus'] = [{'name': torch.cuda.get_device_name(i),
                                  'memory_gib': round(torch.cuda.get_device_properties(i).total_memory / 2**30, 2)}
                                 for i in range(torch.cuda.device_count())]
        except Exception as exc:
            report['torch_error'] = f'{type(exc).__name__}: {exc}'
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--doctor', action='store_true', help='Report only; never install packages')
    parser.add_argument('--check', action='store_true', help='Run tests with existing dependencies; do not install')
    parser.add_argument('--torch-index', help='Optional official PyTorch wheel index for matching torch and torchvision')
    parser.add_argument('--quantized', action='store_true', help='Install optional bitsandbytes for four-bit CUDA inference')
    parser.add_argument('--checkpoint', help='Checkpoint path to diagnose; weights are never loaded by this script')
    parser.add_argument('--output-dir', default=str(ROOT / 'outputs/inference'), help='Output path to diagnose and probe during checks')
    args = parser.parse_args()
    if args.doctor:
        doctor(args.checkpoint, args.output_dir)
        return
    if args.quantized and args.mode != 'cuda':
        parser.error('--quantized requires --mode cuda')
    try:
        if not args.check:
            if importlib.util.find_spec('pip') is None:
                run([sys.executable, '-m', 'ensurepip', '--upgrade'])
            if not (ROOT / 'SAM-Med3D/segment_anything/build_sam3D.py').is_file():
                run(['git', 'submodule', 'update', '--init', '--recursive'])
            run([sys.executable, 'scripts/prepare_sam.py'])
            # Keep the user's matching CUDA pair by default. CPU setup explicitly
            # chooses CPU wheels, separately from application dependencies.
            index = args.torch_index or ('https://download.pytorch.org/whl/cpu' if args.mode == 'cpu' else None)
            if index:
                run([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', '--index-url', index])
            run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements/dev.txt'])
            if args.quantized:
                run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements/quantized.txt'])
        report = doctor(args.checkpoint, args.output_dir)
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=output) as probe:
            probe.write(b'output write probe')
        print('Output directory write probe passed.')
        missing = [name for name, present in report['dependencies'].items() if not present]
        if missing:
            raise RuntimeError('Missing dependencies: ' + ', '.join(missing))
        if args.mode == 'cuda' and not report['checkpoint_readable']:
            raise RuntimeError('Checkpoint is not readable: ' + report.get('checkpoint_error', report['checkpoint']))
        if args.mode == 'cuda' and not report.get('cuda_available'):
            raise RuntimeError('CUDA is unavailable. Install a matching PyTorch CUDA pair for your driver, or use --mode cpu for regression tests.')
        run([sys.executable, '-m', 'compileall', '-q', 'medsam_agent3d', 'scripts', 'run.py'])
        run([sys.executable, '-m', 'pytest', '-q'])
        run([sys.executable, 'tests/test_pipeline.py'])
        run([sys.executable, 'tests/test_benchmark.py'])
    except (subprocess.CalledProcessError, RuntimeError, OSError) as exc:
        print(f'Bootstrap failed: {exc}', file=sys.stderr)
        print('No successful installation or test run is implied. Network/socket restrictions require running this command in a network-enabled environment, such as Colab. Inspect --doctor for missing packages.', file=sys.stderr)
        raise SystemExit(1)
    print('Software checks passed. Real model quality requires a separate checkpoint evaluation.')


if __name__ == '__main__':
    main()
