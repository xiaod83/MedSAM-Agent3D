"""Run from a local terminal with inference executed on an existing Colab GPU.

Install Google's google-colab-cli in Linux/macOS (WSL on Windows), then:
  colab --auth oauth2 new -s medsam --gpu A100 --high-mem
  python3 scripts/colab_remote.py --session medsam
  colab --auth oauth2 stop -s medsam
Only the selected scan, optional mask, and project source are uploaded. Outputs
are downloaded locally. No local PyTorch or CUDA installation is required.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VOLUME = ROOT / 'data/Totalsegmentator_dataset_v201/s0000/ct.nii.gz'
DEFAULT_MASK = ROOT / 'data/Totalsegmentator_dataset_v201/s0000/segmentations/femur_left.nii.gz'


def create_payload(destination, volume, mask=None):
    sources = set()
    for directory in ('medsam_agent3d', 'scripts', 'tests', 'SAM-Med3D/segment_anything'):
        sources.update(path for path in (ROOT / directory).rglob('*.py') if '__pycache__' not in path.parts)
    for name in ('run.py', 'requirements/base.txt', 'notebooks/colab_runner.py',
                 'requirements.txt', 'requirements/dev.txt', 'requirements/quantized.txt', 'pytest.ini'):
        sources.add(ROOT / name)
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for source in sorted(sources):
            archive.write(source, source.relative_to(ROOT).as_posix())
        volume_name = 'inputs/volume.nii.gz' if volume.name.endswith('.nii.gz') else 'inputs/volume.nii'
        archive.write(volume, volume_name)
        mask_name = None
        if mask is not None:
            mask_name = 'inputs/mask.nii.gz' if mask.name.endswith('.nii.gz') else 'inputs/mask.nii'
            archive.write(mask, mask_name)
    return volume_name, mask_name


def remote_script(config):
    """Send Python code to the official CLI's kernel; subprocess logs stream home."""
    return 'import json\nCONFIG = json.loads(' + repr(json.dumps(config)) + ')\n' + r'''
from pathlib import Path
import hashlib
import shutil
import subprocess
import sys
import traceback
import urllib.request
import zipfile

root = Path(CONFIG['root'])
root.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(CONFIG['payload']) as archive:
    for member in archive.infolist():
        if not (root / member.filename).resolve().is_relative_to(root.resolve()):
            raise ValueError('Unsafe source archive path')
    archive.extractall(root)
output = root / 'outputs'
output.mkdir(exist_ok=True)
status = {'success': False, 'run_id': CONFIG['run_id'], 'mode': CONFIG['mode']}

def run(command):
    print('+', subprocess.list2cmdline(command), flush=True)
    with (output / 'remote.log').open('a', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        process.stdout.close()
        code = process.wait()
        if code:
            raise RuntimeError(f'Command failed with exit status {code}: {command}')

def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()

try:
    # Cache verified weights on the remote VM, never on the local laptop.
    weights = Path('/content/medsam_weights')
    weights.mkdir(exist_ok=True)
    checkpoint = weights / 'sam_med3d_turbo.pth'
    expected = '899a46d04d3b70f723282ceb489149373558bf0aaba389a346f5ab57da5cdd3c'
    if not checkpoint.is_file() or digest(checkpoint) != expected:
        print('Downloading SAM-Med3D Turbo to Colab...', flush=True)
        partial = checkpoint.with_suffix('.partial')
        urllib.request.urlretrieve('https://huggingface.co/blueyo0/SAM-Med3D/resolve/main/sam_med3d_turbo.pth', partial)
        if digest(partial) != expected:
            raise ValueError('Downloaded SAM checkpoint does not match its published SHA256')
        partial.replace(checkpoint)
    command = [sys.executable, 'scripts/bootstrap.py', '--mode', 'cuda',
               '--checkpoint', str(checkpoint), '--output-dir', str(output / 'checks')]
    if CONFIG['quantized']:
        command.append('--quantized')
    run(command)
    volume = str(root / CONFIG['volume'])
    mask = str(root / CONFIG['mask']) if CONFIG['mask'] else None
    if CONFIG['mode'] == 'sam':
        if not mask:
            raise ValueError('SAM checkpoint validation needs --mask for its privileged interior click')
        command = [sys.executable, 'scripts/validate_checkpoint.py', '--checkpoint', str(checkpoint),
                   '--volume', volume, '--mask', mask, '--sam-device', 'cuda',
                   '--target', CONFIG['target'], '--seed', str(CONFIG['seed']),
                   '--output-dir', str(output / 'prediction')]
    else:
        command = [sys.executable, 'run.py', 'single', '--volume', volume,
                   '--target', CONFIG['target'], '--sam-ckpt', str(checkpoint),
                   '--device', 'cuda', '--sam-device', 'cuda', '--model', CONFIG['model'],
                   '--max-turns', str(CONFIG['max_turns']), '--temperature', str(CONFIG['temperature']),
                   '--seed', str(CONFIG['seed']), '--output-dir', str(output / 'prediction')]
        if mask:
            command += ['--mask', mask]
        if CONFIG['quantized']:
            command.append('--load-in-4bit')
    run(command)
    status['success'] = True
except Exception as exc:
    status['error'] = f'{type(exc).__name__}: {exc}'
    status['traceback'] = traceback.format_exc()
finally:
    (output / 'job_status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    shutil.make_archive(str(Path(CONFIG['result']).with_suffix('')), 'zip', output)
    print('Remote results:', CONFIG['result'], flush=True)
if not status['success']:
    raise RuntimeError(status.get('error', 'Remote job failed'))
'''


def extract_results(archive_path, output):
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if not (output / member.filename).resolve().is_relative_to(output.resolve()) or ((member.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError('Unsafe results archive entry')
        archive.extractall(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', default='medsam')
    parser.add_argument('--auth', choices=['oauth2', 'adc'], default='oauth2')
    parser.add_argument('--volume', type=Path, default=DEFAULT_VOLUME)
    parser.add_argument('--mask', type=Path)
    parser.add_argument('--target', default='femur_left')
    parser.add_argument('--model', default='Qwen/Qwen3-VL-8B-Instruct')
    parser.add_argument('--mode', choices=['agent', 'sam'], default='agent')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--max-turns', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--timeout', type=int, default=3600)
    parser.add_argument('--dry-run', action='store_true', help='Build/inspect transfer without connecting or allocating compute')
    args = parser.parse_args(argv)
    if args.max_turns < 1 or args.timeout < 1:
        parser.error('--max-turns and --timeout must be positive')
    import math
    if not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error('--temperature must be finite and nonnegative')
    volume = args.volume.resolve()
    mask = args.mask.resolve() if args.mask else (DEFAULT_MASK if volume == DEFAULT_VOLUME.resolve() else None)
    for path in (volume, mask):
        if path is not None and (not path.is_file() or not (path.name.endswith('.nii') or path.name.endswith('.nii.gz'))):
            parser.error(f'Expected a readable NIfTI path: {path}')
    if args.mode == 'sam' and mask is None:
        parser.error('--mode sam requires --mask')
    cli = shutil.which('colab')
    if not cli and not args.dry_run:
        parser.error('Install google-colab-cli in Linux/macOS (WSL on Windows): uv tool install google-colab-cli. Authenticate and create a session with colab --auth oauth2 new -s medsam --gpu A100.')
    run_id = uuid.uuid4().hex[:12]
    output = (args.output_dir or ROOT / 'outputs/inference/colab' / run_id).resolve()
    output.mkdir(parents=True, exist_ok=True)
    prefix = [cli or 'colab', '--auth', args.auth]
    remote_root = f'/content/medsam_{run_id}'
    remote_payload = remote_root + '.zip'
    remote_results = remote_root + '_results.zip'
    with tempfile.TemporaryDirectory(prefix='medsam_colab_') as directory:
        payload = Path(directory) / 'source_and_inputs.zip'
        volume_name, mask_name = create_payload(payload, volume, mask)
        config = {'run_id': run_id, 'root': remote_root, 'payload': remote_payload,
                  'result': remote_results, 'volume': volume_name, 'mask': mask_name,
                  'mode': args.mode, 'target': args.target, 'model': args.model,
                  'quantized': args.load_in_4bit, 'max_turns': args.max_turns,
                  'temperature': args.temperature, 'seed': args.seed}
        worker = Path(directory) / 'remote_job.py'
        worker.write_text(remote_script(config), encoding='utf-8')
        commands = [prefix + ['status', '-s', args.session],
                    prefix + ['upload', '-s', args.session, str(payload), remote_payload],
                    prefix + ['exec', '-s', args.session, '--timeout', str(args.timeout), '-f', str(worker)],
                    prefix + ['download', '-s', args.session, remote_results, str(output / 'results.zip')]]
        (output / 'submission.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
        if args.dry_run:
            print(json.dumps({'dry_run': True, 'transfer_bytes': payload.stat().st_size,
                              'commands': commands, 'output_dir': str(output)}, indent=2))
            return 0
        subprocess.run(commands[0], check=True)
        subprocess.run(commands[1], check=True)
        execution = subprocess.run(commands[2])
        retrieval = subprocess.run(commands[3])
        if retrieval.returncode:
            print(f'Result retrieval failed; session {args.session} remains available. Remote archive: {remote_results}', file=sys.stderr)
            return execution.returncode or retrieval.returncode
        extract_results(output / 'results.zip', output)
        print('Local results:', output)
        status = json.loads((output / 'job_status.json').read_text())
        if not status['success']:
            print(status.get('error', 'Remote inference failed'), file=sys.stderr)
            return 1
        return execution.returncode


if __name__ == '__main__':
    raise SystemExit(main())
