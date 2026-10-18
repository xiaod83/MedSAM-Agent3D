# %%
"""Colab Pro handoff for the locally patched source bundle.
Choose an A100/high-memory GPU runtime before starting. Run cells in order.
Upload MedSAM-Agent3D-colab.zip built locally, or place it in the Drive path below.
No remote Git checkout is used: the ZIP contains the actual reviewed source.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from google.colab import drive, files

drive.mount('/content/drive')
DRIVE_DIR = Path('/content/drive/MyDrive/MedSAM-Agent3D')
DRIVE_BUNDLE = DRIVE_DIR / 'MedSAM-Agent3D-colab.zip'
REPO_DIR = Path('/content/MedSAM-Agent3D')
RESULT_DIR = Path('/content/medsam_results')
RUN_ID = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
DRIVE_RESULTS = DRIVE_DIR / 'runs' / RUN_ID
DRIVE_CHECKPOINT = DRIVE_DIR / 'sam_med3d_turbo.pth'
CHECKPOINT_URL = 'https://huggingface.co/blueyo0/SAM-Med3D/resolve/main/sam_med3d_turbo.pth'
CHECKPOINT_SHA256 = '899a46d04d3b70f723282ceb489149373558bf0aaba389a346f5ab57da5cdd3c'
MODEL_ID = 'Qwen/Qwen3-VL-8B-Instruct'
LOAD_IN_4BIT = False  # Opt-in for smaller supported GPUs; fitting is not guaranteed.
RUN_AGENT = True
SAMPLE_DIR = REPO_DIR / 'data/Totalsegmentator_dataset_v201/s0000'
VOLUME_PATH = SAMPLE_DIR / 'ct.nii.gz'
MASK_PATH = SAMPLE_DIR / 'segmentations/femur_left.nii.gz'
TARGET = 'femur_left'
RESULT_DIR.mkdir(parents=True, exist_ok=True)

def preserve_results():
    DRIVE_RESULTS.mkdir(parents=True, exist_ok=True)
    shutil.copytree(RESULT_DIR, DRIVE_RESULTS, dirs_exist_ok=True)
    print('Saved logs and results:', DRIVE_RESULTS)

def run_logged(command, name):
    log_path = RESULT_DIR / f'{name}.log'
    try:
        with log_path.open('w', encoding='utf-8') as log:
            log.write(subprocess.list2cmdline([str(arg) for arg in command]) + '\n')
            log.flush()
            process = subprocess.Popen(command, cwd=REPO_DIR, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            with process.stdout:
                for line in process.stdout:
                    print(line, end='')
                    log.write(line)
            code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)
    finally:
        preserve_results()

# %%
if DRIVE_BUNDLE.is_file():
    bundle = DRIVE_BUNDLE
else:
    print('Upload the locally built MedSAM-Agent3D-colab.zip now.')
    uploaded = files.upload()
    if len(uploaded) != 1:
        raise ValueError('Upload exactly one source ZIP.')
    name, payload = next(iter(uploaded.items()))
    bundle = Path('/content') / Path(name).name
    bundle.write_bytes(payload)
    del uploaded, payload
# Extraction refuses traversal or links, and writes source only under /content.
REPO_DIR.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(bundle) as archive:
    for entry in archive.infolist():
        target = (REPO_DIR / entry.filename).resolve()
        if not target.is_relative_to(REPO_DIR.resolve()) or (entry.external_attr >> 16) & 0o170000 == 0o120000:
            raise ValueError(f'Unsafe ZIP entry: {entry.filename}')
    archive.extractall(REPO_DIR)
if not (REPO_DIR / 'scripts/bootstrap.py').is_file():
    raise RuntimeError('Bundle must contain project files at ZIP root. Rebuild using scripts/build_colab_bundle.py.')
manifest = REPO_DIR / 'bundle_manifest.json'
if not manifest.is_file():
    raise RuntimeError('Source bundle manifest missing. Rebuild the handoff ZIP.')
for relative, expected in json.loads(manifest.read_text(encoding='utf-8'))['files'].items():
    source_path = (REPO_DIR / relative).resolve()
    if not source_path.is_relative_to(REPO_DIR.resolve()):
        raise ValueError(f'Unsafe manifest path: {relative}')
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != expected:
        raise ValueError(f'Source bundle hash mismatch: {relative}')
shutil.copy2(manifest, RESULT_DIR / manifest.name)
print('Using patched source extracted to', REPO_DIR)

# %%
# Only a verified checkpoint is reused. Unrecognized Drive files are preserved.
def verify_checkpoint(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != CHECKPOINT_SHA256:
        raise ValueError(f'Checkpoint SHA256 mismatch: {path}. Expected {CHECKPOINT_SHA256}, got {actual}.')


def ensure_checkpoint(destination, drive_checkpoint):
    import urllib.request
    if destination.is_file():
        try:
            verify_checkpoint(destination)
            return destination
        except (OSError, ValueError) as exc:
            print(f'Runtime checkpoint will be replaced with a verified download: {exc}')
    if drive_checkpoint.is_file():
        try:
            verify_checkpoint(drive_checkpoint)
        except (OSError, ValueError) as exc:
            print(f'Preserving unrecognized Drive checkpoint and downloading official weights: {exc}')
        else:
            if drive_checkpoint.resolve() != destination.resolve():
                shutil.copy2(drive_checkpoint, destination)
            verify_checkpoint(destination)
            return destination
    if not CHECKPOINT_URL:
        raise FileNotFoundError('No verified checkpoint available and no download URL configured.')
    partial = destination.with_suffix('.partial')
    urllib.request.urlretrieve(CHECKPOINT_URL, partial)
    verify_checkpoint(partial)
    partial.replace(destination)
    return destination


SAM_CKPT = ensure_checkpoint(REPO_DIR / 'sam_med3d_turbo.pth', DRIVE_CHECKPOINT)
for path in (SAM_CKPT, VOLUME_PATH, MASK_PATH):
    with path.open('rb') as source:
        if not source.read(1):
            raise ValueError(f'Empty input: {path}')
# Application dependencies use PyPI; Colab's CUDA torch/torchvision pair is retained.
command = [sys.executable, 'scripts/bootstrap.py', '--mode', 'cuda',
           '--checkpoint', str(SAM_CKPT), '--output-dir', str(RESULT_DIR / 'checks')]
if LOAD_IN_4BIT:
    command.append('--quantized')
run_logged(command, 'bootstrap_and_tests')

# %%
import torch
if not torch.cuda.is_available():
    raise RuntimeError('Choose a GPU runtime before model validation.')
print('GPU:', torch.cuda.get_device_name(0))
print('VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
print('Native BF16:', torch.cuda.is_bf16_supported())
# First establish actual SAM checkpoint loading, prompts, refinement, and NIfTI export.
validation = [sys.executable, 'scripts/validate_checkpoint.py', '--checkpoint', str(SAM_CKPT),
              '--volume', str(VOLUME_PATH), '--mask', str(MASK_PATH), '--target', TARGET,
              '--sam-device', 'cuda', '--seed', '42']
run_logged(validation + ['--output-dir', str(RESULT_DIR / 'sam_validation')], 'real_sam_validation')

# %%
# Qwen weights download from Hugging Face. A100 is recommended for full precision.
# T4 quantization is opt-in, with no guarantee that both models fit in memory.
if RUN_AGENT:
    command = validation + ['--agent', '--model', MODEL_ID, '--agent-device', 'cuda',
                            '--output-dir', str(RESULT_DIR / 'agent_validation')]
    if LOAD_IN_4BIT:
        command.append('--load-in-4bit')
    run_logged(command, 'real_agent_validation')
preserve_results()
print('Inspect validation_report.json, per-turn records, overlays, and predicted NIfTI masks in Drive.')
print('The SAM smoke test uses a ground-truth foreground click. Only the agent stage tests autonomous prompting.')
