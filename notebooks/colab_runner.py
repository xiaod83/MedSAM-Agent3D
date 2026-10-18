"""Paired cell source for notebooks/colab.ipynb. Execute with a Colab kernel in VS Code."""

# %% [markdown]
# # MedSAM-Agent3D - laptop in charge, Colab on leg day
#
# Open **this local notebook in VS Code**, then **Select Kernel - Colab - a GPU server**. The notebook stays in your workspace; Python and the models execute on Colab. No WSL or terminal-to-Colab CLI needed.
#
# **One upload before Run All:**
# 1. The ready-made source + the evaluation scans is `outputs/bundles/MedSAM-Agent3D-colab.zip` (size printed by the bundle builder).
# 2. After connecting the Colab kernel, right-click that ZIP in VS Code Explorer - **Upload to Colab**.
# 3. This ZIP includes all grounding and Qwen3.8 tool-call fixes. Run the cells below in order; no separate patches or extra cells are needed. To rebuild after editing local code, run `.\.venv\Scripts\python.exe run.py bundle` in your local PowerShell terminal, then upload the new ZIP.
#
# The default mode is a **12-case development evaluation**: two new test-split scans, each with liver, spleen, left/right kidneys and left/right femurs. The original s0000 left femur remains the SAM smoke control. Set EVALUATION_MODE="single" for that original run. Upload your own scan/mask via the same Explorer action and change their remote paths below when ready.
#
# [Official extension guide](https://github.com/googlecolab/colab-vscode/wiki/User-Guide) - If the upload menu is absent, enable **Colab: Uploading** in VS Code settings. Results can be downloaded from the extension's **Colab - Contents** view.

# %% [markdown]
# ## 1 - Pick the case
#
# These are **paths on the Colab machine**, not Windows paths. This profile uses **Qwen3.8-27B in four-bit** on an **A100 40GB or larger**. Actual fit depends on input and runtime allocations. The base checkpoint download is much larger than 8B; four-bit conversion happens during loading.

# %%
from pathlib import Path
from datetime import datetime, timezone
import csv, hashlib, json, platform, shutil, subprocess, sys, zipfile

BUNDLE = Path('/content/MedSAM-Agent3D-colab.zip')
REPO_DIR = Path('/content/MedSAM-Agent3D')
VOLUME_PATH = REPO_DIR / 'data/Totalsegmentator_dataset_v201/s0000/ct.nii.gz'
MASK_PATH = REPO_DIR / 'data/Totalsegmentator_dataset_v201/s0000/segmentations/femur_left.nii.gz'
EVALUATION_MODE = 'suite'  # 'suite': 12 new scan/target pairs; 'single': original control or custom scan.
SHOW_CASE_IMAGES = 3  # All case PNGs are saved; limit inline images.
TARGET = 'femur_left'
MODEL_ID = 'Qwen/Qwen3.8-27B'  # Native vision-language model; requires the new Qwen3.5-family loader.
MAX_TURNS = 5
SEED = 42
LOAD_IN_4BIT = True  # 27B: four-bit on A100 40GB. Unquantized weights alone are about 54 GB.
RUN_TESTS = False    # Optional full software regression suite.
RUN_SAM_SMOKE = True  # Run SAM with reference-selected clicks before testing Qwen grounding.
EXISTING_CHECKPOINT = None  # Optional readable REMOTE .pth path; mismatches trigger verified download.
RUN_ID = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
RUN_DIR = REPO_DIR / 'outputs' / 'notebook' / RUN_ID
WEIGHTS_DIR = Path('/content/medsam_weights')
CHECKPOINT_URL = 'https://huggingface.co/blueyo0/SAM-Med3D/resolve/main/sam_med3d_turbo.pth'
CHECKPOINT_SHA256 = '899a46d04d3b70f723282ceb489149373558bf0aaba389a346f5ab57da5cdd3c'

if platform.system() != 'Linux' or not Path('/content').is_dir():
    raise RuntimeError('Select the Colab GPU kernel in VS Code, then rerun this cell. This is currently a local kernel.')
try:
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'], text=True).splitlines()[0]
    GPU_NAME, memory_mib = gpu.rsplit(',', 1)
    VRAM_GIB = float(memory_mib) / 1024
except (OSError, subprocess.CalledProcessError, IndexError, ValueError) as exc:
    raise RuntimeError('This kernel has no visible NVIDIA GPU. Select a GPU Colab server.') from exc
if LOAD_IN_4BIT is None:
    LOAD_IN_4BIT = VRAM_GIB < 24 or '32B' in MODEL_ID or '27B' in MODEL_ID
if any(size in MODEL_ID for size in ('32B', '27B')) and VRAM_GIB < (32 if LOAD_IN_4BIT else 72):
    raise RuntimeError('The larger-model notebook profile requires an A100-class GPU with at least 32 GiB for four-bit loading (72 GiB unquantized). Select an A100, or explicitly choose the 8B model. Fit still depends on input length and runtime allocations.')
print(f'{GPU_NAME.strip()} - {VRAM_GIB:.1f} GiB VRAM - four-bit Qwen: {LOAD_IN_4BIT}')
print('Remote Python:', sys.executable)

# %% [markdown]
# ## 2 - Unpack the good stuff
#
# The ZIP contains source, the two evaluation scans with selected masks, and the original smoke-control scan. The full 30 GiB dataset and model weights are not uploaded.

# %%
if not BUNDLE.is_file():
    raise FileNotFoundError(f'Right-click the local source ZIP - Upload to Colab, then set BUNDLE to its remote path. Expected: {BUNDLE}')
REPO_DIR.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(BUNDLE) as archive:
    for member in archive.infolist():
        target = (REPO_DIR / member.filename).resolve()
        is_link = ((member.external_attr >> 16) & 0o170000) == 0o120000
        if not target.is_relative_to(REPO_DIR.resolve()) or is_link:
            raise ValueError(f'Unsafe archive entry: {member.filename}')
    archive.extractall(REPO_DIR)
manifest_path = REPO_DIR / 'bundle_manifest.json'
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
if manifest.get('runtime_profile') != 'qwen38-native-tools-v1':
    raise RuntimeError('This is an older bundle. Replace the remote ZIP with the current outputs/bundles/MedSAM-Agent3D-colab.zip and rerun. The current bundle includes native Qwen3.8 tool calls; no patch upload is needed.')
for name, expected in manifest['files'].items():
    path = (REPO_DIR / name).resolve()
    if not path.is_relative_to(REPO_DIR.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f'Source integrity check failed: {name}. Rebuild and re-upload the bundle.')
if 'normalized_0_999_panel_uv_or_zyx' not in (REPO_DIR / 'medsam_agent3d/loop.py').read_text(encoding='utf-8'):
    raise RuntimeError('This uploaded bundle predates the panel grounding fixes. Upload the newly rebuilt local ZIP, replacing the old remote ZIP, and rerun this cell.')
if EVALUATION_MODE == 'suite' and not (REPO_DIR / 'suite_cases.json').is_file():
    raise RuntimeError('Upload the newly rebuilt evaluation bundle; this ZIP has only the original control scan.')
if 'Qwen3.8' in MODEL_ID and not (REPO_DIR / 'requirements/qwen38.txt').is_file():
    raise RuntimeError('Uploaded bundle lacks Qwen3.8 support. Replace the remote ZIP with the newly rebuilt local bundle.')
print('Bundle SHA256:', hashlib.sha256(BUNDLE.read_bytes()).hexdigest())
print('Model profile:', MODEL_ID, '| bundle:', manifest['runtime_profile'])
RUN_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
shutil.copy2(manifest_path, RUN_DIR / 'bundle_manifest.json')


def archive_results():
    destination = REPO_DIR / 'outputs' / f'notebook_{RUN_ID}'
    archive = Path(shutil.make_archive(str(destination), 'zip', RUN_DIR))
    print('Download via Colab Contents:', archive)
    return archive


def run_logged(command, label):
    print('Running:', subprocess.list2cmdline([str(arg) for arg in command]), flush=True)
    try:
        with (RUN_DIR / f'{label}.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=REPO_DIR, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            with process.stdout:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
            status = process.wait()
            if status:
                raise subprocess.CalledProcessError(status, command)
    finally:
        archive_results()

print('Source verified. Results:', RUN_DIR)

# %% [markdown]
# ## 3 - Give the remote Python its dependencies
#
# This installs into the Colab kernel's environment. Model commands run in fresh Python processes so the notebook does not need to restart after installation.

# %%
requirements = ('requirements/qwen38.txt' if 'Qwen3.8' in MODEL_ID else
                'requirements/quantized.txt' if LOAD_IN_4BIT else 'requirements/base.txt')
run_logged([sys.executable, '-m', 'pip', 'install', '-q', '-r', requirements], 'install')
run_logged([sys.executable, 'scripts/prepare_sam.py'], 'sam_import_setup')
if RUN_TESTS:
    run_logged([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements/dev.txt'], 'install_tests')
    run_logged([sys.executable, 'run.py', 'test', '-q'], 'tests')

# %% [markdown]
# ## 4 - Fetch SAM, without the checkpoint drama
#
# Reuse a hash-verified remote checkpoint or download the published Turbo weights. An unrecognized existing checkpoint is left untouched. No Drive mount is required.

# %%
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
    if drive_checkpoint is not None and drive_checkpoint.is_file():
        try:
            verify_checkpoint(drive_checkpoint)
        except (OSError, ValueError) as exc:
            print(f'Preserving unrecognized existing checkpoint and downloading official weights: {exc}')
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



SAM_CKPT = ensure_checkpoint(WEIGHTS_DIR / 'sam_med3d_turbo.pth',
                             Path(EXISTING_CHECKPOINT) if EXISTING_CHECKPOINT else None)
print('Verified SAM checkpoint:', SAM_CKPT)
if not VOLUME_PATH.is_file():
    raise FileNotFoundError(f'Upload your scan through Explorer or use the bundled sample: {VOLUME_PATH}')
if MASK_PATH is not None and not MASK_PATH.is_file():
    raise FileNotFoundError(f'Mask not found: {MASK_PATH}. Set MASK_PATH=None for inference without ground truth.')
config = dict(evaluation_mode=EVALUATION_MODE, target=TARGET, model=MODEL_ID, volume=str(VOLUME_PATH),
              mask=str(MASK_PATH) if MASK_PATH else None, max_turns=MAX_TURNS,
              seed=SEED, load_in_4bit=LOAD_IN_4BIT, gpu=GPU_NAME.strip(), vram_gib=VRAM_GIB)
(RUN_DIR / 'notebook_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
SAM_SMOKE_READY = not RUN_SAM_SMOKE
if RUN_SAM_SMOKE:
    if MASK_PATH is None:
        raise ValueError('The optional SAM smoke test needs a nonempty reference mask.')
    print('SAM smoke test first: reference-selected clicks isolate SAM from Qwen grounding.')
    try:
        run_logged([sys.executable, 'run.py', 'validate', '--checkpoint', str(SAM_CKPT),
                    '--volume', str(VOLUME_PATH), '--mask', str(MASK_PATH), '--target', TARGET,
                    '--sam-device', 'cuda', '--output-dir', str(RUN_DIR / 'sam_smoke')], 'sam_smoke')
    finally:
        smoke_report_path = RUN_DIR / 'sam_smoke' / 'validation_report.json'
        if smoke_report_path.is_file():
            smoke_report = json.loads(smoke_report_path.read_text(encoding='utf-8'))
            print('SAM smoke execution checks passed:', smoke_report.get('success', False))
            scored = [step for step in smoke_report.get('steps', []) if step.get('iou') is not None]
            for step in scored:
                print('SAM', step.get('operation'), '| IoU:', step.get('iou'), '| Dice:', step.get('dice'))
            overlap = scored[-1]['iou'] > 0 if scored else None
            quality = 'unscored' if overlap is None else ('overlap_unvalidated' if overlap else 'no_overlap')
            SAM_SMOKE_READY = bool(smoke_report.get('success') and overlap is True)
            print('SAM quality_status:', quality, '| target_overlap:', overlap)
            if overlap is False:
                print('SEGMENTATION QUALITY FAILURE: SAM has ZERO target overlap even with a reference-selected click. Inspect SAM geometry/checkpoint before assessing Qwen.')
            if smoke_report.get('error'):
                print('SAM smoke error:', smoke_report['error'])
            print('A passing smoke test or any overlap does not establish segmentation accuracy.')
        else:
            print('No SAM report generated. Inspect sam_smoke.log before running the agent.')
else:
    print('SAM smoke test disabled; SAM and Qwen grounding failures will not be isolated.')

# %% [markdown]
# ## 5 - Let Qwen drive
#
# Review the SAM smoke metrics above before starting the agent. The actual loop: tri-view - Qwen action - SAM mask - updated tri-view. Ground truth, when present, is used only for scoring this agent run. First use downloads the Qwen weights to the remote runtime.

# %%
if not SAM_SMOKE_READY:
    raise RuntimeError('SAM smoke did not pass execution and target-overlap checks. Inspect sam_smoke.log and validation_report.json before downloading/running Qwen. A larger Qwen cannot repair this SAM failure.')
common = ['--model', MODEL_ID, '--sam-ckpt', str(SAM_CKPT),
          '--device', 'cuda', '--sam-device', 'cuda', '--max-turns', str(MAX_TURNS),
          '--temperature', '0', '--seed', str(SEED), '--output-dir', str(RUN_DIR)]
if LOAD_IN_4BIT:
    common.append('--load-in-4bit')
if EVALUATION_MODE == 'suite':
    cases_path = REPO_DIR / 'suite_cases.json'
    cases = json.loads(cases_path.read_text(encoding='utf-8'))
    shutil.copy2(cases_path, RUN_DIR / 'suite_cases.json')
    shutil.copy2(REPO_DIR / 'suite_metadata.json', RUN_DIR / 'suite_metadata.json')
    print('Evaluating', len(cases), 'scan/target pairs; models load once. All cases count.')
    command = [sys.executable, 'run.py', 'batch', '--cases-json', str(cases_path),
               '--max-samples', str(len(cases)), '--no-shuffle'] + common
elif EVALUATION_MODE == 'single':
    command = [sys.executable, 'run.py', 'single', '--volume', str(VOLUME_PATH),
               '--target', TARGET] + common
    if MASK_PATH is not None:
        command += ['--mask', str(MASK_PATH)]
else:
    raise ValueError("EVALUATION_MODE must be 'suite' or 'single'")
try:
    run_logged(command, 'agent')
except subprocess.CalledProcessError:
    # Batch evaluates every case before reporting quality/operational failures.
    # Preserve and display those results; initialization failures still raise.
    if EVALUATION_MODE != 'suite' or not (RUN_DIR / 'evaluation_summary.json').is_file():
        raise
    print('Batch contains failed cases. Review the complete table below; failures are included.')

# %% [markdown]
# ## 6 - Review every result
#
# The summary reports execution success separately from IoU/Dice. A completed loop is not a guarantee of accurate segmentation. If the previous cell failed, you can still run this cell to inspect saved errors and logs.

# %%
from IPython.display import display, Markdown
from PIL import Image

summary_path = RUN_DIR / 'evaluation_summary.json'
if EVALUATION_MODE == 'suite' and summary_path.is_file():
    results = json.loads(summary_path.read_text(encoding='utf-8'))
    rows = ['| Scan | Target | Executed | Dice | IoU | Error |', '|---|---|---|---:|---:|---|']
    csv_rows = []
    for result in results:
        dice, iou = result.get('final_dice'), result.get('final_iou')
        completed = bool(result.get('success'))
        csv_rows.append(dict(scan=result['volume_id'], target=result['target_name'], executed=completed,
                             dice=dice, iou=iou, error=result.get('error', '')))
        dice_text = '-' if dice is None else f'{dice:.4f}'
        iou_text = '-' if iou is None else f'{iou:.4f}'
        error = result.get('error', '').replace('|', '/').replace('\n', ' ')
        rows.append(f"| {result['volume_id']} | {result['target_name']} | {completed} | {dice_text} | {iou_text} | {error} |")
    display(Markdown('\n'.join(rows)))
    if results:
        def score(result, metric):
            return float(result.get(metric) or 0) if result.get('success') else 0.0
        print('All-case mean Dice (failures=0):', sum(score(r, 'final_dice') for r in results) / len(results))
        print('All-case mean IoU (failures=0):', sum(score(r, 'final_iou') for r in results) / len(results))
        print('Execution failures:', sum(not r.get('success') for r in results), '/', len(results))
        for target in sorted({r['target_name'] for r in results}):
            group = [r for r in results if r['target_name'] == target]
            print(target, '| mean Dice:', sum(score(r, 'final_dice') for r in group) / len(group))
        with (RUN_DIR / 'case_scores.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
            writer.writeheader(); writer.writerows(csv_rows)
        for result in results[:SHOW_CASE_IMAGES]:
            image_path = RUN_DIR / result['volume_id'] / result['target_name'] / 'final.png'
            if image_path.is_file():
                print(result['volume_id'], result['target_name'])
                display(Image.open(image_path))
    print('Per-case masks, PNGs and trajectories:', RUN_DIR)
else:
    trajectory_path = RUN_DIR / 'trajectory.json'
    if trajectory_path.is_file():
        result = json.loads(trajectory_path.read_text(encoding='utf-8'))
        print('Execution completed:', result['success'], '| termination:', result.get('termination_reason'))
        target_overlap = result.get('target_overlap')
        if target_overlap is None and result.get('final_iou') is not None:
            target_overlap = result['final_iou'] > 0
        quality_status = result.get('quality_status')
        if quality_status is None:
            quality_status = 'unscored' if target_overlap is None else ('overlap_unvalidated' if target_overlap else 'no_overlap')
        print('quality_status:', quality_status, '| target_overlap:', target_overlap)
        if quality_status == 'no_overlap' or target_overlap is False:
            display(Markdown('**SEGMENTATION QUALITY FAILURE: ZERO overlap with the reference target. The loop may have executed successfully, but this prediction did not segment the requested structure. Inspect the smoke test, prompts, and overlays.**'))
        elif target_overlap is True:
            print('Some target overlap is present; this does not establish accurate segmentation.')
        else:
            print('Quality is unscored. Review the mask; execution completion does not establish accuracy.')
        if result.get('error'):
            print('Error:', result['error'])
        print('Final IoU:', result.get('final_iou'), '| Dice:', result.get('final_dice'))
        rows = ['| Attempt | Action | IoU | Dice | Mask voxels |', '|---:|---|---:|---:|---:|']
        for turn in result.get('turns', []):
            iou = '-' if turn.get('iou') is None else f"{turn['iou']:.4f}"
            dice = '-' if turn.get('dice') is None else f"{turn['dice']:.4f}"
            rows.append(f"| {turn['turn']} | {turn['action'].get('name', '?')} | {iou} | {dice} | {turn['mask_voxels']} |")
        display(Markdown('\n'.join(rows)))
    else:
        print('No trajectory yet; inspect the printed error and agent.log in the result ZIP.')
    final_image = RUN_DIR / 'final.png'
    if final_image.is_file():
        display(Image.open(final_image))
RESULT_ZIP = archive_results()

# %% [markdown]
# ### Take the results home
#
# In the **Colab activity bar - Contents**, find the ZIP printed above and choose **Download**. It contains the prediction, PNGs, per-turn actions, metrics, configuration and logs. Save this notebook in VS Code to keep its displayed results locally.
#
# After editing project source locally, rebuild/upload the source ZIP and rerun from cell 2. Runtime files and downloaded model caches are ephemeral unless copied out. When finished, remove the GPU server using the Colab toolbar.
