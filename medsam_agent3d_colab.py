"""
MedSAM-Agent3D — Google Colab Evaluation Runner
================================================
Upload this .py to Colab and run cell by cell.
Evaluates vanilla (non-trained) Qwen3-VL + frozen SAM-Med3D
on 3D medical segmentation using the agentic inference loop.

Requirements: T4 GPU (or better) with ~16 GB VRAM
"""

# ======================================================================
# Cell 1: Mount Drive & set workdir
# ======================================================================

from google.colab import drive
drive.mount('/content/drive')

import os
WORK_DIR = '/content/drive/MyDrive/MedSAM-Agent3D'
os.makedirs(WORK_DIR, exist_ok=True)
os.chdir(WORK_DIR)
print(f'Working directory: {WORK_DIR}')

# ======================================================================
# Cell 2: Install system + Python deps
# ======================================================================

!apt-get update -qq && apt-get install -y -qq libgl1-mesa-glx libglib2.0-0 > /dev/null 2>&1

!pip install -q \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 \
    transformers>=4.45.0 \
    accelerate \
    nibabel \
    scipy \
    opencv-python \
    Pillow \
    scikit-image \
    tqdm \
    huggingface-hub \
    bitsandbytes \
    peft \
    qwen-vl-utils \
    torchio \
    SimpleITK

print('\nAll dependencies installed')

# ======================================================================
# Cell 3: Verify GPU
# ======================================================================

import torch
print(f'PyTorch:  {torch.__version__}')
print(f'CUDA:     {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:      {torch.cuda.get_device_name(0)}')
    print(f'VRAM:     {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')

# ======================================================================
# Cell 4: Clone/Setup Repo
# ======================================================================

REPO_DIR = os.path.join(WORK_DIR, 'MedSAM-Agent3D')

if os.path.exists(os.path.join(REPO_DIR, 'agent_inference.py')):
    print('Repo exists with agent_inference.py. Updating...')
    %cd $REPO_DIR
    !git pull 2>/dev/null || echo '(git pull skipped — using existing files)'
else:
    %cd $WORK_DIR
    # Replace with your repo URL
    !git clone https://github.com/your-org/MedSAM-Agent3D.git 2>/dev/null || \
      echo '⚠️ Git clone failed. Upload the MedSAM-Agent3D folder to Drive manually.'
    # If clone failed but you uploaded manually, it'll still work below

%cd $REPO_DIR
import sys
sys.path.insert(0, REPO_DIR)
print(f'Repo directory: {REPO_DIR}')

# ======================================================================
# Cell 5: Download Qwen3-VL-8B-Instruct (~16 GB, one time only)
# ======================================================================

MODEL_CACHE = os.path.join(WORK_DIR, 'model_cache')
os.makedirs(MODEL_CACHE, exist_ok=True)

from huggingface_hub import snapshot_download

QWEN_DIR = os.path.join(MODEL_CACHE, 'Qwen3-VL-8B-Instruct')
if not os.path.exists(os.path.join(QWEN_DIR, 'config.json')):
    print('Downloading Qwen3-VL-8B-Instruct (~16 GB, may take 10–20 min)...')
    snapshot_download(
        repo_id='Qwen/Qwen3-VL-8B-Instruct',
        local_dir=QWEN_DIR,
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print('✅ Qwen3-VL-8B-Instruct downloaded.')
else:
    print('✅ Qwen3-VL-8B-Instruct already cached.')

# ======================================================================
# Cell 6: Verify SAM-Med3D checkpoint
# ======================================================================

SAM_CKPT = os.path.join(REPO_DIR, 'sam_med3d_turbo.pth')
if os.path.exists(SAM_CKPT) and os.path.getsize(SAM_CKPT) > 1e6:
    print(f'✅ SAM-Med3D checkpoint: {os.path.getsize(SAM_CKPT)/1e6:.0f} MB')
else:
    print('⚠️ SAM-Med3D checkpoint NOT found!')
    print('   Upload sam_med3d_turbo.pth to your Drive MedSAM-Agent3D/ folder.')
    print(f'   Expected path: {SAM_CKPT}')

# ======================================================================
# Cell 7: Verify TotalSegmentator data
# ======================================================================

from utils.io_utils import get_totalsegmentator_dataset

TS_DIR = os.path.join(REPO_DIR, 'Totalsegmentator_dataset_v201')
dataset = get_totalsegmentator_dataset(TS_DIR)
print(f'TotalSegmentator samples found: {len(dataset)}')

if len(dataset) == 0:
    print('⚠️ No data found! Upload your TotalSegmentator dataset to:')
    print(f'   {TS_DIR}/')
else:
    # Show a few samples
    for d in dataset[:5]:
        print(f'  {d["volume_id"]:<8s}  {d["target_name"]:<20s}')

# ======================================================================
# Cell 8: Load models (Qwen3-VL + SAM-Med3D)
# ======================================================================

from agent_inference import QwenVLAgent
from models.sam_med3d_wrapper import FrozenSAMMed3D
from transformers import BitsAndBytesConfig

print('Loading Qwen3-VL (4-bit quantized for T4)...')
mllm = QwenVLAgent(
    QWEN_DIR,
    device_map='auto',
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
    ),
)
print('✅ Qwen3-VL loaded.')

print('\nLoading SAM-Med3D...')
sam = FrozenSAMMed3D(checkpoint_path=SAM_CKPT)
print('✅ SAM-Med3D loaded.')

# ======================================================================
# Cell 9 [DEMO]: Run agent on a single volume
# ======================================================================

from agent_inference import run_agent_loop
from utils.io_utils import load_nifti
import numpy as np

# Pick the first sample from the dataset
sample = dataset[0]
print(f'Demo sample: {sample["volume_id"]} / {sample["target_name"]}')

vol, spacing = load_nifti(sample['ct_path'])
mask, _ = load_nifti(sample['mask_path'])
mask = (mask > 0).astype(np.uint8)

print(f'Volume shape: {vol.shape}  |  mask voxels: {int(mask.sum()):,}  |  spacing: {spacing}')

result = run_agent_loop(
    mllm_agent=mllm,
    sam_model=sam,
    volume=vol,
    gt_mask=mask,
    target_name=sample['target_name'],
    volume_id=sample['volume_id'],
    spacing=spacing,
    max_turns=5,
    output_dir=os.path.join(REPO_DIR, 'eval_outputs', 'demo'),
    temperature=0.1,
    seed=42,
)

print(f'\n{"="*50}')
print(f'DEMO RESULT')
print(f'  Target:   {result.target_name}')
print(f'  Success:  {result.success}')
print(f'  Turns:    {len(result.turns)}')
print(f'  Final IoU:  {result.final_iou:.4f}')
print(f'  Final Dice: {result.final_dice:.4f}')
print(f'{"="*50}')

for t in result.turns:
    print(f'  Turn {t.turn}: action={t.action["name"]}, IoU={t.iou:.4f}, Dice={t.dice:.4f}')

# ======================================================================
# Cell 10 [BATCH]: Run agent on the full dataset / subset
# ======================================================================

RUN_BATCH = True         # ← set to False to skip batch eval
MAX_SAMPLES = 10         # how many samples to evaluate
MAX_TURNS = 5
SEED = 42

if RUN_BATCH:
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    subset = dataset[:MAX_SAMPLES]
    print(f'Evaluating {len(subset)} samples (max_turns={MAX_TURNS})...\n')

    batch_results = []
    for i, entry in enumerate(subset):
        vol_id = entry['volume_id']
        tgt = entry['target_name']
        print(f'[{i+1}/{len(subset)}] {vol_id} / {tgt} ...', end=' ')

        try:
            vol, spacing = load_nifti(entry['ct_path'])
            mask, _ = load_nifti(entry['mask_path'])
            mask = (mask > 0).astype(np.uint8)

            if mask.sum() == 0:
                print('SKIP (empty mask)')
                continue

            res = run_agent_loop(
                mllm_agent=mllm,
                sam_model=sam,
                volume=vol,
                gt_mask=mask,
                target_name=tgt,
                volume_id=vol_id,
                spacing=spacing,
                max_turns=MAX_TURNS,
                output_dir=os.path.join(REPO_DIR, 'eval_outputs', vol_id, tgt),
                temperature=0.1,
                seed=SEED,
            )
            print(f'IoU={res.final_iou:.4f}  Dice={res.final_dice:.4f}  turns={len(res.turns)}')
            batch_results.append(res)

        except Exception as e:
            print(f'ERROR: {e}')

    # --- Summary ---
    successful = [r for r in batch_results if r.success]
    ious = [r.final_iou for r in successful]
    dices = [r.final_dice for r in successful]
    turns_list = [len(r.turns) for r in successful]

    print(f'\n{"="*50}')
    print('BATCH EVALUATION SUMMARY')
    print(f'{"="*50}')
    print(f'Evaluated: {len(batch_results)}')
    print(f'Success:   {len(successful)}')
    if successful:
        print(f'IoU:       {np.mean(ious):.4f} ± {np.std(ious):.4f}')
        print(f'Dice:      {np.mean(dices):.4f} ± {np.std(dices):.4f}')
        print(f'Turns:     {np.mean(turns_list):.2f} ± {np.std(turns_list):.2f}')

    # Per-organ breakdown
    from collections import defaultdict
    by_organ = defaultdict(lambda: {'ious': [], 'dices': [], 'turns': []})
    for r in successful:
        by_organ[r.target_name]['ious'].append(r.final_iou)
        by_organ[r.target_name]['dices'].append(r.final_dice)
        by_organ[r.target_name]['turns'].append(len(r.turns))
    if len(by_organ) > 1:
        print(f'\nPer-organ:')
        for org, m in sorted(by_organ.items()):
            print(f'  {org:<20s}  IoU={np.mean(m["ious"]):.4f}  Dice={np.mean(m["dices"]):.4f}  Turns={np.mean(m["turns"]):.1f}  (n={len(m["ious"])})')

else:
    print('Batch eval skipped. Set RUN_BATCH = True to run.')

# ======================================================================
# Cell 11 [OPTIONAL]: Inspect output images in Drive
# ======================================================================

import glob
imgs = glob.glob(os.path.join(REPO_DIR, 'eval_outputs', '**', '*.png'), recursive=True)
print(f'Generated {len(imgs)} tri-view PNGs')

if imgs:
    from IPython.display import Image, display
    # Show the first turn of the demo
    demo_imgs = sorted([p for p in imgs if 'demo' in p])
    if demo_imgs:
        print(f'\nDemo tri-view (turn 0):')
        display(Image(filename=demo_imgs[0]))
