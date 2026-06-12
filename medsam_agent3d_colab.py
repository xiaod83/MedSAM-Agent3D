"""
MedSAM-Agent3D — Google Colab Runner Script
============================================
Usage: Upload this script to Colab and run cell by cell.

Trains Qwen3-VL-8B-Instruct to use SAM-Med3D for interactive 3D medical segmentation.
"""

# ──────────────────────────────────────────────────────────────────────
# Cell 1: Mount Drive & Install Dependencies
# ──────────────────────────────────────────────────────────────────────

from google.colab import drive
drive.mount('/content/drive')

import os
WORK_DIR = '/content/drive/MyDrive/MedSAM-Agent3D'
os.makedirs(WORK_DIR, exist_ok=True)
os.chdir(WORK_DIR)
print(f'Working directory: {WORK_DIR}')

# ──────────────────────────────────────────────────────────────────────
# Cell 2: Install system + Python deps
# ──────────────────────────────────────────────────────────────────────

!apt-get update -qq && apt-get install -y -qq libgl1-mesa-glx libglib2.0-0 > /dev/null 2>&1

!pip install -q \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 \
    transformers>=4.45.0 \
    accelerate \
    trl>=0.9.0 \
    nibabel \
    scipy \
    opencv-python \
    Pillow \
    scikit-image \
    tqdm \
    datasets \
    huggingface-hub \
    bitsandbytes \
    peft \
    qwen-vl-utils

print('\n✅ All dependencies installed')

# ──────────────────────────────────────────────────────────────────────
# Cell 3: Verify GPU
# ──────────────────────────────────────────────────────────────────────

import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')

# ──────────────────────────────────────────────────────────────────────
# Cell 4: Clone/Setup Repo
# ──────────────────────────────────────────────────────────────────────

REPO_DIR = os.path.join(WORK_DIR, 'MedSAM-Agent3D')
if os.path.exists(REPO_DIR):
    print('Repo already exists.')
    %cd $REPO_DIR
else:
    %cd $WORK_DIR
    # Replace with your repo URL, or upload manually
    !git clone https://github.com/your-org/MedSAM-Agent3D.git 2>/dev/null || \
      print('Git clone failed. Upload the MedSAM-Agent3D folder to your Drive.')

%cd $REPO_DIR
print(f'Repo at: {REPO_DIR}')

# ──────────────────────────────────────────────────────────────────────
# Cell 5: Download Model Checkpoints
# ──────────────────────────────────────────────────────────────────────

MODEL_CACHE = os.path.join(WORK_DIR, 'model_cache')
os.makedirs(MODEL_CACHE, exist_ok=True)

from huggingface_hub import snapshot_download

QWEN_DIR = os.path.join(MODEL_CACHE, 'Qwen3-VL-8B-Instruct')
if not os.path.exists(os.path.join(QWEN_DIR, 'config.json')):
    print('Downloading Qwen3-VL-8B-Instruct (~16 GB, may take a while)...')
    snapshot_download(
        repo_id='Qwen/Qwen3-VL-8B-Instruct',
        local_dir=QWEN_DIR,
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print('✅ Qwen3-VL-8B-Instruct downloaded')
else:
    print('✅ Qwen3-VL-8B-Instruct already cached')

# ──────────────────────────────────────────────────────────────────────
# Cell 6: Download SAM-Med3D checkpoint
# ──────────────────────────────────────────────────────────────────────

SAM_CKPT = os.path.join(REPO_DIR, 'sam_med3d_turbo.pth')
if not os.path.exists(SAM_CKPT) or os.path.getsize(SAM_CKPT) < 1e6:
    print('Downloading SAM-Med3D Turbo checkpoint...')
    # Replace URL with the actual checkpoint location
    !wget -qO $SAM_CKPT https://huggingface.co/your-org/sam-med3d-turbo/resolve/main/sam_med3d_turbo.pth 2>/dev/null || \
      echo 'WARNING: SAM-Med3D checkpoint not available. Using mock wrapper.'
    if os.path.exists(SAM_CKPT) and os.path.getsize(SAM_CKPT) > 1e6:
        print(f'✅ SAM-Med3D checkpoint: {os.path.getsize(SAM_CKPT)/1e6:.0f} MB')
else:
    print(f'✅ SAM-Med3D checkpoint exists: {os.path.getsize(SAM_CKPT)/1e6:.0f} MB')

# ──────────────────────────────────────────────────────────────────────
# Cell 7: Run Data Pipeline
# ──────────────────────────────────────────────────────────────────────

# ─── Configuration ───
MODE = 'dummy'  # 'dummy' or 'totalsegmentator'
MAX_SAMPLES = 5
SEED = 42
# ─────────────────────

%cd $REPO_DIR

if MODE == 'dummy':
    !python run_data_pipeline.py --mode dummy --seed $SEED
elif MODE == 'totalsegmentator':
    TS_DIR = os.path.join(REPO_DIR, 'data', 'totalsegmentator')
    os.makedirs(TS_DIR, exist_ok=True)
    !python run_data_pipeline.py --mode totalsegmentator --ts_dir $TS_DIR --max_samples $MAX_SAMPLES --seed $SEED

print('\n✅ Data pipeline complete!')

# ──────────────────────────────────────────────────────────────────────
# Cell 8: Inspect Outputs
# ──────────────────────────────────────────────────────────────────────

import json, os

print('=== Trajectories ===')
traj_dir = os.path.join(REPO_DIR, 'data', 'trajectories')
if os.path.exists(traj_dir):
    for f in os.listdir(traj_dir):
        if f.endswith('.json'):
            with open(os.path.join(traj_dir, f)) as fh:
                traj = json.load(fh)
            print(f'  {f}: {len(traj)} turns')

print(f'\n=== Images ===')
img_dir = os.path.join(REPO_DIR, 'data', 'images')
if os.path.exists(img_dir):
    pngs = [f for f in os.listdir(img_dir) if f.endswith('.png')]
    print(f'  {len(pngs)} PNG files')

print(f'\n=== SFT Dataset ===')
sft_path = os.path.join(REPO_DIR, 'data', 'sft_dataset.jsonl')
if os.path.exists(sft_path):
    with open(sft_path) as f:
        count = sum(1 for _ in f)
    print(f'  {count} SFT samples')

# ──────────────────────────────────────────────────────────────────────
# Cell 9: SFT Training (Optional — requires GPU)
# ──────────────────────────────────────────────────────────────────────

# ─── Configuration ───
RUN_SFT = False  # Set to True to run SFT training
SFT_OUTPUT = os.path.join(REPO_DIR, 'models', 'sft_output')
# ─────────────────────

if RUN_SFT:
    from training.sft_trainer import train_sft
    from transformers import BitsAndBytesConfig
    import torch

    # 4-bit QLoRA config to fit T4 (16 GB VRAM)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4'
    )

    print('Starting SFT training with 4-bit QLoRA...')
    train_sft(
        dataset_path=os.path.join(REPO_DIR, 'data/sft_dataset.jsonl'),
        model_id=QWEN_DIR,
        output_dir=SFT_OUTPUT,
        seed=42,
        quantization_config=bnb_config
    )
    print(f'✅ SFT training complete. Model saved to {SFT_OUTPUT}')
else:
    print('Skipping SFT. Set RUN_SFT = True to enable.')

# ──────────────────────────────────────────────────────────────────────
# Cell 10: RL Fine-Tuning (Optional — requires A100 or similar)
# ──────────────────────────────────────────────────────────────────────

RUN_RL = False  # Set to True to run RL training
RL_OUTPUT = os.path.join(REPO_DIR, 'models', 'rl_output')

if RUN_RL:
    from training.rl_trainer import train_rl

    print('Starting RL fine-tuning...')
    train_rl(
        sft_model_path=SFT_OUTPUT if RUN_SFT else QWEN_DIR,
        output_dir=RL_OUTPUT
    )
    print(f'✅ RL training complete. Model saved to {RL_OUTPUT}')
else:
    print('Skipping RL. Set RUN_RL = True to enable.')
