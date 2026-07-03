# MedSAM-Agent3D

**MLLM-driven agentic loop for 3D medical segmentation with frozen SAM-Med3D.**

A multimodal LLM (Qwen3-VL) looks at tri-view 2D slices from a 3D volume, reasons about segmentation quality, and iteratively calls SAM-Med3D with 3D prompt coordinates — no training required to run the loop.

---

## Current Focus: The Agentic Loop

```
[3D Medical Volume]
       ↓
[Multi-View Interface] → extracts Axial | Coronal | Sagittal slices
       ↓
[MLLM - Qwen3-VL] ← sees tri-view image + text instruction
       ↓
[JSON Tool Call] → add_3d_bbox / add_3d_point / stop_action
       ↓
[SAM-Med3D - FROZEN] → receives 3D prompt → outputs 3D mask
       ↓
[Low-Res Logits] → propagated to next turn for iterative refinement
       ↓
[Updated Mask] → re-render tri-view → feed back to MLLM
       ↓
[Loop until stop_action]
```

The MLLM is **not fine-tuned** — this works with a vanilla instruct model out of the box.

---

## Quick Start

```bash
# Single case
python agent_inference.py \
    --volume data/some_volume.nii.gz \
    --mask data/some_mask.nii.gz \
    --target liver \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --max-turns 5

# Batch evaluation on TotalSegmentator
python eval_batch.py \
    --ts-dir Totalsegmentator_dataset_v201 \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --max-samples 10 \
    --max-turns 5
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | `Qwen/Qwen3-VL-8B-Instruct` | HuggingFace model ID or local path |
| `--sam-ckpt` | `sam_med3d_turbo.pth` | Path to frozen SAM-Med3D checkpoint |
| `--max-turns` | `5` | Max MLLM → SAM-Med3D iterations |
| `--temperature` | `0.1` | MLLM sampling temp (0 = greedy) |
| `--load-in-4bit` | off | 4-bit quantized MLLM |
| `--output-dir` | `eval_outputs` | Tri-view snapshots + metrics |

---

## Architecture

### 1. Multi-View Interface

Bridges the dimensionality gap: SAM-Med3D works in 3D, the MLLM sees 2D.

- Extracts three orthogonal slices from the 3D volume (axial, coronal, sagittal)
- Centers on the current **focus point**:
  - Turn 0: volume center
  - Turn 1+: centroid of the current predicted mask
- Overlays the predicted mask as semi-transparent green on each slice
- Stitches into one image: `[Axial | Coronal | Sagittal]`

### 2. Action Space

The MLLM emits structured JSON tool calls with normalized `[0, 999]` coordinates
(following MedSAM-Agent's convention for volume-size invariance):

```json
// Initialize with a 3D bounding box
{"name": "add_3d_bbox", "arguments": {"bbox_3d": [z1, y1, x1, z2, y2, x2]}}

// Refine with a 3D point
{"name": "add_3d_point", "arguments": {"point_3d": [z, y, x], "point_type": "positive"}}

// Stop the refinement loop
{"name": "stop_action"}
```

Coordinates are denormalized to voxel space inside the agent loop (`norm/999 * dim`).

### 3. Agent Loop (`agent_inference.py`)

At each turn:
1. Render tri-view centered on mask centroid → save as PNG
2. Feed tri-view + instruction to Qwen3-VL
3. Parse JSON tool call from model output
4. **Bbox turn**: pass 3D bbox → SAM-Med3D → get initial mask + low-res logits
5. **Point turn**: pass latest point + previous logits → SAM-Med3D → refined mask
6. Compute IoU/Dice against ground truth (if available)
7. Update conversation history (text-only to avoid VRAM blowup)
8. Loop until `stop_action` or max turns

Iterative refinement follows the MedSAM-Agent pattern: first call uses `no_mask_embed`
in the prompt encoder; subsequent calls pass prior low-res logits for proper
iterative refinement. Bounding boxes use the newly fixed 3D box embeddings in
SAM-Med3D's prompt encoder.

---

## Project Structure

```
├── agent_inference.py         # ★ Agent loop (the main workflow)
├── eval_batch.py              # Batch evaluation on TotalSegmentator
├── models/
│   └── sam_med3d_wrapper.py   # FrozenSAM-Med3D — torchio preprocessing + coordinate mapping
├── utils/
│   └── io_utils.py            # NIfTI loading & dataset parsing
├── sam_med3d_turbo.pth        # SAM-Med3D checkpoint (frozen)
├── SAM-Med3D/                 # Reference SAM-Med3D repo (modified prompt encoder)
├── MedSAM-Agent/              # Reference 2D MedSAM-Agent code
├── requirements.txt
└── README.md
```

---

## Requirements

Install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

**Key dependencies:** PyTorch, Transformers, Qwen3-VL, nibabel, torchio, SAM-Med3D, OpenCV, scipy.

---

## What Is NOT Built Yet

- ❌ Navigation / scroll tools (scrolling through z-stacks)
- ❌ MIP-based blind search
- ❌ Full autonomous clinical deployment loop
- ❌ Modifications to SAM-Med3D weights (it stays frozen)

---

## References

- [SAM-Med3D](https://github.com/uni-medical/SAM-Med3D)
- [MedSAM-Agent](https://github.com/bowang-lab/MedSAM-Agent)
- [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
