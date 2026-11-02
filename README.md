# MedSAM-Agent3D

Qwen inspects CT slices, places prompts, and refines a frozen SAM-Med3D mask. Ground truth scores agent predictions; a separate SAM smoke test uses reference-selected clicks.

## Run in VS Code + Colab

1. Open `notebooks/colab.ipynb` and select a **Colab A100 GPU kernel**.
2. Upload `outputs/bundles/MedSAM-Agent3D-colab.zip` through the VS Code Explorer **Upload to Colab** action, replacing any old remote copy.
3. Run all notebook cells. No separate patches, WSL, or Drive mount are needed.

The default uses **Qwen3.8-27B in four-bit** and evaluates 12 scan/target pairs: two new scans, each with liver, spleen, both kidneys and both femurs. Set `EVALUATION_MODE="single"` in the first cell for a custom scan or the original femur case. Configuration paths refer to the remote Colab machine.

After source changes, rebuild from this folder and upload the new ZIP:

```powershell
.\.venv\Scripts\python.exe run.py bundle
```

Bundle creation needs the local TotalSegmentator dataset, including `meta.csv`. A fresh clone does not include scans or weights. The builder selects two test-split subjects with seed 42, excluding `s0000`, and checks all six masks contain foreground. The notebook downloads and verifies the official SAM checkpoint.

<<<<<<< HEAD
## Find results

- **Colab:** `/content/MedSAM-Agent3D/outputs/notebook/<run-id>/`; download `outputs/notebook_<run-id>.zip` through **Colab ? Contents** before the runtime disappears.
- **Saved local runs:** `outputs/archive/` contains downloaded ZIPs; `outputs/review/` contains extracted images, trajectories and reviews.
- Each suite run exports `case_scores.csv`, all-case and per-organ scores, masks, PNGs, logs and action histories. Execution failures count as zero in the reported averages.

## Local commands
=======
>>>>>>> d634889b28ac97e62c3fcbe9cee2edf5fca3f536

Use Python 3.11 and run commands from this folder. With an activated environment:

```bash
python run.py --help
python run.py doctor
python run.py test --light
python run.py setup --mode cpu
python run.py test -q
python run.py single --volume scan.nii.gz --target liver
```

Add `--mask reference.nii.gz` for metrics. Use `single --help`, `batch --help` or `benchmark --help` for options. GPU inference needs compatible dependencies and sufficient VRAM; the notebook installs `requirements/qwen38.txt`, including the Transformers version used by the successful run.

Local SAM weights belong in `checkpoints/sam_med3d_turbo.pth`; the root-level checkpoint is also supported. Override with `--sam-ckpt PATH`.

## Where things live

| Folder | Contents |
|---|---|
| `medsam_agent3d/` | Model integration, rendering, geometry, agent loop and evaluation |
| `notebooks/` | Main Colab notebook and paired Python cell source |
| `scripts/` | Setup, checkpoint validation and bundle builder, called by `run.py` |
| `tests/` | Regression tests and model mocks |
| `requirements/` | Dependency profiles |
| `data/` | Local scans and reference masks; excluded from Git |
| `checkpoints/` | Local model weights; excluded from Git |
| `outputs/` | Bundles, predictions, results and preserved experiments |
| `SAM-Med3D/` | Upstream dependency; clone with `--recurse-submodules` |

## Evidence and limitations

The saved original left-femur run reached **Dice 0.857 / IoU 0.750** with Qwen3.8-27B on A100 40GB. This is one development case, not a general accuracy claim. The 12-case suite is a development generalization check, not an untouched final benchmark. Its aggregate results have not yet been reviewed here. Nonempty predictions and nonzero overlap alone do not establish accurate segmentation.

## Credits

Built on [SAM-Med3D](https://github.com/uni-medical/SAM-Med3D) and the [MedSAM-Agent](https://github.com/CUHK-AIM-Group/MedSAM-Agent) approach, with Qwen models and TotalSegmentator data. Upstream code, weights and data retain their respective licenses and terms.
