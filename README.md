# Start here

Run everything from this project folder using **`run.py`**. Implementation files are in `medsam_agent3d/`; you do not need to execute them individually.

In Windows PowerShell:

```powershell
cd C:\Users\zhouq\MedSAM-Agent3D
.\.venv\Scripts\python.exe run.py --help
.\.venv\Scripts\python.exe run.py doctor
.\.venv\Scripts\python.exe run.py test --light
```

If your normal Python environment is activated, use `python run.py` instead.

## What to run

| Task | Command after `python run.py` |
|---|---|
| Check dependencies, GPU and checkpoint access | `doctor` |
| Install a CPU environment and run software checks | `setup --mode cpu` |
| Run tests without ML dependencies | `test --light` |
| Run the full regression suite | `test -q` |
| Segment one scan | `single --volume scan.nii.gz --target liver` |
| Evaluate the dataset | `batch --ts-dir data/Totalsegmentator_dataset_v201` |
| Compare methods | `benchmark --ts-dir data/Totalsegmentator_dataset_v201 --method all` |
| Validate the actual SAM checkpoint | `validate --sam-device cpu` |

Add `--mask path/to/mask.nii.gz` to single-case inference to calculate reference metrics. Run a command with `--help` for its options.

## Folder map

| Location | Purpose |
|---|---|
| `run.py` | The entry point for all commands |
| `medsam_agent3d/` | Application source, including SAM integration and NIfTI I/O |
| `scripts/` | Setup and support tools, called through `run.py` |
| `requirements/` | Base, development and optional quantization dependencies |
| `tests/` | Regression and mock tests |
| `data/Totalsegmentator_dataset_v201/` | Existing scan and segmentation dataset |
| `checkpoints/` | Preferred location for model weights; see its README |
| `outputs/` | Predictions, test artifacts, validation reports and old experiments |
| `notebooks/` | Optional Colab notebook and paired script |
| `docs/` | Development notes and reference papers |
| `SAM-Med3D/` | Upstream SAM dependency; its existing edits are preserved |
| `MedSAM-Agent/` | Upstream reference repository |
| `.venv/`, `.cache/` | Local environment and cached tools |

## Current execution status

The lightweight tests run locally. Full inference still requires the scientific Python dependencies and readable model weights. This session blocks package downloads and checkpoint reads; the local MX110 also has only 2 GB of VRAM. Folder cleanup does not remove those execution limits.

**You do not need WSL for this folder, the launcher, or local tests.** The optional terminal-to-Colab CLI route currently requires Linux/macOS; it is not the default workflow. The separate Colab notebook is under `notebooks/`.

Developer details and optional remote execution: [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).
