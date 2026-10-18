# Development guide

Use `run.py` at the repository root for execution. `scripts/` contains support implementations, not competing main workflows.

## Application code

- `medsam_agent3d/cli.py`: arguments and single/batch dispatch.
- `medsam_agent3d/agent.py`: Qwen vision-language inference.
- `medsam_agent3d/sam.py`: frozen SAM integration, device selection and geometry.
- `medsam_agent3d/io.py`: NIfTI loading/saving and dataset discovery.
- `medsam_agent3d/loop.py`: rendering, actions, refinement and termination.
- `medsam_agent3d/evaluation.py`: output persistence and provenance.
- `medsam_agent3d/benchmark.py`: direct, oracle and agent comparisons.
- `parsing.py`, `prompting.py`, `rendering.py`, `clicker.py`, `types.py`: supporting components.

## Dependencies and testing

```powershell
python -m pip install -r requirements/dev.txt
python run.py test -q
```

`requirements.txt` includes `requirements/base.txt`. Development requirements add pytest; `requirements/quantized.txt` adds optional four-bit support. Full scientific tests need the base dependencies. `python run.py test --light` uses only Python's standard library and does not validate numerical model inference.

`python run.py doctor` reports the active interpreter, dependency availability, GPU support and checkpoint access. `python run.py validate --help` describes the real-checkpoint smoke test. Its foreground prompt comes from ground truth and does not establish autonomous performance.

## Optional Colab routes

`notebooks/colab.ipynb` opens locally in VS Code and executes on a Colab GPU kernel using Google's extension. Upload the ZIP with the Explorer context menu; no browser-only `files.upload()` or mandatory Drive mount is used. Build its updated source archive with `python run.py bundle`; the archive appears in `outputs/bundles/`. The paired notebook source is `notebooks/colab_runner.py`. The notebook preserves an unrecognized Drive checkpoint and downloads verified official weights rather than stopping on the Drive file.

A separate `python run.py remote` wrapper controls an existing runtime using [Google's Colab CLI](https://github.com/googlecolab/google-colab-cli). That CLI currently supports Linux/macOS, so this optional route cannot run natively in Windows PowerShell. It is not needed for normal project navigation or local tests. Authentication and live remote execution remain unverified from this restricted environment.

## Preserved material

The two upstream repositories remain at the root to preserve their nested Git state and local modifications. Papers are in `docs/references/`. Existing experimental artifacts are under `outputs/experiments/`; historical paths inside them were preserved as provenance. Earlier training code remains in Git history.

Cleanup changed paths and entry points, not model weights or segmentation logic. The current checkpoint remained at the root because the filesystem denied moving it; SAM defaults prefer `checkpoints/` and fall back to the root file.
