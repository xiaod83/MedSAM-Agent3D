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

`SAM-Med3D/` is the upstream runtime dependency. The unused MedSAM-Agent reference checkout, extracted paper notes and retired launcher wrappers are preserved locally in the ignored `outputs/archive/portfolio-cleanup/` directory. They are not required to run the project. Existing experimental artifacts are under `outputs/experiments/`; historical paths inside them were preserved as provenance.

Generated notebook ZIP downloads are kept locally in `outputs/archive/` and excluded from future source commits. Rebuild the current upload bundle with `python run.py bundle` rather than keeping timestamped ZIPs in the repository root.

SAM defaults prefer `checkpoints/sam_med3d_turbo.pth` and fall back to the root file for compatibility. Model weights, downloaded scan arrays, environments and caches are excluded from Git.

## Working with the nested repositories

`SAM-Med3D/` has its own Git history. The main repository records an exact commit pointer, rather than storing its files as ordinary project source. Committing a new pointer does not upload the referenced commit to the nested repository's remote.

Use `git submodule status` to inspect the pointers. A pointer committed to the main repository must be fetchable from the URL in `.gitmodules`, or GitHub Actions and fresh recursive clones will fail during checkout. Local modifications inside a nested repository are not included by committing its pointer.

Run `git status` at the root and `git -C SAM-Med3D status` before committing. Keep project changes and upstream repository changes separate, and avoid recursive pushes unless you intend to publish the nested changes too.
