# VS Code + Colab notebook

Open `colab.ipynb` locally in VS Code. Select **Kernel → Colab → GPU server** and sign in through the extension.

1. Right-click `outputs/bundles/MedSAM-Agent3D-colab.zip` in local Explorer → **Upload to Colab**. If missing, enable the extension's **Colab: Uploading** setting.
2. Run the six code cells in order. Configuration lives in the first code cell.
3. Inspect the mask and turn metrics in the notebook. Download the result ZIP using **Colab → Contents**.

`colab_runner.py` is the paired Python cell source, retained for debugging and checkpoint-selection regression tests. It uses remote `/content` paths and is not intended to run in your local Windows Python.

After editing application code locally, rebuild the ZIP from the project root with `python run.py bundle`, upload it again and rerun from the unpacking cell. No WSL or terminal-to-Colab CLI is involved.

Extension instructions: https://github.com/googlecolab/colab-vscode/wiki/User-Guide

The default experiment uses `Qwen/Qwen3.8-27B`, a native vision-language model, with four-bit loading on an A100 40GB or larger. It uses the Qwen3.5 architecture and a newer Transformers profile (`requirements/qwen38.txt`). The supplied case completed on A100 40GB with Dice 0.857 and IoU 0.750; this is a single-case result. All grounding, native tool prompting and parsing fixes are included in the main bundle. No patch ZIP or patch cell is needed. The unquantized checkpoint download is roughly 54 GB; quantization happens during loading. SAM smoke execution and overlap must pass before the agent starts. The notebook rejects bundles predating the panel grounding repair and prints the uploaded ZIP hash, so an unchanged remote ZIP cannot silently repeat the old experiment.

## Broader development evaluation

The default `EVALUATION_MODE="suite"` runs 12 scan/target pairs: two new test-split subjects, each evaluated on liver, spleen, both kidneys and both femurs. Subjects are selected deterministically with seed 42, exclude the original s0000 control, and require nonempty masks for all six structures. This is a development check, not an untouched final benchmark. No reference-derived points are supplied to Qwen; ground truth only scores autonomous predictions.

Upload the rebuilt main bundle and Run All on A100. The SAM smoke remains the original control. Qwen and SAM load once for the batch. The final cell shows every score, averages by organ, failure counts, and a downloadable ZIP with `case_scores.csv`, predictions and trajectories. Set `EVALUATION_MODE="single"` to use a custom scan or rerun the original case. Fixed seed 42, greedy decoding and five actions per case match the previous run; no case is dropped for poor quality.
