# Evaluation and limitations

## Reported development result

The project README and notebook notes previously recorded a completed left-femur experiment with Dice **0.857** and IoU **0.750**, using Qwen3.8-27B with four-bit loading on an A100 40 GB GPU. These values are retained as a reported result; the repository cleanup did not rerun inference or independently verify the original outputs.

One case does not establish generalization. It should not be compared directly with published dataset averages or presented as a clinical validation result.

## Development suite

The bundle builder selects two additional test-split subjects deterministically with seed 42, excluding the original `s0000` control. Each subject must have nonempty masks for liver, spleen, left and right kidney, and left and right femur. This produces 12 scan/target pairs.

The notebook's default `EVALUATION_MODE="suite"` evaluates those pairs with five agent actions per case. Reference masks score predictions; they do not supply the autonomous agent's prompt points. The separate SAM smoke test uses the original control and a reference-derived prompt to check checkpoint execution.

The suite is a development evaluation, not an untouched final benchmark. Its results must be inspected before reporting averages. The notebook exports per-case scores, masks and trajectories; retain failures when summarizing performance.

## Reproduction

1. Follow the [notebook guide](../notebooks/README.md) and obtain the required dataset and checkpoint.
2. Run `python run.py bundle` to generate the source and case bundle.
3. Upload the bundle to a Colab GPU runtime and run the notebook cells in order.
4. Preserve the result ZIP, bundle hash, model configuration, subject IDs, per-case scores and failure counts.

Aggregate suite scores are not included here because a completed suite result has not been established in the repository documentation.

## What software tests establish

`python run.py test --light` checks selected behavior without ML dependencies. The full pytest suite exercises additional numerical, geometry and pipeline behavior. Mock tests verify software behavior, not model segmentation quality. Passing these checks does not substitute for evaluation with real scans and checkpoints.
