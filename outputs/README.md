# Outputs

- `inference/`: local and retrieved predictions, masks, metrics and logs.
- `tests/`: existing synthetic test artifacts and future smoke-test output.
- `validation/`: checkpoint validation reports and environment diagnostics.
- `experiments/`: preserved early images, trajectories and SFT-format records. Paths inside those historical records refer to the layout used when they were generated.
- `archive/`: older generated handoff bundles, preserved intact.
- `bundles/`: optional Colab handoff archives; rebuild with `python run.py bundle`.

These are generated or historical artifacts, not application source. Old experiment scores and mock scores are not validated autonomous model results.
