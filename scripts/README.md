# Support scripts

Use the root `run.py` launcher:

| Implementation | Launcher command |
|---|---|
| `bootstrap.py` | `doctor` or `setup` |
| `validate_checkpoint.py` | `validate` |
| `build_colab_bundle.py` | `bundle` |
| `colab_remote.py` | `remote` (optional Linux/macOS Colab CLI route) |
| `prepare_sam.py` | Called by setup to preserve the SAM import compatibility fix |

`legacy/` contains the former single/batch wrappers, retained for compatibility. New work should use `run.py single` and `run.py batch`.
