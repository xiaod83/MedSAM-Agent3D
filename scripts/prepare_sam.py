"""Apply the minimal SAM import compatibility fix without altering other edits.

Run after `git submodule update --init --recursive`. This operation is idempotent.
The upstream absolute `from utils import *` resolves to this project's utils and
is unnecessary: SAM imports its own helpers explicitly using relative imports.
"""
from pathlib import Path


def main():
    target = Path(__file__).resolve().parents[1] / "SAM-Med3D/segment_anything/__init__.py"
    if not target.is_file():
        raise SystemExit("Initialize SAM-Med3D first: git submodule update --init --recursive")
    content = target.read_text(encoding="utf-8")
    unwanted = "from utils import *"
    lines = content.splitlines(keepends=True)
    patched = "".join(line for line in lines if line.strip() != unwanted)
    if patched != content:
        target.write_text(patched, encoding="utf-8")
        print("Removed upstream absolute utils wildcard import.")
    else:
        print("SAM import compatibility fix already applied.")


if __name__ == "__main__":
    main()
