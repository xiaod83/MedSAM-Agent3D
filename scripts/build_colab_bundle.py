"""Build the complete Colab handoff, including one real CT/reference-mask pair.

The bundle contains source and a selected local sample only. It excludes the
checkpoint, credentials, caches, Git metadata, and the rest of the dataset.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/bundles/MedSAM-Agent3D-colab.zip')
    args = parser.parse_args(argv)
    files = set()
    for folder in ('medsam_agent3d', 'tests', 'scripts', 'SAM-Med3D/segment_anything'):
        files.update(path for path in (ROOT / folder).rglob('*.py') if '__pycache__' not in path.parts)
    for name in ('run.py', 'requirements/base.txt', 'notebooks/colab_runner.py',
                 'notebooks/colab.ipynb', 'requirements.txt', 'requirements/dev.txt',
                 'requirements/quantized.txt', 'requirements/qwen38.txt', 'pytest.ini', 'README.md', 'docs/DEVELOPMENT.md',
                 'notebooks/README.md', 'scripts/README.md', 'data/README.md', 'checkpoints/README.md',
                 'data/Totalsegmentator_dataset_v201/s0000/ct.nii.gz',
                 'data/Totalsegmentator_dataset_v201/s0000/segmentations/femur_left.nii.gz'):
        path = ROOT / name
        if not path.is_file():
            raise FileNotFoundError(f'Bundle input missing: {path}')
        files.add(path)
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    manifest = {'runtime_profile': 'qwen38-native-tools-v1', 'source_revision': revision, 'contains_uncommitted_repairs': True,
                'validation_case': 's0000/femur_left', 'sample_is_smoke_test_not_held_out_benchmark': True,
                'checkpoint_download': {
                    'url': 'https://huggingface.co/blueyo0/SAM-Med3D/resolve/main/sam_med3d_turbo.pth',
                    'sha256': '899a46d04d3b70f723282ceb489149373558bf0aaba389a346f5ab57da5cdd3c'},
                'files': {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.partial')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
        for path in sorted(files):
            data = path.read_bytes()
            relative = path.relative_to(ROOT).as_posix()
            archive.writestr(relative, data)
            manifest['files'][relative] = hashlib.sha256(data).hexdigest()
        archive.writestr('bundle_manifest.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(temporary) as archive:
        invalid = archive.testzip()
        if invalid:
            raise RuntimeError(f'Bundle integrity check failed: {invalid}')
    temporary.replace(args.output)
    print(json.dumps({'bundle': str(args.output), 'files': len(files),
                      'size_bytes': args.output.stat().st_size,
                      'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest()}, indent=2))


if __name__ == '__main__':
    main()
