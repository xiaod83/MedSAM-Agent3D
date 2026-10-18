"""Build the complete Colab handoff, including real smoke and development cases.

The bundle contains source, a smoke control and two development-evaluation scans only. It excludes the
checkpoint, credentials, caches, Git metadata, and the rest of the dataset.
"""
import argparse
import csv
import gzip
import math
import random
import struct
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


SUITE_TARGETS = ('liver', 'spleen', 'kidney_left', 'kidney_right', 'femur_left', 'femur_right')


def nifti_has_foreground(path):
    """Stream a scalar NIfTI-1 label volume and check actual nonzero voxels.

    Compressed file size is not evidence of foreground: empty masks compress
    well too. This parser requires no scientific dependencies and stops at the
    first nonzero voxel. The bundled dataset uses scalar integer label maps.
    """
    with gzip.open(path, 'rb') as handle:
        header = handle.read(348)
        if len(header) != 348:
            raise ValueError(f'Truncated NIfTI header: {path}')
        endian = '<' if struct.unpack('<i', header[:4])[0] == 348 else '>'
        if struct.unpack(endian + 'i', header[:4])[0] != 348 or header[344:348] != b'n+1\x00':
            raise ValueError(f'Expected a single-file NIfTI-1 label map: {path}')
        dimensions = struct.unpack_from(endian + '8h', header, 40)
        if dimensions[0] != 3 or any(size <= 0 for size in dimensions[1:4]):
            raise ValueError(f'Expected a nonempty 3D label map: {path}')
        datatype, bitpix = struct.unpack_from(endian + '2h', header, 70)
        formats = {2: 'B', 4: 'h', 8: 'i', 16: 'f', 64: 'd', 256: 'b', 512: 'H', 768: 'I', 1024: 'q', 1280: 'Q'}
        if datatype not in formats:
            raise ValueError(f'Unsupported NIfTI label datatype {datatype}: {path}')
        fmt = endian + formats[datatype]
        size = struct.calcsize(fmt)
        if bitpix != size * 8:
            raise ValueError(f'Inconsistent NIfTI label bit depth: {path}')
        offset = struct.unpack_from(endian + 'f', header, 108)[0]
        if not math.isfinite(offset) or offset < 352 or offset != int(offset):
            raise ValueError(f'Invalid NIfTI data offset: {path}')
        # Label maps must have identity intensity scaling. NIfTI slope=0/NaN
        # disables scaling; interpreting raw nonzero bytes is then correct.
        slope, intercept = struct.unpack_from(endian + '2f', header, 112)
        if math.isfinite(slope) and slope != 0 and (slope != 1 or (math.isfinite(intercept) and intercept != 0)):
            raise ValueError(f'Unexpected scaled NIfTI label map: {path}')
        handle.seek(int(offset))
        remaining = math.prod(dimensions[1:4]) * size
        while remaining:
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk or len(chunk) % size:
                raise ValueError(f'Truncated NIfTI voxel data: {path}')
            remaining -= len(chunk)
            if datatype in (16, 64):
                if any(math.isfinite(value) and value != 0 for (value,) in struct.iter_unpack(fmt, chunk)):
                    return True
            elif chunk.strip(b'\x00'):
                return True
    return False


def select_development_suite(dataset_root, seed=42, excluded=('s0000',)):
    """Select two seeded test-split subjects with all six nonempty organs.

    These cases support a development generalization check. Because they guide
    debugging, their results are not a formal held-out benchmark estimate.
    """
    dataset_root = Path(dataset_root)
    with (dataset_root / 'meta.csv').open(newline='', encoding='utf-8-sig') as handle:
        candidates = sorted({row['image_id'] for row in csv.DictReader(handle, delimiter=';')
                             if row.get('split') == 'test' and row['image_id'] not in excluded})
    random.Random(seed).shuffle(candidates)
    selected, checks, files, cases = [], [], set(), []
    for subject in candidates:
        root = dataset_root / subject
        volume = root / 'ct.nii.gz'
        masks = [root / 'segmentations' / f'{target}.nii.gz' for target in SUITE_TARGETS]
        if not volume.is_file() or not all(mask.is_file() for mask in masks):
            checks.append({'volume_id': subject, 'eligible': False, 'reason': 'missing CT or required mask'})
            continue
        empty = []
        for target, mask in zip(SUITE_TARGETS, masks):
            if not nifti_has_foreground(mask):
                empty.append(target)
        if empty:
            checks.append({'volume_id': subject, 'eligible': False, 'empty_targets': empty})
            continue
        selected.append(subject)
        checks.append({'volume_id': subject, 'eligible': True})
        files.update((volume, *masks))
        for target, mask in zip(SUITE_TARGETS, masks):
            cases.append({'volume_id': subject, 'target_name': target,
                          'volume_path': volume.relative_to(ROOT).as_posix(),
                          'mask_path': mask.relative_to(ROOT).as_posix()})
        if len(selected) == 2:
            break
    if len(selected) != 2:
        raise RuntimeError('Could not find two test-split subjects with six nonempty required organ masks')
    metadata = {'purpose': 'development generalization check; not a formal held-out benchmark',
                'selection_protocol': 'Sort unique test-split subject IDs, exclude smoke subject, shuffle with Python Random(seed), take first two with CT and all six nonempty scalar NIfTI label maps.',
                'source_split': 'test', 'split': 'test', 'seed': seed,
                'source_metadata_sha256': hashlib.sha256((dataset_root / 'meta.csv').read_bytes()).hexdigest(), 'excluded_subjects': list(excluded),
                'selected_test_subjects': selected, 'targets': list(SUITE_TARGETS),
                'subject_count': 2, 'case_count': len(cases), 'foreground_validation': 'streamed actual nonzero voxels; no compressed-size heuristic',
                'selection_checks': checks,
                'ct_files': [{'volume_id': subject,
                              'path': (dataset_root / subject / 'ct.nii.gz').relative_to(ROOT).as_posix(),
                              'size_bytes': (dataset_root / subject / 'ct.nii.gz').stat().st_size}
                             for subject in selected]}
    return cases, metadata, files


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
    suite_cases, suite_metadata, suite_files = select_development_suite(ROOT / 'data/Totalsegmentator_dataset_v201')
    files.update(suite_files)
    generated_files = {'suite_cases.json': json.dumps(suite_cases, indent=2).encode('utf-8'),
                       'suite_metadata.json': json.dumps(suite_metadata, indent=2).encode('utf-8')}
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    manifest = {'runtime_profile': 'qwen38-native-tools-v1', 'source_revision': revision, 'contains_uncommitted_repairs': True,
                'validation_case': 's0000/femur_left', 'sample_is_smoke_test_not_held_out_benchmark': True,
                'development_suite': {'manifest': 'suite_cases.json', 'metadata': 'suite_metadata.json',
                                      'case_count': 12, 'selected_test_subjects': suite_metadata['selected_test_subjects'],
                                      'purpose': suite_metadata['purpose']},
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
        for relative, data in sorted(generated_files.items()):
            archive.writestr(relative, data)
            manifest['files'][relative] = hashlib.sha256(data).hexdigest()
        archive.writestr('bundle_manifest.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(temporary) as archive:
        invalid = archive.testzip()
        if invalid:
            raise RuntimeError(f'Bundle integrity check failed: {invalid}')
    temporary.replace(args.output)
    print(json.dumps({'bundle': str(args.output), 'files': len(files) + len(generated_files),
                      'size_bytes': args.output.stat().st_size,
                      'development_suite': suite_metadata,
                      'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest()}, indent=2))


if __name__ == '__main__':
    main()
