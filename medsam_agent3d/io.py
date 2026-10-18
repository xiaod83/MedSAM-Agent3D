"""NIfTI I/O: arrays are canonical RAS in Z,Y,X order; affines map X,Y,Z."""
import csv
import glob
import os
import numpy as np
import nibabel as nib


def load_nifti(file_path, return_affine=False, reference_path=None, is_label=False):
    """Load canonical RAS data, spacing, and optionally its XYZ-to-world affine.

    Masks should specify reference_path and is_label=True: this aligns them to
    the image grid using nearest-neighbor interpolation, including oblique data.
    """
    image = nib.as_closest_canonical(nib.load(file_path))
    if len(image.shape) != 3:
        raise ValueError(f'Expected a 3D NIfTI volume: {file_path} has shape {image.shape}')
    if reference_path is not None:
        reference = nib.as_closest_canonical(nib.load(reference_path))
        if image.shape != reference.shape or not np.allclose(image.affine, reference.affine):
            from nibabel.processing import resample_from_to
            image = resample_from_to(image, reference, order=0 if is_label else 1)
    data = np.asarray(image.get_fdata(dtype=np.float32)).transpose(2, 1, 0).copy()
    spacing = tuple(float(s) for s in nib.affines.voxel_sizes(image.affine)[::-1])
    result = (data, spacing)
    return (*result, image.affine.copy()) if return_affine else result


def save_nifti(data, file_path, affine=None, reference_path=None):
    """Save canonical ZYX data, optionally returning it to a reference's grid."""
    array = np.asarray(data)
    if array.ndim != 3:
        raise ValueError('NIfTI output must be a 3D ZYX array')
    reference = nib.load(reference_path) if reference_path is not None else None
    if affine is None:
        if reference is None:
            raise ValueError('Provide affine or reference_path to preserve physical geometry')
        affine = nib.as_closest_canonical(reference).affine
    image = nib.Nifti1Image(array.transpose(2, 1, 0), affine)
    if reference is not None and (image.shape != reference.shape or not np.allclose(image.affine, reference.affine)):
        from nibabel.processing import resample_from_to
        image = resample_from_to(image, reference, order=0)
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    nib.save(image, file_path)


def get_totalsegmentator_dataset(base_dir, split=None):
    """Return sorted organ entries; optional split is read strictly from meta.csv.

    Both volume_path and the historical ct_path refer to the source volume.
    Requesting a split without metadata raises rather than silently leaking cases.
    """
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f'Dataset directory does not exist: {base_dir}')
    membership = None
    if split is not None:
        if split not in ('train', 'val', 'test'):
            raise ValueError('split must be train, val, test, or None')
        meta_path = os.path.join(base_dir, 'meta.csv')
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f'Split selection requires {meta_path}')
        with open(meta_path, newline='', encoding='utf-8-sig') as handle:
            sample = handle.read(4096)
            handle.seek(0)
            rows = csv.DictReader(handle, delimiter=';' if ';' in sample.splitlines()[0] else ',')
            membership = {row['image_id'] for row in rows if row.get('split') == split}
    dataset = []
    for subject in sorted(os.listdir(base_dir)):
        if membership is not None and subject not in membership:
            continue
        root = os.path.join(base_dir, subject)
        volume = next((os.path.join(root, name) for name in ('mri.nii.gz', 'ct.nii.gz') if os.path.isfile(os.path.join(root, name))), None)
        if volume is None:
            continue
        for mask in sorted(glob.glob(os.path.join(root, 'segmentations', '*.nii.gz'))):
            dataset.append({'volume_id': subject, 'target_name': os.path.basename(mask)[:-7], 'volume_path': volume, 'ct_path': volume, 'mask_path': mask})
    return dataset
