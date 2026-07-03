"""
Minimal NIfTI I/O utilities for loading 3D medical volumes.

Extracted from totalsegmentator_loader (no training dependencies).
"""

import os
import glob
import numpy as np
import nibabel as nib


def load_nifti(file_path):
    """
    Load a NIfTI file and return (data, spacing).

    Returns:
        data: np.ndarray in (z, y, x) order
        spacing: (dz, dy, dx) voxel spacing
    """
    nii = nib.load(file_path)
    data = nii.get_fdata()
    zooms = nii.header.get_zooms()  # (dx, dy, dz, ...)

    spacing = zooms[:3] if len(zooms) >= 3 else (1.0, 1.0, 1.0)
    spacing = tuple(abs(float(s)) for s in spacing)

    # NIfTI is (x, y, z) → transpose to (z, y, x)
    data = np.transpose(data, (2, 1, 0))
    spacing = (spacing[2], spacing[1], spacing[0])

    return data, spacing


def get_totalsegmentator_dataset(base_dir):
    """
    Parse a TotalSegmentator dataset directory.

    Expected structure:
        base_dir/
          s0001/
            ct.nii.gz (or mri.nii.gz)
            segmentations/
              liver.nii.gz
              spleen.nii.gz
              ...

    Returns list of dicts with keys:
        volume_id, target_name, ct_path, mask_path
    """
    dataset = []

    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist.")
        return dataset

    subject_dirs = [d for d in os.listdir(base_dir)
                    if os.path.isdir(os.path.join(base_dir, d))]

    for subj in subject_dirs:
        subj_path = os.path.join(base_dir, subj)
        vol_path = os.path.join(subj_path, "mri.nii.gz")
        if not os.path.exists(vol_path):
            vol_path = os.path.join(subj_path, "ct.nii.gz")
            if not os.path.exists(vol_path):
                continue

        seg_dir = os.path.join(subj_path, "segmentations")
        if not os.path.exists(seg_dir):
            continue

        masks = glob.glob(os.path.join(seg_dir, "*.nii.gz"))

        for mask_path in masks:
            target_name = os.path.basename(mask_path).replace(".nii.gz", "")
            dataset.append({
                "volume_id": subj,
                "target_name": target_name,
                "ct_path": vol_path,
                "mask_path": mask_path,
            })

    return dataset
