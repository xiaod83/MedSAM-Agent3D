import os
import glob
import numpy as np
import nibabel as nib

def get_totalsegmentator_dataset(base_dir):
    """
    Parses a TotalSegmentator dataset directory and returns a list of dictionaries 
    containing paths to the CT/MRI volume and its corresponding masks.
    
    Expected structure:
    base_dir/
      s0001/
        mri.nii.gz
        segmentations/
          liver.nii.gz
          spleen.nii.gz
          ...
    """
    dataset = []
    
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist.")
        return dataset
        
    subject_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for subj in subject_dirs:
        subj_path = os.path.join(base_dir, subj)
        # Check for mri.nii.gz or ct.nii.gz
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
                "mask_path": mask_path
            })
            
    return dataset

def load_nifti(file_path):
    """
    Loads a NIfTI file and returns its data as a numpy array along with its voxel spacing.
    Rearranges axes from (x, y, z) to (z, y, x) for compatibility with our pipeline.
    """
    nii = nib.load(file_path)
    data = nii.get_fdata()
    spacing = nii.header.get_zooms() # typically (dx, dy, dz)
    
    # Typical NIfTI is (x, y, z). We want (z, y, x)
    data = np.transpose(data, (2, 1, 0))
    # Match spacing to (z, y, x)
    if len(spacing) >= 3:
        spacing = (spacing[2], spacing[1], spacing[0])
    else:
        spacing = (1.0, 1.0, 1.0)
        
    return data, spacing