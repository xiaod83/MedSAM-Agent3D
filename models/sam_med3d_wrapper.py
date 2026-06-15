"""
Wrapper around SAM-Med3D Turbo model.
Uses torchio preprocessing (matching official pipeline) with proper
coordinate mapping using affine transforms.
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torchio as tio

_SAM_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'SAM-Med3D')
if _SAM_DIR not in sys.path:
    sys.path.insert(0, _SAM_DIR)

from segment_anything.build_sam3D import build_sam3D_vit_b_ori

CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sam_med3d_turbo.pth')
CROP_SIZE = 128
TARGET_SPACING = (1.5, 1.5, 1.5)


def _load_turbo_checkpoint(path, device):
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    return ckpt.get('model_state_dict', ckpt)


class FrozenSAMMed3D:

    def __init__(self, checkpoint_path=CHECKPOINT_PATH):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"SAM-Med3D checkpoint not found at {checkpoint_path}.")

        print("Loading SAM-Med3D Turbo...")
        sam = build_sam3D_vit_b_ori()
        state_dict = _load_turbo_checkpoint(checkpoint_path, 'cpu')
        sam.load_state_dict(state_dict, strict=True)
        self.model = sam.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self._features = None
        self._orig_shape = None
        self._roi_shape = None
        self._roi_affine = None
        self._preprocess_subject = None
        print(f"SAM-Med3D loaded ({self.device}).")

    def set_image(self, volume_3d, gt_mask=None):
        """Torchio preprocessing pipeline."""
        vol = volume_3d.astype(np.float32)
        self._orig_shape = vol.shape

        # Build subject and store original subject for later coordinate mapping
        vol_tensor = torch.from_numpy(vol).unsqueeze(0).float()
        subject = tio.Subject(image=tio.ScalarImage(tensor=vol_tensor))

        if gt_mask is not None and gt_mask.sum() > 0:
            gt_tensor = torch.from_numpy(gt_mask.astype(np.float32)).unsqueeze(0)
            subject.add_image(tio.LabelMap(tensor=gt_tensor), 'label')

        # ToCanonical (reorient to RAS+) + Resample + CropOrPad
        # ToCanonical ensures NIfTI files with non-standard orientations
        # (e.g., LPS+) produce correctly aligned crops.
        subject = tio.ToCanonical()(subject)
        subject = tio.Resample(TARGET_SPACING)(subject)
        cp_kwargs = dict(target_shape=(CROP_SIZE, CROP_SIZE, CROP_SIZE))
        if 'label' in subject:
            cp_kwargs['mask_name'] = 'label'
        subject = tio.CropOrPad(**cp_kwargs)(subject)
        self._roi_affine = subject.image.affine  # affine of the ROI crop

        self._roi_shape = tuple(subject.image.data.shape[1:])

        # ZNormalization on ROI
        roi = subject.image.data.numpy().astype(np.float32)
        tissue = roi > 0
        mu = roi[tissue].mean() if tissue.any() else roi.mean()
        sigma = roi[tissue].std() + 1e-8 if tissue.any() else roi.std() + 1e-8
        roi_norm = (roi - mu) / sigma

        roi_tensor = torch.from_numpy(roi_norm).unsqueeze(0).to(self.device)
        with torch.no_grad():
            self._features = self.model.image_encoder(roi_tensor)

        # Store the processed subject and original subject for mask back-projection
        self._roi_subject = subject
        self._orig_subject = tio.Subject(image=tio.ScalarImage(
            tensor=torch.from_numpy(vol).unsqueeze(0).float()
        ))

        print(f"Features: {self._orig_shape} -> ROI {self._roi_shape}")

    def predict_mask(self, points, labels):
        """Predict mask for points in original (z,y,x) coordinates."""
        if self._features is None:
            raise RuntimeError("Call set_image() first")

        device = self._features.device

        # Map all points from original (z,y,x) -> ROI (x,y,z) pixel coords
        # Uses the stored ROI affine for exact mapping through the full
        # ToCanonical -> Resample -> CropOrPad chain.
        # Original affine is identity: world(x,y,z) = original(x,y,z).
        # ROI affine maps ROI voxel(z,y,x) -> world(x_ras, y_ras, z_ras).
        # So: ROI voxel = inv(roi_affine) @ [x_orig, y_orig, z_orig, 1]
        pts = np.array(points, dtype=np.float32)
        pts_xyz = np.zeros((0, 3), dtype=np.float32)  # safe default for empty points
        if len(pts) > 0:
            inv_roi = np.linalg.inv(self._roi_affine)
            pts_roi = []
            for p in pts:
                # p = (z, y, x) in original array coords
                # Original identity affine: world = [x, y, z] = [p[2], p[1], p[0]]
                world = np.array([p[2], p[1], p[0], 1.0])
                roi_vox = inv_roi @ world  # (x_roi, y_roi, z_roi, 1)
                # SAM expects (x, y, z) order
                pts_roi.append([roi_vox[0], roi_vox[1], roi_vox[2]])
            pts_xyz = np.clip(np.array(pts_roi, dtype=np.float32), 0, CROP_SIZE - 1)

        zero_mask = torch.zeros(1, 1, CROP_SIZE, CROP_SIZE, CROP_SIZE, device=device)
        coords = torch.tensor(pts_xyz, dtype=torch.float32, device=device).unsqueeze(0)
        lbls = torch.tensor([labels], device=device)

        sparse_emb, dense_emb = self.model.prompt_encoder(
            points=(coords, lbls), boxes=None, masks=zero_mask)

        with torch.no_grad():
            masks, _ = self.model.mask_decoder(
                image_embeddings=self._features,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
            )

        # Upsample to ROI size
        m = F.interpolate(masks, size=self._roi_shape,
                          mode='trilinear', align_corners=False)
        binary_roi = (torch.sigmoid(m) > 0.5).float()

        # Map ROI binary mask back to original volume space using torchio Resample
        roi_label = tio.LabelMap(tensor=binary_roi.squeeze(0).cpu(), affine=self._roi_affine)
        resampler = tio.Resample(target=self._orig_subject.image,
                                  image_interpolation='nearest')
        back_projected = resampler(roi_label)
        mask = back_projected.data.numpy()[0].astype(np.uint8)

        return mask
