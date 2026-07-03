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

        # SAM-Med3D's checkpoint only has 2 point embeddings (pos/neg).
        # We added 2 more (bbox_tl, bbox_br) in PromptEncoder3D.
        # Initialize missing bbox corner embeddings with small random values.
        embed_dim = sam.prompt_encoder.embed_dim
        for i in range(2, 4):
            key = f'prompt_encoder.point_embeddings.{i}.weight'
            if key not in state_dict:
                state_dict[key] = (torch.randn(1, embed_dim) * 0.02).to('cpu')
                print(f"  Initialized missing {key} with random init")

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

    def _map_original_to_roi(self, point_zyx):
        """Map a single (z,y,x) point from original volume coords to ROI (x,y,z)."""
        inv_roi = np.linalg.inv(self._roi_affine)
        world = np.array([point_zyx[2], point_zyx[1], point_zyx[0], 1.0])
        roi_vox = inv_roi @ world
        return np.clip([roi_vox[0], roi_vox[1], roi_vox[2]], 0, CROP_SIZE - 1)

    def _map_original_bbox_to_roi(self, bbox_zyx):
        """Map 3D bbox (z1,y1,x1,z2,y2,x2) from original -> ROI (x1,y1,z1,x2,y2,z2)."""
        z1, y1, x1, z2, y2, x2 = bbox_zyx
        tl = self._map_original_to_roi((z1, y1, x1))
        br = self._map_original_to_roi((z2, y2, x2))
        return [tl[0], tl[1], tl[2], br[0], br[1], br[2]]

    def predict_mask(self, points=None, labels=None, box_3d=None, mask_logits=None):
        """
        Predict mask from 3D prompts.

        Follows MedSAM-Agent's iterative refinement pattern:
        - First call: points + optional box_3d, with mask_logits=None
          (uses no_mask_embed in prompt encoder).
        - Subsequent calls: only the latest point + previous mask_logits.

        Args:
            points: List of (z, y, x) tuples in original voxel coords.
            labels: List of ints (1=foreground, 0=background).
            box_3d: Optional (z1, y1, x1, z2, y2, x2) in original voxel coords.
            mask_logits: Low-res logits from previous prediction (for iterative refinement),
                         or None on first call.

        Returns:
            (binary_mask_3d, low_res_logits):
                binary_mask_3d: np.ndarray (Z, Y, X) uint8 in original space.
                low_res_logits: torch.Tensor (1, 1, 32, 32, 32) for next iteration.
        """
        if self._features is None:
            raise RuntimeError("Call set_image() first")

        device = self._features.device

        # --- Convert points to ROI (x,y,z) coordinates ---
        if points is not None and len(points) > 0:
            pts_roi = [self._map_original_to_roi(p) for p in points]
            pts_xyz = np.array(pts_roi, dtype=np.float32)
            coords = torch.tensor(pts_xyz, dtype=torch.float32, device=device).unsqueeze(0)
            lbls = torch.tensor([labels], device=device)
            points_tensor = (coords, lbls)
        else:
            points_tensor = None

        # --- Convert box to ROI (x1,y1,z1,x2,y2,z2) coordinates ---
        if box_3d is not None:
            box_roi = self._map_original_bbox_to_roi(box_3d)
            box_tensor = torch.tensor(box_roi, dtype=torch.float32, device=device).unsqueeze(0)
        else:
            box_tensor = None

        # --- Mask logits: None -> no_mask_embed; provided -> embed_masks ---
        # MedSAM-Agent pattern: pass None on first call, prior logits on subsequent.
        if mask_logits is not None:
            masks_input = mask_logits.to(device)
        else:
            masks_input = None

        # --- Prompt encoding ---
        sparse_emb, dense_emb = self.model.prompt_encoder(
            points=points_tensor,
            boxes=box_tensor,
            masks=masks_input,
        )

        # --- Mask decoding ---
        with torch.no_grad():
            low_res_masks, _ = self.model.mask_decoder(
                image_embeddings=self._features,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
            )

        # low_res_masks shape: (1, 1, 32, 32, 32) — raw logits before upscaling.
        # Store these for next iteration.
        low_res_logits = low_res_masks.detach().cpu()

        # Upsample to ROI size, sigmoid, threshold
        m = F.interpolate(low_res_masks, size=self._roi_shape,
                          mode='trilinear', align_corners=False)
        binary_roi = (torch.sigmoid(m) > 0.5).float()

        # Map ROI binary mask back to original volume space
        roi_label = tio.LabelMap(tensor=binary_roi.squeeze(0).cpu(), affine=self._roi_affine)
        resampler = tio.Resample(target=self._orig_subject.image,
                                  image_interpolation='nearest')
        back_projected = resampler(roi_label)
        mask = back_projected.data.numpy()[0].astype(np.uint8)

        return mask, low_res_logits
