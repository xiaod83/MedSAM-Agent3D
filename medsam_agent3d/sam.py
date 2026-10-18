"""SAM-Med3D inference with physical-space preprocessing and movable prompt crops.

Public arrays/points use Z,Y,X; TorchIO and SAM tensors use X,Y,Z, matching
upstream training's argwhere click coordinates. Crops cover 192 mm per axis;
multiple prompted regions are accumulated, not a whole-volume sliding-window model.
"""
import gc
import os
import sys
import warnings
import numpy as np
import torch
import torch.nn.functional as F
import torchio as tio

_SAM_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'SAM-Med3D')
if _SAM_DIR not in sys.path:
    sys.path.insert(0, _SAM_DIR)
from segment_anything.build_sam3D import build_sam3D_vit_b_ori

_ROOT = os.path.dirname(os.path.dirname(__file__))
CHECKPOINT_PATH = os.path.join(_ROOT, 'checkpoints', 'sam_med3d_turbo.pth')
if not os.path.isfile(CHECKPOINT_PATH):
    CHECKPOINT_PATH = os.path.join(_ROOT, 'sam_med3d_turbo.pth')
CROP_SIZE = 128
TARGET_SPACING = (1.5, 1.5, 1.5)


def _select_device(device='auto'):
    requested = str(device or 'auto')
    if requested == 'auto':
        if not torch.cuda.is_available():
            return torch.device('cpu')
        # A 128-cubed 3D ViT needs considerably more memory than its weights.
        # Small laptop GPUs are safer on CPU; auto also retries CUDA OOM on CPU.
        if torch.cuda.get_device_properties(0).total_memory < 4 * 1024 ** 3:
            warnings.warn('SAM auto device selected CPU because GPU memory is below 4 GiB.', RuntimeWarning, stacklevel=2)
            return torch.device('cpu')
        try:
            free_memory, _ = torch.cuda.mem_get_info()
        except (RuntimeError, AttributeError):
            free_memory = None
        if free_memory is not None and free_memory < 2 * 1024 ** 3:
            warnings.warn('SAM auto device selected CPU because available GPU memory is below 2 GiB.', RuntimeWarning, stacklevel=2)
            return torch.device('cpu')
        return torch.device('cuda')
    resolved = torch.device(requested)
    if resolved.type not in ('cpu', 'cuda'):
        raise ValueError('SAM device must be auto, cpu, cuda, or cuda:N')
    if resolved.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was explicitly requested for SAM but is unavailable; use device=cpu or auto')
    return resolved


def _load_turbo_checkpoint(path, device):
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except PermissionError as exc:
        raise PermissionError(f'SAM checkpoint exists but cannot be read: {os.path.abspath(path)}. Run from an environment with permission to read this file.') from exc
    return ckpt.get('model_state_dict', ckpt)


class FrozenSAMMed3D:
    def __init__(self, checkpoint_path=CHECKPOINT_PATH, use_native_boxes=False, device='auto'):
        """Native boxes require explicit opt-in and verified checkpoint training provenance."""
        self.checkpoint_path = os.path.abspath(checkpoint_path)
        self._allow_cpu_fallback = device is None or str(device) == 'auto'
        self.device = _select_device(device)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f'SAM-Med3D checkpoint not found at {checkpoint_path}.')
        state = _load_turbo_checkpoint(checkpoint_path, 'cpu')
        sam = build_sam3D_vit_b_ori()
        has_corner_weights = all(f'prompt_encoder.point_embeddings.{i}.weight' in state for i in (2, 3))
        # Weight presence alone does not establish training provenance: older
        # wrapper versions could save randomly initialized corner embeddings.
        if use_native_boxes and not has_corner_weights:
            raise ValueError('Native box prompts require a checkpoint with trained corner embeddings')
        self._native_boxes = bool(use_native_boxes and has_corner_weights)
        # Stock Turbo has only trained positive/negative embeddings. Do not create
        # random frozen corner embeddings and pretend they are trained box prompts.
        if not self._native_boxes:
            for i in (2, 3):
                state.pop(f'prompt_encoder.point_embeddings.{i}.weight', None)
            sam.prompt_encoder.point_embeddings = torch.nn.ModuleList(list(sam.prompt_encoder.point_embeddings)[:2])
            sam.prompt_encoder.num_point_embeddings = 2
        sam.load_state_dict(state, strict=True)
        try:
            self.model = sam.to(self.device).eval()
        except torch.cuda.OutOfMemoryError:
            if not self._allow_cpu_fallback:
                raise
            self.device = torch.device('cpu')
            self.model = sam.to(self.device).eval()
            gc.collect()
            torch.cuda.empty_cache()
            warnings.warn('SAM CUDA allocation failed; continuing on CPU.', RuntimeWarning, stacklevel=2)
        for p in self.model.parameters():
            p.requires_grad = False
        self._features = None

    def set_image(self, volume_3d, gt_mask=None, spacing=(1., 1., 1.), affine=None):
        """Set a ZYX volume. Affine maps XYZ voxel indices to world millimeters.

        Without an affine, spacing defines an axis-aligned physical grid. Ground
        truth is accepted for legacy callers but never used to choose the crop.
        """
        vol = np.asarray(volume_3d, dtype=np.float32)
        if vol.ndim != 3 or any(size == 0 for size in vol.shape) or not np.isfinite(vol).all():
            raise ValueError('volume must be a finite 3D ZYX array')
        spacing = np.asarray(spacing, dtype=float)
        if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
            raise ValueError('spacing must contain three positive finite values')
        self._orig_shape = vol.shape
        self._orig_affine = np.diag([*spacing[::-1], 1.]) if affine is None else np.asarray(affine, dtype=float)
        if self._orig_affine.shape != (4, 4) or not np.isfinite(self._orig_affine).all() or abs(np.linalg.det(self._orig_affine[:3, :3])) < 1e-12:
            raise ValueError('affine must be a finite invertible 4x4 matrix')
        image = tio.ScalarImage(tensor=torch.from_numpy(vol.transpose(2, 1, 0).copy())[None], affine=self._orig_affine)
        self._orig_subject = tio.Subject(image=image)
        self._resampled = tio.Resample(TARGET_SPACING)(image)
        self._global_mask = np.zeros(vol.shape, dtype=np.uint8)
        self._features = None
        self._encode_crop((np.asarray(vol.shape) - 1) / 2)

    def _world(self, point_zyx):
        return self._orig_affine @ np.r_[np.asarray(point_zyx, dtype=float)[::-1], 1.]

    def _map_original_to_roi(self, point_zyx):
        """Return un-clipped XYZ crop coordinates; out-of-crop prompts recenter."""
        return (np.linalg.inv(self._roi_affine) @ self._world(point_zyx))[:3]

    def _map_original_bbox_to_roi(self, bbox_zyx):
        # Eight corners handle rotations/reflections in the input affine.
        from itertools import product
        bounds = np.asarray(bbox_zyx, dtype=float).reshape(2, 3)
        corners = np.array([self._map_original_to_roi(p) for p in product(*zip(bounds[0], bounds[1]))])
        return np.r_[corners.min(0), corners.max(0)]

    def _encode_crop(self, center_zyx):
        center = (np.linalg.inv(self._resampled.affine) @ self._world(center_zyx))[:3]
        shape = np.asarray(self._resampled.spatial_shape)
        start = np.rint(center).astype(int) - CROP_SIZE // 2
        # Near an image boundary, retain as much real context as possible.
        start = np.minimum(np.maximum(start, np.minimum(0, shape - CROP_SIZE)), np.maximum(0, shape - CROP_SIZE))
        # Resampling can place an original edge voxel just beyond the last
        # resampled center. Permit one padded voxel rather than dropping a
        # physically valid edge click after the context-preserving clamp.
        start = np.maximum(start, np.ceil(center - (CROP_SIZE - 1)).astype(int))
        start = np.minimum(start, np.floor(center).astype(int))
        if self._features is not None and hasattr(self, '_roi_start') and np.array_equal(start, self._roi_start):
            return
        roi = np.zeros((1, CROP_SIZE, CROP_SIZE, CROP_SIZE), dtype=np.float32)
        lo, hi = np.maximum(start, 0), np.minimum(start + CROP_SIZE, shape)
        dst = tuple(slice(int(a), int(b)) for a, b in zip(lo - start, hi - start))
        src = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        roi[(slice(None), *dst)] = self._resampled.data.numpy()[(slice(None), *src)]
        self._roi_affine = self._resampled.affine.copy()
        self._roi_affine[:3, 3] = (self._resampled.affine @ np.r_[start, 1.])[:3]
        self._roi_shape = (CROP_SIZE,) * 3
        self._roi_start = start
        tissue = roi > 0
        values = roi[tissue] if tissue.any() else roi.ravel()
        sigma = float(values.std())
        if sigma < 1e-6:
            sigma = float(roi.std())
        if sigma < 1e-6:
            sigma = 1.
        norm = (roi - values.mean()) / sigma
        with torch.inference_mode():
            try:
                self._features = self.model.image_encoder(torch.from_numpy(norm)[None].to(self.device))
            except torch.cuda.OutOfMemoryError:
                if not getattr(self, '_allow_cpu_fallback', False):
                    raise
                self._fallback_to_cpu()
                self._features = self.model.image_encoder(torch.from_numpy(norm)[None])

    def _fallback_to_cpu(self):
        self.device = torch.device('cpu')
        self.model.to(self.device)
        if self._features is not None:
            self._features = self._features.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        warnings.warn('SAM ran out of CUDA memory; retrying on CPU.', RuntimeWarning, stacklevel=2)

    def _prior_in_crop(self):
        image = tio.LabelMap(tensor=torch.from_numpy(self._global_mask.transpose(2, 1, 0).copy())[None], affine=self._orig_affine)
        target = tio.ScalarImage(tensor=torch.zeros(1, *self._roi_shape), affine=self._roi_affine)
        prior = tio.Resample(target, image_interpolation='nearest')(image).data.float()[None].to(self.device)
        # Decoder returns a quarter-resolution mask; prompt downscaling expects it.
        return F.interpolate(prior * 20. - 10., size=(32, 32, 32), mode='trilinear', align_corners=False)

    def predict_mask(self, points=None, labels=None, box_3d=None, mask_logits=None):
        """Predict from original ZYX voxel prompts, returning mask and crop logits.

        Stock Turbo boxes localize the crop and add a positive center click; they
        are not learned bounding-box embeddings. Moving crops reprojects the last
        global prediction as a prior and preserves predictions outside the crop.
        """
        if self._features is None:
            raise RuntimeError('Call set_image() first')
        pts = np.asarray(points if points is not None else [], dtype=float).reshape(-1, 3)
        lbl = np.asarray(labels if labels is not None else [], dtype=int)
        if len(lbl) != len(pts) or np.any(~np.isin(lbl, [0, 1])) or not np.isfinite(pts).all():
            raise ValueError('points require matching binary labels and finite ZYX coordinates')
        if len(pts) and (np.any(pts < 0) or np.any(pts > np.asarray(self._orig_shape) - 1)):
            raise ValueError('point lies outside the original volume')
        center = None
        if box_3d is not None:
            box = np.asarray(box_3d, dtype=float)
            if box.shape != (6,) or not np.isfinite(box).all() or np.any(box[:3] > box[3:]) or np.any(box[:3] < 0) or np.any(box[3:] > np.asarray(self._orig_shape) - 1):
                raise ValueError('box must contain ordered in-volume ZYX corners')
            center = (box[:3] + box[3:]) / 2
            if not self._native_boxes:
                warnings.warn('Turbo checkpoint lacks trained box embeddings; using box-centered crop and positive center click.', RuntimeWarning, stacklevel=2)
                pts = np.vstack([pts, center])
                lbl = np.r_[lbl, 1]
        if center is None and len(pts):
            outside = [p for p in pts if np.any(self._map_original_to_roi(p) < 0) or np.any(self._map_original_to_roi(p) > CROP_SIZE - 1)]
            if outside:
                center = outside[-1]
        moved = False
        if center is not None:
            old_affine = self._roi_affine.copy()
            self._encode_crop(center)
            moved = not np.allclose(old_affine, self._roi_affine)
        if box_3d is not None:
            extent = self._map_original_bbox_to_roi(box_3d)
            if np.any(extent[:3] < 0) or np.any(extent[3:] > CROP_SIZE - 1):
                warnings.warn('Requested box exceeds the current 192 mm crop; further prompts are required to cover other regions.', RuntimeWarning, stacklevel=2)
        coords = np.array([self._map_original_to_roi(p) for p in pts]).reshape(-1, 3)
        keep = np.all((coords >= 0) & (coords <= CROP_SIZE - 1), axis=1)
        # Historical clicks outside a new crop cannot be represented; never clip.
        prompt = (torch.as_tensor(coords[keep], dtype=torch.float32, device=self.device)[None], torch.as_tensor(lbl[keep], device=self.device)[None]) if keep.any() else None
        boxes = None
        if box_3d is not None and self._native_boxes:
            boxes = torch.as_tensor(np.clip(self._map_original_bbox_to_roi(box_3d), 0, CROP_SIZE - 1), dtype=torch.float32, device=self.device)[None]
        prior = self._prior_in_crop() if moved and self._global_mask.any() else (None if moved or mask_logits is None else mask_logits.to(self.device))
        def decode():
            sparse, dense = self.model.prompt_encoder(points=prompt, boxes=boxes, masks=prior)
            result, _ = self.model.mask_decoder(image_embeddings=self._features, image_pe=self.model.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense, multimask_output=False)
            return result
        with torch.inference_mode():
            try:
                logits = decode()
            except torch.cuda.OutOfMemoryError:
                if not getattr(self, '_allow_cpu_fallback', False):
                    raise
                self._fallback_to_cpu()
                prompt = None if prompt is None else tuple(item.cpu() for item in prompt)
                boxes = None if boxes is None else boxes.cpu()
                prior = None if prior is None else prior.cpu()
                logits = decode()
            binary = (F.interpolate(logits, size=self._roi_shape, mode='trilinear', align_corners=False) > 0).float()
        label = tio.LabelMap(tensor=binary[0].cpu(), affine=self._roi_affine)
        resample = tio.Resample(self._orig_subject.image, image_interpolation='nearest')
        mask = resample(label).data.numpy()[0].transpose(2, 1, 0).astype(np.uint8)
        coverage = resample(tio.LabelMap(tensor=torch.ones(1, *self._roi_shape), affine=self._roi_affine)).data.numpy()[0].transpose(2, 1, 0) > 0
        self._global_mask[coverage] = mask[coverage]
        return self._global_mask.copy(), logits.detach().cpu()
