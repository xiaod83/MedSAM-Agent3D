"""
Tri-view rendering: 3D medical volume -> 2D stitched image.

Extracts three orthogonal slices (axial, coronal, sagittal) centered on a
focus point, overlays the current segmentation mask as semi-transparent
green, and stitches them side-by-side for MLLM consumption.

References:
    MedSAM-Agent visual_utils.py -- overlay_mask, overlay_points pattern
"""

from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw


def normalize_slice(slice_2d: np.ndarray) -> np.ndarray:
    """Normalize a 2D slice to [0, 255] uint8 for display."""
    if slice_2d.max() == slice_2d.min():
        return np.zeros_like(slice_2d, dtype=np.uint8)
    norm = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min())
    return (norm * 255).astype(np.uint8)


def resample_slice(
    slice_2d: np.ndarray,
    spacing_2d: Tuple[float, float],
    target_spacing: Tuple[float, float] = (1.0, 1.0),
) -> np.ndarray:
    """Resample a 2D slice to isotropic target spacing via cubic interpolation."""
    import cv2

    scale_y = spacing_2d[0] / target_spacing[0]
    scale_x = spacing_2d[1] / target_spacing[1]
    if scale_y == 1.0 and scale_x == 1.0:
        return slice_2d
    new_shape = (max(1, int(slice_2d.shape[0] * scale_y)), max(1, int(slice_2d.shape[1] * scale_x)))
    return cv2.resize(slice_2d, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_CUBIC)


def overlay_mask(
    slice_2d: np.ndarray,
    mask_2d: Optional[np.ndarray],
    alpha: float = 0.3,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Overlay a binary mask as a semi-transparent colored layer on a 2D slice.

    Always returns (H, W, 3) RGB uint8 array.

    References:
        MedSAM-Agent visual_utils.overlay_mask -- same alpha blending pattern.
    """
    if mask_2d is None or mask_2d.sum() == 0:
        return np.stack([slice_2d] * 3, axis=-1).astype(np.uint8)

    rgb = np.stack([slice_2d] * 3, axis=-1).astype(np.float32)
    mask_bool = mask_2d.astype(bool)
    for c in range(3):
        rgb[..., c] = np.where(
            mask_bool,
            rgb[..., c] * (1 - alpha) + color[c] * alpha,
            rgb[..., c],
        )
    return rgb.astype(np.uint8)


def extract_triview(
    volume: np.ndarray,
    center: Tuple[int, int, int],
    mask: Optional[np.ndarray] = None,
):
    """Extract three orthogonal 2D slices from a 3D volume.

    Args:
        volume: (Z, Y, X) array.
        center: (z, y, x) focus point.
        mask: Optional (Z, Y, X) binary mask.

    Returns:
        If mask is None: (axial, coronal, sagittal) 2D slices.
        If mask is provided: (axial, coronal, sagittal), (mask_axial, mask_coronal, mask_sagittal).
    """
    z, y, x = [max(0, min(int(c), s - 1)) for c, s in zip(center, volume.shape)]
    axial = volume[z, :, :]
    coronal = volume[:, y, :]
    sagittal = volume[:, :, x]

    if mask is not None:
        return (axial, coronal, sagittal), (mask[z, :, :], mask[:, y, :], mask[:, :, x])

    return axial, coronal, sagittal


def _pad_to_height(img: np.ndarray, target_height: int) -> np.ndarray:
    """Vertically center-pad an image to a target height."""
    if img.shape[0] < target_height:
        top = (target_height - img.shape[0]) // 2
        bot = target_height - img.shape[0] - top
        pw = [(top, bot), (0, 0)] + ([(0, 0)] if img.ndim == 3 else [])
        return np.pad(img, pw, mode="constant")
    return img


def _resample_mask(mask_2d: np.ndarray, spacing_2d: Tuple[float, float]) -> Optional[np.ndarray]:
    """Preserve labels with nearest-neighbor interpolation."""
    import cv2
    height = max(1, int(mask_2d.shape[0] * spacing_2d[0]))
    width = max(1, int(mask_2d.shape[1] * spacing_2d[1]))
    return cv2.resize(mask_2d.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0


def render_triview(
    volume: np.ndarray,
    center: Tuple[int, int, int],
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    mask: Optional[np.ndarray] = None,
) -> Image.Image:
    """Render tri-view (Axial | Coronal | Sagittal) as a single PIL Image.

    This is the main public entry point. The MLLM sees this image.

    Args:
        volume: (Z, Y, X) 3D volume array.
        center: (z, y, x) focus point for slice extraction.
        spacing: (dz, dy, dx) voxel spacing.
        mask: Optional (Z, Y, X) binary mask to overlay in green.

    Returns:
        PIL Image of the tri-view stitched side-by-side.
    """
    if volume.ndim != 3 or len(center) != 3:
        raise ValueError("Expected a 3D volume and three-coordinate focus")
    if len(spacing) != 3 or any(not np.isfinite(v) or v <= 0 for v in spacing):
        raise ValueError("spacing must contain three positive finite values")
    if mask is not None and mask.shape != volume.shape:
        raise ValueError("Mask shape must match volume")
    center = tuple(max(0, min(int(c), size - 1)) for c, size in zip(center, volume.shape))
    # Extract slices
    if mask is not None and mask.sum() > 0:
        (ax, cor, sag), (m_ax, m_cor, m_sag) = extract_triview(volume, center, mask=mask)
    else:
        ax, cor, sag = extract_triview(volume, center, mask=None)
        m_ax = m_cor = m_sag = None

    # Resample to isotropic spacing
    ax_res = resample_slice(ax, (spacing[1], spacing[2]))
    cor_res = resample_slice(cor, (spacing[0], spacing[2]))
    sag_res = resample_slice(sag, (spacing[0], spacing[1]))

    # Resample masks
    if m_ax is not None:
        m_ax_r = _resample_mask(m_ax, (spacing[1], spacing[2]))
        m_cor_r = _resample_mask(m_cor, (spacing[0], spacing[2]))
        m_sag_r = _resample_mask(m_sag, (spacing[0], spacing[1]))
    else:
        m_ax_r = m_cor_r = m_sag_r = None

    # Normalize and overlay
    ax_norm = normalize_slice(ax_res)
    cor_norm = normalize_slice(cor_res)
    sag_norm = normalize_slice(sag_res)

    ax_norm = overlay_mask(ax_norm, m_ax_r)
    cor_norm = overlay_mask(cor_norm, m_cor_r)
    sag_norm = overlay_mask(sag_norm, m_sag_r)

    # Crosshairs mark slice intersection, independent of panel center/padding.
    z, y, x = center
    dz, dy, dx = volume.shape
    panels = []
    definitions = [
        (ax_norm, (x / max(dx - 1, 1), y / max(dy - 1, 1)), f"Axial Z={z} | X right, Y down"),
        (cor_norm, (x / max(dx - 1, 1), z / max(dz - 1, 1)), f"Coronal Y={y} | X right, Z down"),
        (sag_norm, (y / max(dy - 1, 1), z / max(dz - 1, 1)), f"Sagittal X={x} | Y right, Z down"),
    ]
    height = max(ax_norm.shape[0], cor_norm.shape[0], sag_norm.shape[0])
    for pixels, (fx, fy), label in definitions:
        panel = Image.fromarray(pixels)
        draw = ImageDraw.Draw(panel)
        cx, cy = round(fx * (panel.width - 1)), round(fy * (panel.height - 1))
        draw.line((cx, 0, cx, panel.height - 1), fill=(0, 220, 255))
        draw.line((0, cy, panel.width - 1, cy), fill=(0, 220, 255))
        framed = Image.new("RGB", (max(panel.width, 250), height + 42), (16, 16, 16))
        framed.paste(panel, (0, 32))
        frame_draw = ImageDraw.Draw(framed)
        frame_draw.text((3, 3), label, fill="white")
        frame_draw.text((3, 16), "Image edges: 0 -> 999 (exclude padding)", fill="white")
        frame_draw.rectangle((0, 32, panel.width - 1, 32 + panel.height - 1), outline=(180, 180, 180))
        panels.append(framed)
    result = Image.new("RGB", (sum(p.width for p in panels) + 12, height + 42))
    left = 0
    for panel in panels:
        result.paste(panel, (left, 0))
        left += panel.width + 6
    return result
