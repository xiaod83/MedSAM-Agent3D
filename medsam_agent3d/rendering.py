"""
Tri-view rendering: 3D medical volume -> 2D stitched image.

Extracts three orthogonal slices (axial, coronal, sagittal) through a
focus point, overlays the current segmentation mask as semi-transparent
green, and stitches them side-by-side for MLLM consumption.

References:
    MedSAM-Agent visual_utils.py -- overlay_mask, overlay_points pattern
"""

from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .views import view_point_to_voxel


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
    affine=None,
) -> Image.Image:
    """Render tri-view (Axial | Coronal | Sagittal) as a single PIL Image.

    This is the main public entry point. The MLLM sees this image.

    Args:
        volume: (Z, Y, X) 3D volume array.
        center: (z, y, x) focus point for slice extraction.
        spacing: (dz, dy, dx) voxel spacing.
        mask: Optional (Z, Y, X) binary mask to overlay in green.
        affine: Optional XYZ voxel-to-world affine, used for anatomical labels.

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

    # Each image owns its own normalized (u,v) grid. The focus marker stays
    # outside tissue: an interior crosshair strongly biased real model clicks
    # toward unrelated tissue at the slice intersection.
    z, y, x = center
    dz, dy, dx = volume.shape
    directions = _anatomical_axis_directions(affine)
    definitions = [
        (ax_norm, "AXIAL", f"z={z}/{dz - 1}", "X", "Y", x / max(dx - 1, 1), y / max(dy - 1, 1)),
        (cor_norm, "CORONAL", f"y={y}/{dy - 1}", "X", "Z", x / max(dx - 1, 1), z / max(dz - 1, 1)),
        (sag_norm, "SAGITTAL", f"x={x}/{dx - 1}", "Y", "Z", y / max(dy - 1, 1), z / max(dz - 1, 1)),
    ]
    panels = []
    for pixels, name, fixed, u_axis, v_axis, focus_u, focus_v in definitions:
        panels.append(_coordinate_panel(pixels, name, fixed, u_axis, v_axis,
                                        focus_u, focus_v, directions))
    gap, legend_height = 8, 30
    result = Image.new("RGB", (sum(panel.width for panel in panels) + gap * 2,
                              max(panel.height for panel in panels) + legend_height), (16, 16, 16))
    draw = ImageDraw.Draw(result)
    draw.text((4, 1), "Click image grid (u right, v down), each 0-999. Green = prediction; cyan margin ticks = slice focus only, not target.", fill=(235, 235, 235))
    if directions:
        draw.text((4, 14), "Anatomical labels: L/R = patient's Left/Right; P/A = Posterior/Anterior; I/S = Inferior/Superior.", fill=(200, 200, 200))
    else:
        draw.text((4, 14), "Anatomical directions unavailable. Grid coordinates refer to voxel axes; exclude labels and padding.", fill=(200, 200, 200))
    left = 0
    for panel in panels:
        result.paste(panel, (left, legend_height))
        left += panel.width + gap
    return result


def _anatomical_axis_directions(affine):
    """Derive increasing voxel-axis directions from a supplied RAS affine.

    Unknown or strongly oblique axes receive no anatomical claim. World +X is
    patient's right, +Y anterior, +Z superior; display axes follow array indices.
    """
    if affine is None:
        return {}
    matrix = np.asarray(affine, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("affine must be a finite 4x4 XYZ voxel-to-world matrix")
    positive, negative = ("R", "A", "S"), ("L", "P", "I")
    result = {}
    for column, axis in enumerate(("X", "Y", "Z")):
        vector = matrix[:3, column]
        norm = np.linalg.norm(vector)
        if norm == 0:
            continue
        world_axis = int(np.argmax(np.abs(vector)))
        if abs(vector[world_axis]) / norm < .8:
            continue
        increasing = positive[world_axis] if vector[world_axis] > 0 else negative[world_axis]
        decreasing = negative[world_axis] if vector[world_axis] > 0 else positive[world_axis]
        result[axis] = f"{decreasing}->{increasing}"
    return result


def _coordinate_panel(pixels, name, fixed, u_axis, v_axis, focus_u, focus_v, directions):
    """Frame a panel with ticks, light grid, and focus ticks outside anatomy."""
    image = Image.fromarray(pixels)
    # Use faint dashed lines, preserving the image features beneath the grid.
    grid = Image.new("RGBA", image.size)
    grid_draw = ImageDraw.Draw(grid)
    for value in (250, 500, 750):
        gx, gy = round(value / 999 * (image.width - 1)), round(value / 999 * (image.height - 1))
        for position in range(0, image.height, 12):
            grid_draw.line((gx, position, gx, min(position + 3, image.height - 1)), fill=(215, 215, 215, 65))
        for position in range(0, image.width, 12):
            grid_draw.line((position, gy, min(position + 3, image.width - 1), gy), fill=(215, 215, 215, 65))
    image = Image.alpha_composite(image.convert("RGBA"), grid).convert("RGB")
    left, top, right, bottom = 29, 52, 5, 9
    panel = Image.new("RGB", (left + image.width + right, top + image.height + bottom), (16, 16, 16))
    panel.paste(image, (left, top))
    draw = ImageDraw.Draw(panel)
    draw.text((left, 1), f"{name} | {fixed}", fill="white")
    u_direction = f" ({directions[u_axis]})" if u_axis in directions else ""
    v_direction = f" ({directions[v_axis]})" if v_axis in directions else ""
    draw.text((left, 14), f"u={u_axis}{u_direction}; v={v_axis}{v_direction}", fill=(220, 220, 220))
    for value in (0, 250, 500, 750, 999):
        cx = left + round(value / 999 * (image.width - 1))
        cy = top + round(value / 999 * (image.height - 1))
        text = str(value)
        box = draw.textbbox((0, 0), text)
        width = box[2] - box[0]
        tx = min(max(left, cx - width // 2), left + image.width - width)
        draw.text((tx, top - 18), text, fill=(235, 235, 235))
        draw.line((cx, top - 5, cx, top - 1), fill=(160, 160, 160))
        draw.text((left - 5 - width, cy - 5), text, fill=(235, 235, 235))
        draw.line((left - 3, cy, left - 1, cy), fill=(160, 160, 160))
    draw.rectangle((left, top, left + image.width - 1, top + image.height - 1), outline=(130, 130, 130))
    cx = left + round(focus_u * (image.width - 1))
    cy = top + round(focus_v * (image.height - 1))
    # Short cyan markers outside the image indicate focus without suggesting a
    # tissue target. The exact ZYX focus is provided in the model request.
    draw.line((cx - 3, top + image.height + 3, cx + 3, top + image.height + 3), fill=(0, 220, 255), width=2)
    draw.line((left + image.width + 2, cy - 3, left + image.width + 2, cy + 3), fill=(0, 220, 255), width=2)
    return panel
