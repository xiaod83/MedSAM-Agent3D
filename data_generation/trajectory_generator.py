import cv2
import numpy as np
import os
import json
from scipy.ndimage import distance_transform_edt, label, center_of_mass
from PIL import Image

def normalize_slice(slice_2d):
    """Normalize a 2D slice to 0-255 uint8 format."""
    if slice_2d.max() == slice_2d.min():
        return np.zeros_like(slice_2d, dtype=np.uint8)
    norm = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min())
    return (norm * 255).astype(np.uint8)

def resample_slice(slice_2d, spacing_2d, target_spacing=(1.0, 1.0)):
    """
    Resamples a 2D slice to a target isotropic spacing using cubic interpolation.
    spacing_2d: (dy, dx)
    """
    scale_y = spacing_2d[0] / target_spacing[0]
    scale_x = spacing_2d[1] / target_spacing[1]
    
    if scale_y == 1.0 and scale_x == 1.0:
        return slice_2d
        
    new_shape = (int(slice_2d.shape[0] * scale_y), int(slice_2d.shape[1] * scale_x))
    
    if new_shape[0] == 0 or new_shape[1] == 0:
        return slice_2d
        
    # cv2.resize expects (width, height)
    resampled = cv2.resize(slice_2d, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_CUBIC)
    return resampled

def overlay_mask(slice_2d, mask_2d, alpha=0.3, color=(0, 255, 0)):
    """Overlay a binary mask on a grayscale slice in semi-transparent green."""
    if mask_2d is None or mask_2d.sum() == 0:
        return slice_2d
    rgb = np.stack([slice_2d] * 3, axis=-1).astype(np.float32)
    mask_bool = mask_2d.astype(bool)
    for c in range(3):
        rgb[..., c] = np.where(mask_bool,
                               rgb[..., c] * (1 - alpha) + color[c] * alpha,
                               rgb[..., c])
    return rgb.astype(np.uint8)


def save_triview_image(axial, coronal, sagittal, spacing, output_path,
                       mask_axial=None, mask_coronal=None, mask_sagittal=None):
    """
    Concatenates the three slices horizontally and saves as a PNG image.
    If mask_* are provided, overlays them in semi-transparent green.
    Applies resampling based on physical spacing to correct anisotropy.
    spacing: (dz, dy, dx)
    """
    # Spacing for each view
    # Axial: (dy, dx)
    ax_spacing = (spacing[1], spacing[2])
    # Coronal: (dz, dx)
    cor_spacing = (spacing[0], spacing[2])
    # Sagittal: (dz, dy)
    sag_spacing = (spacing[0], spacing[1])
    
    # Resample to isotropic (1.0, 1.0)
    ax_res = resample_slice(axial, ax_spacing)
    cor_res = resample_slice(coronal, cor_spacing)
    sag_res = resample_slice(sagittal, sag_spacing)

    # Resample mask slices if provided
    if mask_axial is not None:
        ax_mask = resample_slice(mask_axial.astype(np.float32), ax_spacing) > 0.5
    else:
        ax_mask = None
    if mask_coronal is not None:
        cor_mask = resample_slice(mask_coronal.astype(np.float32), cor_spacing) > 0.5
    else:
        cor_mask = None
    if mask_sagittal is not None:
        sag_mask = resample_slice(mask_sagittal.astype(np.float32), sag_spacing) > 0.5
    else:
        sag_mask = None

    # Normalize slices
    ax_norm = normalize_slice(ax_res)
    cor_norm = normalize_slice(cor_res)
    sag_norm = normalize_slice(sag_res)
    
    # Overlay masks in semi-transparent green
    if ax_mask is not None:
        ax_norm = overlay_mask(ax_norm, ax_mask)
    if cor_mask is not None:
        cor_norm = overlay_mask(cor_norm, cor_mask)
    if sag_mask is not None:
        sag_norm = overlay_mask(sag_norm, sag_mask)
    
    # Resize to have same height for horizontal concatenation
    h = max(ax_norm.shape[0], cor_norm.shape[0], sag_norm.shape[0])
    
    def pad_to_height(img, target_h):
        if img.shape[0] < target_h:
            pad_top = (target_h - img.shape[0]) // 2
            pad_bottom = target_h - img.shape[0] - pad_top
            pad_widths = [(pad_top, pad_bottom), (0, 0)]
            if img.ndim == 3:
                pad_widths.append((0, 0))  # RGB channel
            return np.pad(img, pad_widths, mode='constant')
        return img
        
    ax_pad = pad_to_height(ax_norm, h)
    cor_pad = pad_to_height(cor_norm, h)
    sag_pad = pad_to_height(sag_norm, h)
    
    triview = np.hstack([ax_pad, cor_pad, sag_pad])
    img = Image.fromarray(triview)
    img.save(output_path)

def extract_triview_slices(volume, center, mask=None):
    """
    Extracts tri-view slices (Axial, Coronal, Sagittal) centered at the given 3D coordinate.
    If mask is provided, also returns corresponding mask slices for overlay.
    """
    z, y, x = center
    z = max(0, min(z, volume.shape[0] - 1))
    y = max(0, min(y, volume.shape[1] - 1))
    x = max(0, min(x, volume.shape[2] - 1))
    
    axial = volume[z, :, :]
    coronal = volume[:, y, :]
    sagittal = volume[:, :, x]
    
    if mask is not None:
        m_ax = mask[z, :, :]
        m_cor = mask[:, y, :]
        m_sag = mask[:, :, x]
        return (axial, coronal, sagittal), (m_ax, m_cor, m_sag)
    
    return axial, coronal, sagittal

def compute_iou_3d(pred_mask, gt_mask):
    """
    Computes the 3D Intersection over Union (IoU) between predicted and ground truth masks.
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    if union == 0:
        return 0.0
    return intersection / union

def find_next_clicks(pred_mask, gt_mask, spacing=(1.0, 1.0, 1.0)):
    """
    Finds the next point(s) to click by analyzing both False Negatives and False Positives.
    
    Returns a dict with:
      - "positive" (tuple or None): center of largest FN cluster (model needs to add)
      - "negative" (tuple or None): center of largest FP cluster (model needs to remove)
    
    Prioritizes FN over FP when both exist.
    """
    results = {"positive": None, "negative": None}
    
    # False Negatives: GT is 1 but Pred is 0 (regions to add)
    fn_mask = np.logical_and(gt_mask == 1, pred_mask == 0)
    
    if np.any(fn_mask):
        labeled_fn, num_features = label(fn_mask)
        component_sizes = np.bincount(labeled_fn.ravel())
        component_sizes[0] = 0
        largest_component_label = component_sizes.argmax()
        largest_fn_mask = (labeled_fn == largest_component_label)
        
        dist_transform = distance_transform_edt(largest_fn_mask, sampling=spacing)
        next_click = np.unravel_index(np.argmax(dist_transform), dist_transform.shape)
        results["positive"] = next_click
    
    # False Positives: Pred is 1 but GT is 0 (regions to remove)
    fp_mask = np.logical_and(gt_mask == 0, pred_mask == 1)
    
    if np.any(fp_mask):
        labeled_fp, num_features = label(fp_mask)
        component_sizes = np.bincount(labeled_fp.ravel())
        component_sizes[0] = 0
        largest_component_label = component_sizes.argmax()
        largest_fp_mask = (labeled_fp == largest_component_label)
        
        dist_transform = distance_transform_edt(largest_fp_mask, sampling=spacing)
        next_click = np.unravel_index(np.argmax(dist_transform), dist_transform.shape)
        results["negative"] = next_click
    
    return results


def generate_trajectory(sam_model, volume, gt_mask, target_name, volume_id, spacing=(1.0, 1.0, 1.0), max_turns=5, output_dir="data/trajectories", image_dir="data/images"):
    """
    Simulates an expert interaction trajectory. Generates tri-view images WITH mask overlay
    (semi-transparent green), action JSON with correct schema, and delta-IoU progress filtering.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    
    trajectory = {}
    points = []
    labels = []
    
    if np.sum(gt_mask) == 0:
        print(f"Skipping {volume_id}: Empty GT mask.")
        return
    
    z_c, y_c, x_c = [int(c) for c in center_of_mass(gt_mask)]
    # Turn 0 focus: volume center (not GT centroid) so MLLM sees a realistic initial view
    current_center = (volume.shape[0]//2, volume.shape[1]//2, volume.shape[2]//2)
    
    sam_model.set_image(volume, gt_mask)
    pred_mask = np.zeros(volume.shape, dtype=np.uint8)
    prev_iou = 0.0
    
    for turn in range(max_turns):
        # Render tri-view with mask overlay
        slices, mask_slices = extract_triview_slices(volume, current_center, mask=pred_mask)
        ax, cor, sag = slices
        m_ax, m_cor, m_sag = mask_slices
        img_name = f"{volume_id}_{target_name}_t{turn}.png"
        img_path = os.path.join(image_dir, img_name)
        save_triview_image(ax, cor, sag, spacing, img_path,
                           mask_axial=m_ax, mask_coronal=m_cor, mask_sagittal=m_sag)
        
        # Determine action
        if turn == 0:
            # Initial positive click at GT centroid (the prompt point, not the view center)
            action_point = (z_c, y_c, x_c)
            click_type = "positive"
        else:
            clicks = find_next_clicks(pred_mask, gt_mask, spacing)
            if clicks["positive"] is not None:
                action_point = clicks["positive"]
                click_type = "positive"
            elif clicks["negative"] is not None:
                action_point = clicks["negative"]
                click_type = "negative"
            else:
                trajectory[f"turn_{turn}"] = {
                    "image": img_path.replace("\\", "/"),
                    "instruction": f"Segment the {target_name}",
                    "iou_after": round(prev_iou, 4),
                    "action": {"name": "stop_action"}
                }
                break
        
        # Predict mask
        points.append(action_point)
        labels.append(1 if click_type == "positive" else 0)
        pred_mask = sam_model.predict_mask(points, labels)
        iou = compute_iou_3d(pred_mask, gt_mask)
        delta_iou = iou - prev_iou
        prev_iou = iou
        
        # Delta-IoU progress check: retry at next error cluster if delta <= threshold
        if turn > 0 and delta_iou <= 0.04:
            for retry in range(3):
                clicks = find_next_clicks(pred_mask, gt_mask, spacing)
                alt_pt = clicks.get("positive") or clicks.get("negative")
                if alt_pt is None or alt_pt == action_point:
                    break
                test_pts = points[:-1] + [alt_pt]
                test_labels = labels[:-1] + [1 if clicks.get("positive") == alt_pt else 0]
                test_mask = sam_model.predict_mask(test_pts, test_labels)
                test_iou = compute_iou_3d(test_mask, gt_mask)
                if test_iou - (iou - delta_iou) > 0.04:
                    points[-1] = alt_pt
                    labels[-1] = test_labels[-1]
                    action_point = alt_pt
                    click_type = "positive" if clicks.get("positive") == alt_pt else "negative"
                    pred_mask = test_mask
                    iou = test_iou
                    prev_iou = iou
                    break
        
        # Update focus center to predicted mask centroid for Turn 1+
        if pred_mask.sum() > 0:
            current_center = [int(c) for c in center_of_mass(pred_mask.astype(np.uint8))]
        else:
            current_center = action_point
        
        print(f"Turn {turn}: IoU = {iou:.4f}, Delta = {delta_iou:.4f}, Click = {action_point}, Type = {click_type}")
        
        # Record turn with CORRECT action schema (nested arguments, from README/MedSAM-Agent)
        trajectory[f"turn_{turn}"] = {
            "image": img_path.replace("\\", "/"),
            "instruction": f"Segment the {target_name}",
            "iou_after": round(iou, 4),
            "action": {
                "name": "add_3d_point",
                "arguments": {
                    "point_3d": [int(p) for p in action_point],
                    "point_type": click_type
                }
            }
        }
        
        if iou > 0.80:  # Convergence threshold (0.90 is target; 0.80 triggers stop for near-perfect)
            turn += 1
            slices, ms = extract_triview_slices(volume, current_center, mask=pred_mask)
            img_name = f"{volume_id}_{target_name}_t{turn}.png"
            img_path = os.path.join(image_dir, img_name)
            save_triview_image(*slices, spacing, img_path,
                               mask_axial=ms[0], mask_coronal=ms[1], mask_sagittal=ms[2])
            trajectory[f"turn_{turn}"] = {
                "image": img_path.replace("\\", "/"),
                "instruction": f"Segment the {target_name}",
                "iou_after": round(iou, 4),
                "action": {"name": "stop_action"}
            }
            break
    
    out_file = os.path.join(output_dir, f"{volume_id}_{target_name}.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(trajectory, f, indent=2, ensure_ascii=False)
    return trajectory
