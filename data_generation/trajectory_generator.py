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

def save_triview_image(axial, coronal, sagittal, spacing, output_path):
    """
    Concatenates the three slices horizontally and saves as a PNG image.
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

    # Normalize slices
    ax_norm = normalize_slice(ax_res)
    cor_norm = normalize_slice(cor_res)
    sag_norm = normalize_slice(sag_res)
    
    # Resize to have same height for horizontal concatenation
    h = max(ax_norm.shape[0], cor_norm.shape[0], sag_norm.shape[0])
    
    def pad_to_height(img, target_h):
        if img.shape[0] < target_h:
            pad_top = (target_h - img.shape[0]) // 2
            pad_bottom = target_h - img.shape[0] - pad_top
            return np.pad(img, ((pad_top, pad_bottom), (0, 0)), mode='constant')
        return img
        
    ax_pad = pad_to_height(ax_norm, h)
    cor_pad = pad_to_height(cor_norm, h)
    sag_pad = pad_to_height(sag_norm, h)
    
    triview = np.hstack([ax_pad, cor_pad, sag_pad])
    img = Image.fromarray(triview)
    img.save(output_path)

def extract_triview_slices(volume, center):
    """
    Extracts tri-view slices (Axial, Coronal, Sagittal) centered at the given 3D coordinate.
    
    Args:
        volume (np.ndarray): 3D volume data
        center (tuple): (z, y, x) center coordinate
        
    Returns:
        tuple: (axial, coronal, sagittal) 2D slices
    """
    z, y, x = center
    z = max(0, min(z, volume.shape[0] - 1))
    y = max(0, min(y, volume.shape[1] - 1))
    x = max(0, min(x, volume.shape[2] - 1))
    
    axial = volume[z, :, :]
    coronal = volume[:, y, :]
    sagittal = volume[:, :, x]
    
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


def find_next_click(pred_mask, gt_mask, spacing=(1.0, 1.0, 1.0)):
    """
    Legacy: Finds only the next positive click (largest False Negative cluster).
    """
    fn_mask = np.logical_and(gt_mask == 1, pred_mask == 0)
    if not np.any(fn_mask):
        return None
    labeled_fn, num_features = label(fn_mask)
    if num_features == 0:
        return None
    component_sizes = np.bincount(labeled_fn.ravel())
    component_sizes[0] = 0
    largest_component_label = component_sizes.argmax()
    largest_fn_mask = (labeled_fn == largest_component_label)
    dist_transform = distance_transform_edt(largest_fn_mask, sampling=spacing)
    return np.unravel_index(np.argmax(dist_transform), dist_transform.shape)

def generate_trajectory(sam_model, volume, gt_mask, target_name, volume_id, spacing=(1.0, 1.0, 1.0), max_turns=5, output_dir="data/trajectories", image_dir="data/images"):
    """
    Simulates the interaction and generates a trajectory JSON and corresponding tri-view images.
    Supports both positive (add) and negative (remove) clicks.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    
    trajectory = {}
    points = []
    labels = []
    
    # 1. Initial prompt: Centroid of GT mask
    if np.sum(gt_mask) == 0:
        print(f"Skipping {volume_id}: Empty GT mask.")
        return
        
    z_c, y_c, x_c = [int(c) for c in center_of_mass(gt_mask)]
    current_center = (z_c, y_c, x_c)
    
    sam_model.set_image(volume)
    pred_mask = np.zeros(volume.shape, dtype=np.uint8)
    
    for turn in range(max_turns):
        # Render and save tri-view slices (centered on the initial centroid for stability)
        ax, cor, sag = extract_triview_slices(volume, current_center)
        img_name = f"{volume_id}_{target_name}_t{turn}.png"
        img_path = os.path.join(image_dir, img_name)
        save_triview_image(ax, cor, sag, spacing, img_path)
        
        # Determine action
        if turn == 0:
            # Initial positive click at centroid
            action_point = current_center
            click_type = "positive"
        else:
            # Check both FNs (need to add) and FPs (need to remove)
            clicks = find_next_clicks(pred_mask, gt_mask, spacing)
            
            # Prioritize positive clicks (adding missing regions)
            if clicks["positive"] is not None:
                action_point = clicks["positive"]
                click_type = "positive"
            elif clicks["negative"] is not None:
                action_point = clicks["negative"]
                click_type = "negative"
            else:
                # Perfect segmentation: stop
                trajectory[f"turn_{turn}"] = {
                    "image": img_path,
                    "instruction": f"Segment the {target_name}",
                    "action": {"name": "stop_action"}
                }
                break
                
        # Record point and predict
        points.append(action_point)
        labels.append(1 if click_type == "positive" else 0)
        pred_mask = sam_model.predict_mask(points, labels)
        
        # Compute IoU
        iou = compute_iou_3d(pred_mask, gt_mask)
        print(f"Turn {turn}: IoU = {iou:.4f}, Click = {action_point}, Type = {click_type}")
        
        # Record trajectory turn (normalize path to forward slashes)
        img_path_normalized = img_path.replace("\\", "/")
        trajectory[f"turn_{turn}"] = {
            "image": img_path_normalized,
            "instruction": f"Segment the {target_name}",
            "action": {
                "name": "add_3d_point", 
                "point_3d": list(int(p) for p in action_point), 
                "type": click_type
            }
        }
        
        # Update center to latest click point for consistent behavior with RL environment
        current_center = action_point
        
        if iou > 0.95:  # Convergence threshold
            turn += 1
            ax, cor, sag = extract_triview_slices(volume, current_center)
            img_name = f"{volume_id}_{target_name}_t{turn}.png"
            img_path = os.path.join(image_dir, img_name)
            save_triview_image(ax, cor, sag, spacing, img_path)
            
            trajectory[f"turn_{turn}"] = {
                "image": img_path,
                "instruction": f"Segment the {target_name}",
                "action": {"name": "stop_action"}
            }
            break
            
    out_file = os.path.join(output_dir, f"{volume_id}_{target_name}.json")
    # Normalize path separators to forward slashes for cross-platform compatibility
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(trajectory, f, indent=2, ensure_ascii=False)
        
    return trajectory
