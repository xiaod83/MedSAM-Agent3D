import torch
import numpy as np

def compute_reward(iou, previous_iou=None, step_cost=-0.05, prev_iou_peak=None, format_ok=True, is_stop=False):
    """
    Computes RL reward matching the README specification.
    """
    reward = 0.0
    if format_ok:
        reward += 0.5  # Format reward
    dice = 2 * iou / (1 + iou + 1e-8)
    reward += 0.5 * dice + 0.5 * iou  # Quality reward (Dice + IoU)
    if previous_iou is not None:
        delta_iou = iou - previous_iou
        if delta_iou > 0:
            reward += 0.2  # Improvement bonus
        if delta_iou < 0:
            reward -= 0.3  # Overshoot penalty
    if step_cost < 0 and previous_iou is not None:
        reward += step_cost  # Tool-cost penalty (exempts turn 0)
    if is_stop:
        reward += 0.3  # Stop bonus
    return reward

class MedSAMEnvironment:
    def __init__(self, sam_model, volume, gt_mask, target_name, volume_id, spacing=(1.0, 1.0, 1.0)):
        self.sam = sam_model
        self.volume = volume
        self.gt_mask = gt_mask
        self.target_name = target_name
        self.volume_id = volume_id
        self.spacing = spacing
        self.reset()
        
    def reset(self):
        self.points = []
        self.labels = []
        self.current_iou = None  # None = turn 0 exempt from tool-cost
        self.sam.set_image(self.volume, self.gt_mask)
        
        from scipy.ndimage import center_of_mass
        if np.sum(self.gt_mask) > 0:
            z_c, y_c, x_c = [int(c) for c in center_of_mass(self.gt_mask)]
        else:
            z_c, y_c, x_c = (self.volume.shape[0]//2, self.volume.shape[1]//2, self.volume.shape[2]//2)
            
        self.current_center = (z_c, y_c, x_c)
        return self._get_state()
        
    def _get_state(self):
        """
        Renders the tri-view image in-memory and returns it with the prompt.
        Uses BytesIO to avoid disk I/O bottlenecks during RL rollouts.
        """
        from data_generation.trajectory_generator import extract_triview_slices, normalize_slice, resample_slice
        from io import BytesIO
        from PIL import Image
        import numpy as np
        
        ax, cor, sag = extract_triview_slices(self.volume, self.current_center)
        
        # Resample to isotropic and normalize
        ax_res = resample_slice(ax, (self.spacing[1], self.spacing[2]))
        cor_res = resample_slice(cor, (self.spacing[0], self.spacing[2]))
        sag_res = resample_slice(sag, (self.spacing[0], self.spacing[1]))
        
        ax_norm = normalize_slice(ax_res)
        cor_norm = normalize_slice(cor_res)
        sag_norm = normalize_slice(sag_res)
        
        # Pad to same height for horizontal concatenation
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
        
        # Use BytesIO to avoid disk I/O
        buf = BytesIO()
        Image.fromarray(triview).save(buf, format='PNG')
        buf.seek(0)
        image = Image.open(buf).convert("RGB")
        
        instruction = f"Segment the {self.target_name}"
        
        # Return multimodal state
        return {"image": image, "instruction": instruction, "center": self.current_center}
        
    def step(self, action_point, click_type="positive"):
        """
        Execute a segmentation step.
        
        Args:
            action_point: (z, y, x) tuple for the click location
            click_type: "positive" (add region) or "negative" (remove region)
        
        Returns:
            state, reward, done
        """
        self.points.append(action_point)
        self.labels.append(1 if click_type == "positive" else 0)
        
        pred_mask = self.sam.predict_mask(self.points, self.labels)
        
        from data_generation.trajectory_generator import compute_iou_3d
        new_iou = compute_iou_3d(pred_mask, self.gt_mask)
        
        reward = compute_reward(new_iou, self.current_iou, step_cost=-0.05)
        self.current_iou = new_iou
        self.current_center = action_point
        
        done = self.current_iou > 0.95 or len(self.points) >= 10
        
        return self._get_state(), reward, done