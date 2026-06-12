import numpy as np
import torch

class FrozenSAMMed3D:
    def __init__(self, checkpoint_path=None):
        """
        Loads the SAM-Med3D model and freezes its parameters.
        """
        print("Loading Frozen SAM-Med3D...")
        # self.model = load_sam_med3d(checkpoint_path)
        # self.model.eval()
        # for param in self.model.parameters():
        #     param.requires_grad = False
        print("SAM-Med3D loaded and frozen.")

    def set_image(self, volume_3d):
        """
        Pre-computes features for the 3D volume.
        """
        # self.features = self.model.image_encoder(volume_3d)
        self.volume_shape = volume_3d.shape
        print(f"Computed features for volume of shape {self.volume_shape}")

    def predict_mask(self, points, labels):
        """
        Given 3D points and labels (1 for positive, 0 for negative), predicts the 3D mask.
        
        Args:
            points (list of tuples): List of (z, y, x) points
            labels (list of ints): 1 or 0
        
        Returns:
            np.ndarray: 3D binary mask of shape self.volume_shape
        """
        # mock prediction for now
        # pred_mask = self.model.prompt_encoder_and_mask_decoder(self.features, points, labels)
        
        mask = np.zeros(self.volume_shape, dtype=np.uint8)
        if len(points) > 0:
            z, y, x = points[-1]
            # Simple mock: a small cube around the click
            z_min, z_max = max(0, z-10), min(self.volume_shape[0], z+10)
            y_min, y_max = max(0, y-10), min(self.volume_shape[1], y+10)
            x_min, x_max = max(0, x-10), min(self.volume_shape[2], x+10)
            mask[z_min:z_max, y_min:y_max, x_min:x_max] = 1
            
        return mask
