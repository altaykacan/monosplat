from typing import Dict, Tuple, List

import torch

from modules.core.interfaces import BaseBackprojector, BaseModel
from modules.segmentation.utils import combine_segmentation_masks

class Backprojector(BaseBackprojector):
    def __init__(self, cfg: Dict, intrinsics: Tuple):
        self.cfg = cfg
        self.intrinsics = intrinsics

    def backproject(
            self,
            values: torch.Tensor,
            depths: torch.Tensor,
            poses: torch.Tensor,
            masks: torch.Tensor,
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        N, C, H, W = values.shape
        fx, fy, cx, cy = self.intrinsics
        device = depths.device

        # Pixel coordinates
        U, V = torch.meshgrid((torch.arange(0, W), torch.arange(0, H)), indexing="xy")

        # Move origin to the middle of the image plane
        X = U - cx
        Y = V - cy

        # Undo scaling
        X = X / fx
        Y = Y / fy
        Z = torch.ones_like(X)  # z coordinates will be scaled by depth later

        coords = torch.stack([X, Y, Z], axis=2).double().unsqueeze(0).to(device) # [1 ,H, W, 3]

        # Scales coordinates by depth value
        xyz = depths[:, :, :, :, None] * coords  # [N, 1, H, W, 3]
        xyz = xyz.squeeze(1).permute(0, 3, 1, 2) # [N, 3, H , W]

        # Flatten the output tensors and the masks
        xyz = xyz.reshape(N, 3, H * W)
        values = values.reshape(N, C, H * W)
        masks = masks.reshape(N, 1, H * W)

        # Transform the coordinates to world frame, the poses are T_WC (camera to world)
        R = poses[:, :3, :3] # [N, 3, 3]
        t = poses[:, :3, 3].reshape(N, 3, 1) # [N, 3, 1]
        xyz = torch.bmm(R, xyz) + t

        # Apply masks, points that are ignored have all 0's
        xyz = xyz * masks
        values = values * masks

        return xyz, values

    def compute_backprojection_masks(
            self,
            images: torch.Tensor,
            depths: torch.Tensor,
            depth_scales: torch.Tensor = None,
            depth_shifts: torch.Tensor = None) -> torch.Tensor:
        masks = torch.ones_like(depths).bool()
        device = depths.device
        N, C, H, W = masks.shape # C is 1

        # Region of interest as [top, bottom, left, right] edge indices
        roi = self.cfg.get("roi", [])

        if roi != []:
            masks[:, :, : roi[0], :] = False
            masks[:, :, roi[1] :, :] = False
            masks[:, :, :, : roi[2]] = False
            masks[:, :, :, roi[3] :] = False

        # Maximum and minimum depths (unscaled and unshifted) for backprojection
        max_d = torch.ones((N, 1, 1, 1)).to(device) * self.cfg.get("max_d", 50.0)
        min_d = torch.ones((N, 1, 1, 1)).to(device) * self.cfg.get("min_d", 0.0)

        # Adjust the maximum minimum depth thresholds for current scale
        if (depth_scales is not None) and (depth_shifts is not None):
            max_d = depth_scales * max_d + depth_shifts
            min_d = depth_scales * min_d + depth_shifts

        # Depths are already scaled/shifted
        masks = masks & (depths < max_d) & (depths > min_d)

        # Random dropout
        dropout = self.cfg.get("dropout", 0.9)
        if dropout > 0.0:
            masks = masks & (torch.rand((N, C, H, W)).to(device) > dropout)

        return masks

class DepthBasedDropoutBackprojector(Backprojector):
    def compute_backprojection_masks(
            self,
            images: torch.Tensor,
            depths: torch.Tensor,
            depth_scales: torch.Tensor = None,
            depth_shifts: torch.Tensor = None,
            ) -> torch.Tensor:
        """
        Computes a similar backprojection mask like `compute_backprojection_mask()`
        from the super class  but uses a sigmoid function to determine a
        pixel-wise dropout probability based on the normalized depth values.
        """
        masks = torch.ones_like(depths).bool()
        device = depths.device
        N, C, H, W = masks.shape # C is 1

        # Region of interest as [top, bottom, left, right] edge indices
        roi = self.cfg.get("roi", [])

        if roi != []:
            masks[:, :, : roi[0], :] = False
            masks[:, :, roi[1] :, :] = False
            masks[:, :, :, : roi[2]] = False
            masks[:, :, :, roi[3] :] = False

        # Maximum and minimum depths (unscaled and unshifted) for backprojection
        max_d = torch.ones((N, 1, 1, 1)).to(device) * self.cfg.get("max_d", 50.0)
        min_d = torch.ones((N, 1, 1, 1)).to(device) * self.cfg.get("min_d", 0.0)

        # Adjust the maximum minimum depth thresholds for current scale
        if (depth_scales is not None) and (depth_shifts is not None):
            max_d = depth_scales * max_d + depth_shifts
            min_d = depth_scales * min_d + depth_shifts

        # Depths are already scaled/shifted
        masks = masks & (depths < max_d) & (depths > min_d)

        dropout_coeff = self.cfg.get("dropout_coeff", 0.4)
        dropout_prob_min = self.cfg.get("dropout_prob_min", 0.7)

        # Depth between 0 and 1 now
        normalized_depths = (depths - depths.min()) / (depths.max() - depths.min())
        dropout_probs = (dropout_coeff * torch.tanh(normalized_depths) + dropout_prob_min).clamp(0,1).to(device)
        masks = masks & (torch.rand((N, C, H, W)).to(device) > dropout_probs)

        return masks