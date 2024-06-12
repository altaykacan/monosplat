from typing import Dict, Tuple

import torch

from modules.core.interfaces import BaseBackprojector
from modules.segmentation.models import SegmentationModel

class Backprojector(BaseBackprojector):
    def __init__(self, cfg: Dict, intrinsics: Tuple):
        self.cfg = cfg
        self.intrinsics = intrinsics

    def backproject(self, values: torch.Tensor, depths: torch.Tensor, poses: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def compute_backprojection_masks(self, images: torch.Tensor, depths: torch.Tensor) -> torch.Tensor:
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

        # Maximum and minimum depths for backprojection
        max_d = self.cfg.get("max_d", 50.0)
        min_d = self.cfg.get("min_d", 0.0)

        masks = masks & (depths < max_d) & (depths > min_d)

        # Random dropout
        dropout = self.cfg.get("dropout", 0.9)
        if dropout > 0.0:
            masks = masks & (torch.rand((N, C, H, W)).to(device) > dropout)

        return masks


class SemanticBackprojector(Backprojector):
    """
    `Backprojector` that uses a semantic segmentation model to compute
    backprojection masks
    """
    def __init__(self, cfg: Dict, intrinsics: Tuple, segmentation_model: SegmentationModel):
        self.cfg = cfg
        self.intrinsics = intrinsics
        self.segmentation_model = segmentation_model
        self.classes_to_segment = None # TODO implement

    def compute_backprojection_masks(self, images: torch.Tensor, depths: torch.Tensor) -> torch.Tensor:
        geometric_masks = super().compute_backprojection_masks(images, depths)

        semantic_masks = self.segmentation_model({"images": images, "classes_to_segment": self.classes_to_segment})
        # TODO implement
