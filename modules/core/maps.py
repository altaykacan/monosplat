"""Has implementation of standard map classes for the framework"""
from pathlib import Path
from typing import Dict, Union

import torch
import numpy as np
import open3d as o3d

from modules.core.interfaces import BaseMap
from modules.core.utils import unravel_batched_pcd_tensor


class PointCloud(BaseMap):
    """
    Simple open3D based pointcloud implementation that stores the point coordinates,
    and optional color and normal values as `[C, num_points]` tensors. When adding
    points to the cloud, batched tensors with shape `[N, C, H * W]` are
    concatenated together into one big cloud of shape `[C, N * H * W]`.

    Points mapped to `[0,0,0]` are ignored.
    """
    def __init__(self, scale: float = 1.0):
        self.xyz = None
        self.rgb = None
        self.normals = None
        self.scale = scale

    @property
    def num_points(self):
        if self.xyz is None:
            return 0
        else:
            return self.xyz.shape[1]

    @property
    def is_initialized(self):
        return self.xyz is not None

    @property
    def has_rgb(self):
        return self.rgb is not None

    @property
    def has_normals(self):
        return self.normals is not None

    def increment(self, xyz: torch.Tensor, rgb: torch.Tensor = None, normals: torch.Tensor = None):
        # Deal with batched input, convert from [N, C, num_el] to [C, N * num_el]
        if len(xyz.shape) == 3:
            N, _, HW = xyz.shape
            xyz = unravel_batched_pcd_tensor(xyz)
            rgb = unravel_batched_pcd_tensor(rgb) if rgb is not None else None
            normals = unravel_batched_pcd_tensor(normals) if normals is not None else None

        # Initialize or concatenate to existing tensors
        if not self.is_initialized:
            self.xyz = xyz
            self.rgb = rgb
            self.normals = normals
        else:
            self.xyz = torch.cat((self.xyz, xyz), dim=1)

            if self.has_rgb:
                if rgb is not None:
                    self.rgb = torch.cat((self.rgb, rgb), dim=1)
                else:
                    self.rgb = torch.cat((self.rgb, torch.zeros_like(xyz)), dim=1)

            if self.has_normals:
                if normals is not None:
                    self.normals = torch.cat((self.normals, normals), dim=1)
                else:
                    self.normals = torch.cat((self.normals, torch.zeros_like(normals)), dim=1)

        # Remove masked points, i.e. boolean columns that sum up to 0 (all zeros)
        valid_points_mask = (self.xyz.bool().sum(dim=0) != 0)
        self.xyz = self.xyz[:, valid_points_mask]
        self.rgb = self.rgb[:, valid_points_mask] if self.rgb is not None else None
        self.normals = self.normals[:, valid_points_mask] if self.normals is not None else None

        return None

    def transform(self, T: torch.Tensor):
        R = T[:3, :3] # [3, 3]
        t = T[:3, 3].unsqueeze(1) # [3, 1], already scaled in the dataset

        self.xyz = R @ self.xyz + t
        self.normals = R @ self.normals # rotate the normals

        return None

    def postprocess(self):
        # Convert to open3d cloud, open3d expects [N, 3] numpy arrays
        pcd = o3d.t.geometry.PointCloud()
        pcd.point.positions = self.xyz.permute(1, 0).cpu().numpy()

        if self.has_rgb:
            # open3d wants normalized float rgb values
            if self.rgb.dtype == torch.uint8:
                self.rgb = self.rgb.float()
                self.rgb = self.rgb / 255
            pcd.point.colors = self.rgb.permute(1, 0).cpu().numpy()

        if self.has_normals:
            pcd.point.normals = self.normals.permute(1, 0).cpu().numpy()

        self.pcd = pcd

    def clean(self, num_neighbors: int = 20, std_deviation: float = 2.0, save_init_cloud: bool = False):
        """
        Cleans the open3D point cloud that is created after running `postprocess()`
        using statistical outlier removal. Optionally saves the point cloud before
        cleaning.
        """
        if self.pcd is None:
            raise RuntimeError("You need to first run 'PointCloud.postprocess()' before cleaning your map!")

        if save_init_cloud:
            self.prev_pcd = self.pcd # remember cloud before cleaning, useful for debugging
        self.pcd, _ = self.pcd.remove_statistical_outliers(num_neighbors, std_deviation)

    def save(self, filename: Union[Path, str] = "map.ply", output_dir: Union[Path, str] = "."):
        if self.pcd is None:
            raise RuntimeError("You need to first run 'PointCloud.postprocess()' before saving your map!")

        if isinstance(filename, str):
            filename = Path(filename)
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)

        file_path = str(output_dir / filename) # open3d needs paths as strings
        o3d.io.write_point_cloud(file_path, self.pcd.to_legacy())