"""Has implementations of standard map classes for the framework"""
from pathlib import Path
import logging
from typing import Dict, Union, Optional

import torch
import numpy as np
import open3d as o3d

from modules.core.interfaces import BaseMap
from modules.core.utils import unravel_batched_pcd_tensor

log = logging.getLogger(__name__)


class PointCloud(BaseMap):
    """
    Simple open3D based pointcloud implementation that stores the point coordinates,
    and optional color and normal values as `[C, num_points]` tensors. When adding
    points to the cloud, batched tensors with shape `[N, C, H * W]` are
    concatenated together into one big cloud of shape `[C, N * H * W]`.

    Points mapped to `[0,0,0]` are ignored.
    """
    def __init__(self, scale: float = 1.0):
        self.xyz: Optional[torch.Tensor] = None # all saved as [3, num_points] tensors
        self.rgb: Optional[torch.Tensor] = None
        self.normals: Optional[torch.Tensor] = None
        self.is_road: Optional[torch.Tensor]  = None
        self.scale: Optional[float] = scale

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

    @property
    def has_road(self):
        return self.is_road is not None

    def increment(
            self,
            xyz: torch.Tensor,
            rgb: Optional[torch.Tensor] = None,
            normals: Optional[torch.Tensor] = None,
            is_road: Optional[torch.Tensor] = None
            ) -> None:
        # Deal with batched input, convert from [N, C, num_el] to [C, N * num_el]
        if len(xyz.shape) == 3:
            N, _, HW = xyz.shape
            xyz = unravel_batched_pcd_tensor(xyz)
            rgb = unravel_batched_pcd_tensor(rgb) if rgb is not None else None
            normals = unravel_batched_pcd_tensor(normals) if normals is not None else None
            is_road = unravel_batched_pcd_tensor(is_road) if is_road is not None else None

        # Initialize or concatenate to existing tensors
        if not self.is_initialized:
            self.xyz = xyz
            self.rgb = rgb
            self.normals = normals
            self.is_road = is_road
        else:
            self.xyz = torch.cat((self.xyz, xyz), dim=1)

            # If the additional attributes exist, keep adding them, add dummy values if they are missing
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

            if self.has_road:
                if is_road is not None:
                    self.is_road = torch.cat((self.is_road, is_road), dim=1)
                else:
                    self.is_road = torch.cat((self.is_road, torch.zeros_like(is_road)), dim=1)

        # Remove masked points, i.e. boolean columns that sum up to 0 (all zeros)
        valid_points_mask = (self.xyz.bool().sum(dim=0) != 0)
        self.xyz = self.xyz[:, valid_points_mask]
        self.rgb = self.rgb[:, valid_points_mask] if self.rgb is not None else None
        self.normals = self.normals[:, valid_points_mask] if self.normals is not None else None
        self.is_road = self.is_road[:, valid_points_mask] if self.is_road is not None else None

        return None

    def transform(self, T: torch.Tensor) -> None:
        R = T[:3, :3] # [3, 3]
        t = T[:3, 3].unsqueeze(1) # [3, 1], already scaled in the dataset

        self.xyz = R @ self.xyz + t
        self.normals = R @ self.normals # rotate the normals

        return None

    def postprocess(self):
        # Convert to open3d cloud, open3d expects [N, 3] numpy arrays but we have [3, N]
        pcd = o3d.t.geometry.PointCloud()
        pcd.point.positions = self.xyz.permute(1, 0).cpu().numpy()
        if self.has_rgb:
            # open3d wants normalized float rgb values
            if self.rgb.dtype == torch.uint8:
                self.rgb = self.rgb.float()
                self.rgb = self.rgb / 255
            pcd.point.colors = self.rgb.permute(1, 0).cpu().numpy()

        if self.has_normals:
            self.normals = self.normals / torch.linalg.norm(self.normals, dim=0).reshape(1, -1)
            pcd.point.normals = self.normals.float().permute(1, 0).cpu().numpy()

        if self.has_road:
            pcd.point.is_road = self.is_road.permute(1, 0).int().cpu().numpy()

        self.pcd = pcd

    def add_sky_dome(
        self,
        num_points: int = 50000,
        radius_factor: float = 1.0,
        color: tuple = (173, 216, 230),
        ):
        """
        Adds a hemisphere around the input dense point cloud `pcd_in`. The center is
        computed to be the mean of all the coordinates and the radius is half of
        the largest distance of the minimum and maximum coordinates. The
        radius gets scaled by `radius_factor`.

        The uniform points are generated per the last method in this page, i.e.
        by generating 3 normally distributed variables and normalizing them:
        https://mathworld.wolfram.com/SpherePointPicking.html

        Since we are sampling around a hemisphere, double the amount of points
        specified are sampled and the points outside of the hemisphere are discarded.
        """
        if getattr(self, "pcd") is None:
            raise RuntimeError("You need to first run 'PointCloud.postprocess()' before adding the skydome!")
        device = self.xyz.device

        # Uses self.xyz which does not get changed when we add the initial SfM cloud
        max_coords, _ = torch.max(self.xyz, dim=1, keepdim=True)  # [3, 1]
        min_coords, _ = torch.min(self.xyz, dim=1, keepdim=True)  # [3, 1]

        center = torch.mean(self.xyz, dim=1, keepdim=True).to(device)  # [3, 1]

        radius = torch.max(torch.abs(max_coords - min_coords)).to(device) # scalar
        radius *= radius_factor

        # Times two because we will discard (roughly) half of the generated points
        coords = torch.randn(3, 2 * num_points).to(device)
        coords /= torch.linalg.norm(coords, dim=0, keepdim=True) # normalize each point

        # Shifting the unit sphere to the cloud center and scaling the radius
        coords = center + radius * coords  # [3, num_kept_points], point coordinates

        # Only take the points from the dome with y coordinates below maximum y (y points down, in gravity direction)
        y_max = self.xyz[1, :].max()
        valid_points = (coords[1, :] < (1.5 * y_max))
        coords = coords[:, valid_points] # keep the valid rows
        num_kept_points = coords.shape[1]

        # Data type has to be float rgb values for open3d
        dome_rgbs = torch.ones((1, num_kept_points), dtype=torch.uint8) * torch.tensor(color, dtype=torch.uint8).view(3, 1)
        dome_rgbs = dome_rgbs.float() / 255

        skydome = o3d.t.geometry.PointCloud()
        skydome.point.positions = coords.permute(1, 0).cpu().numpy()
        skydome.point.colors = dome_rgbs.permute(1, 0).cpu().numpy()

        # Need to add dummy normals otherwise we can't merge the pointclouds
        if self.has_normals:
            skydome.point.normals = np.zeros_like(coords.permute(1, 0).float().cpu().numpy())

        if self.has_road:
            skydome.point.is_road = torch.zeros((num_kept_points, 1)).int().cpu().numpy()

        self.pcd = self.pcd + skydome # open3d 18.0 allows concatenating pointclouds like this

    def add_init_cloud(self, init_ply_path: Union[str, Path]) -> None:
        """
        Adds an initial point cloud *without* scaling to the existing point cloud.
        This means that the initial cloud and the current cloud is expected
        to have the same scale (if the initial cloud has the same scale as your
        poses this means your `pose_scale` should be 1 and you should set a
        `depth_scale`)
        """
        init_cloud = o3d.t.io.read_point_cloud(str(init_ply_path))

        # We save positions as float64
        if init_cloud.point.positions.dtype == o3d.core.Dtype.Float32:
            init_cloud.point.positions = init_cloud.point.positions.to(o3d.core.Dtype.Float64)

        # Need to make sure we have the right attributes otherwise we can't concat
        xyz = np.asarray(init_cloud.point.positions.to(o3d.core.Dtype.Float32)) # we need float32 for colors and normals
        N, _ = xyz.shape
        if self.has_rgb:
            try:
                # We have colors as float32 values between 0 and 1
                if init_cloud.point.colors.dtype == o3d.core.Dtype.UInt8:
                    init_cloud.point.colors = init_cloud.point.colors.to(o3d.core.Dtype.Float32) / 255.0

            except Exception as E:
                init_cloud.point.colors = torch.zeros_like(xyz).float().cpu().numpy()

        if self.has_normals:
            try:
                # We have normals as float32
                init_cloud.point.normals = init_cloud.point.normals.to(o3d.core.Dtype.Float32)

            except Exception as E:
                init_cloud.point.normals = torch.zeros_like(xyz).float().cpu().numpy()

        if self.has_road:
            init_cloud.point.is_road = torch.zeros((xyz.shape[0], 1)).int().cpu().numpy()

        self.pcd = self.pcd + init_cloud # open3d 18.0 allows concatenating pointclouds like this

    def downsample(
            self,
            voxel_size: float = 0.05,
            depth_scale: float = 1.0,
            remember_init_cloud: bool = False
            ) -> None:
        """
        Downsamples the open3D point cloud that is created after running `postprocess()`
        using voxel downsampling. Optionally saves the point cloud before
        downsampling.
        """
        if getattr(self, "pcd") is None:
            raise RuntimeError("You need to first run 'PointCloud.postprocess()' before downsampling!")

        init_num_points = len(self.pcd.point.positions)

        # To convert metric scale to up-to-scale pose's scale
        scaled_voxel_size = voxel_size * depth_scale

        if remember_init_cloud:
            self.pcd_before_downsample = self.pcd

        self.pcd = self.pcd.voxel_down_sample(voxel_size=scaled_voxel_size)
        downsampled_num_points = len(self.pcd.point.positions)

        log.info(f"Reduced {init_num_points} points to {downsampled_num_points} after downsampling with a voxel size of: {voxel_size} (scaled size: {scaled_voxel_size:0.5f})")

    def clean(self, num_neighbors: int = 20, std_deviation: float = 2.0, remember_init_cloud: bool = False) -> None:
        """
        Cleans the open3D point cloud that is created after running `postprocess()`
        using statistical outlier removal. Optionally saves the point cloud before
        cleaning.
        """
        if getattr(self, "pcd") is None:
            raise RuntimeError("You need to first run 'PointCloud.postprocess()' before cleaning your map!")

        if remember_init_cloud:
            self.pcd_before_clean = self.pcd # remember cloud before cleaning, useful for debugging
        self.pcd, _ = self.pcd.remove_statistical_outliers(num_neighbors, std_deviation)

    def save(self, filename: Union[Path, str] = "map.ply", output_dir: Union[Path, str] = "."):
        if self.pcd is None:
            raise RuntimeError("You need to first run 'PointCloud.postprocess()' before saving your map!")

        if isinstance(filename, str):
            filename = Path(filename)
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)

        file_path = str(output_dir / filename) # open3d needs paths as strings
        o3d.t.io.write_point_cloud(file_path, self.pcd)
        log.info(f"Wrote point cloud to disk at {file_path} with {len(self.pcd.point.positions)} points!")