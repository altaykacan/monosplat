import logging

import torch
import open3d as o3d
import numpy as np

from modules.io.utils import save_image_torch # useful for debug visualization

log = logging.getLogger(__name__)

# TODO this is buggy, not 100% how the open3d internals work
def project_pcd_o3d(
    pcd: o3d.geometry.PointCloud,
    width: int,
    height: int,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    depth_max: float = 30.0,
    depth_scale: float = 1.0,
) -> torch.Tensor:
    """
    Projects the 3d point clouds into a pinhole camera using open3D
    """
    intrinsics = o3d.core.Tensor(intrinsics.float().numpy())
    extrinsics = o3d.core.Tensor(extrinsics.float().numpy())

    # This function only work with float32, might change in the future
    depth_projection = pcd.project_to_depth_image(
        width,
        height,
        intrinsics,
        extrinsics,
        depth_max=depth_max,  # points above this depth are not projected
        depth_scale=depth_scale,  # factor to multiply the depth values, depth_max filtering is applied before
    )

    # Need to convert from open3d Tensor to torch Tensor
    depth_projection = torch.tensor(np.asarray(depth_projection.to_legacy()))

    return depth_projection


def centered_moving_average(array: np.ndarray, window_size: int) -> np.ndarray:
    """
    Computes the centered moving average of a 1D numpy array.
    Uses 'same' 1D convolutions so the arrays are mirrored on the boundaries.

    Parameters:
    - array: The input 1D array
    - window_size: The size of the moving window.
    """
    moving_average = np.convolve(array, np.ones(window_size), "same") / window_size
    return moving_average