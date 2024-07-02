import logging
from pathlib import Path
from typing import List, Union, Tuple

import torch
import numpy as np
import open3d as o3d
from tqdm import tqdm

from modules.core.utils import format_intrinsics
from modules.io.datasets import ColmapDataset
from modules.io.utils import read_ply, read_ply_o3d, save_image_torch
from modules.scale_alignment.utils import project_pcd_o3d
from modules.scale_alignment.visualization import save_scale_plot, save_histogram

log = logging.getLogger(__name__)


# Originally from monosdf, taken from DN-Splatter (https://github.com/maturk/dn-splatter/blob/bc83595b0685f0fba0f0b440d510bb9123a8fdb2/dn_splatter/scripts/align_depth.py#L187)
# Here for double-checking our implementation
def compute_scale_and_shift(prediction, target, mask):
    # System matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # Right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # Solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def least_squares_only_scale(
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
        ) -> torch.Tensor:
    """
    Solves the linear least squares equation per-depth to find an optimal
    scale factor to align the predicted depths to the sparse depths.

    For each pixel in a given frame, we solve `argmin((t_i - scale_j * p_i)**2).sum(dim=i))`
    where `i` indexes the pixels and `j` indexes the frames. We take the sum over
    all pixels in the given frame

    Args:
      `prediction`: The predicted depths as a  `[num_frames, H, W]` tensor
      `target`: The projected sparse depths as a `[num_frames, H, W]` tensor
      `mask`: The mask specifying which pixels to use. Also `[num_frames, H, W]`
    """
    scales = []
    # TODO probably can vectorize this
    for frame_idx in range(prediction.shape[0]):
        t = target[frame_idx] # [H, W]
        p = prediction[frame_idx]
        m = mask[frame_idx]
        a = t[m].view(-1, 1) # target vector, [num_sparse_depths, 1]
        b = p[m].view(-1, 1) # input vectors, [num_sparse_depths]
        B = b


        # Now we have a least squares problem (a - Bx).T (a - Bx)
        try:
            solution = torch.linalg.inv(B.T @ B) @ B.T @ a
            scales.append(solution[0])
        except Exception as E:
            log.warning(f"Encountered exception {E} while trying to solve for sparse scale alignment. Skipping frame idx {frame_idx}.")
            scales.append(solution[0]) # keep the last computed value, otherwise we get messed up results


    scales = torch.cat(scales, dim=0)[:, None, None] # [num_frames, 1, 1]

    return scales.squeeze()


def least_squares_scale_and_shift(
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Solves the linear least squares equation per-depth to find an optimal
    scale and shift factor to align the predicted depths to the sparse depths.

    For each pixel in a given frame, we solve `argmin((t_i - (scale_j * p_i + shift_j)**2).sum(dim=i))`
    where `i` indexes the pixels and `j` indexes the frames. We take the sum over
    all pixels in the given frame

    Args:
      `prediction`: The predicted depths as a  `[num_frames, H, W]` tensor
      `target`: The projected sparse depths as a `[num_frames, H, W]` tensor
      `mask`: The mask specifying which pixels to use. Also `[num_frames, H, W]`
    """
    scales = []
    shifts = []
    # TODO probably can vectorize this
    for frame_idx in range(prediction.shape[0]):
        t = target[frame_idx] # [H, W]
        p = prediction[frame_idx]
        m = mask[frame_idx]
        a = t[m].view(-1, 1) # target vector, [num_sparse_depths, 1]
        b = p[m].view(-1) # input vectors, [num_sparse_depths]
        col_of_ones = torch.ones_like(b)
        B = torch.stack((b, col_of_ones), dim=1) # [num_sparse_depths, 2]

        # Now we have a least squares problem (a - Bx).T (a - Bx)
        try:
            solution = torch.linalg.inv(B.T @ B) @ B.T @ a
            scales.append(solution[0])
            shifts.append(solution[1])
        except Exception as E:
            log.warning(f"Encountered exception {E} while trying to solve for sparse scale alignment. Skipping frame idx {frame_idx}.")
            scales.append(solution[0]) # keep the last computed value, otherwise we get messed up results


    scales = torch.cat(scales, dim=0)[:, None, None] # [num_frames, 1, 1]
    shifts = torch.cat(shifts, dim=0)[:, None, None]  # [num_frames, 1, 1]

    return scales.squeeze(), shifts.squeeze()


def do_sparse_alignment(
        poses: torch.Tensor,
        stamps: List[int],
        sparse_cloud_path: Union[Path, str],
        depth_paths: List[Path],
        intrinsics: List[float],
        ignore_masks: torch.Tensor = None,
        min_d: float = 0.0,
        max_d: float = 30.0,
        log_dir: Path = Path("./alignment_plots"),
        return_only_scale: bool = True,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Finds and returns optimal scale and shift factors for every frame with an
    associated pose. For each frame minimizes the discrepancies between the
    projected depths from a sparse point cloud and the dense depth predictions.
    Expects the `poses` and the sparse cloud to be in the same world coordinate
    system.

    `poses` is an `[num_frames, 4, 4]` tensor. The `intrinsics` is the pinhole
    camera intrinsics as a `[fx, fy, cx, cy]` list. Ignores pixels that are
    marked via `mask` (if the value is `True` it is ignored in scale computation)
    and depths above a maximum depth threshold.
    """
    if sparse_cloud_path is None:
        raise ValueError("You need to specify the path to a sparse reconstruction to align scales with if you want to use 'sparse' alignment.")

    # Read in the sparse cloud
    if sparse_cloud_path.name == "points3D.txt":
        pcd = ColmapDataset.read_colmap_pcd_o3d(sparse_cloud_path, convert_to_float32=True)
    else:
        pcd = read_ply_o3d(sparse_cloud_path, convert_to_float32=True)

    # Iterate over the depth directory, only keep frames that we have poses for
    dense_depths = []
    depth_ids = [int(p.stem) for p in depth_paths]
    log.info("Reading in precomputed depth values for sparse scale alignment, this might take some time...")
    for curr_stamp in tqdm(stamps):
        depth_idx = depth_ids.index(curr_stamp)
        depth = torch.tensor(np.load(depth_paths[depth_idx]))
        dense_depths.append(depth)

    dense_depths = torch.stack(dense_depths, dim=0).float() # [num_frames, H, W]

    # Do some setup with the read-in depth information
    _, H, W = dense_depths.shape
    if ignore_masks is None:
        ignore_masks = torch.zeros_like(dense_depths)

    K = format_intrinsics(intrinsics) # [3, 3]

    # Get sparse depths for each posed frame from sparse cloud
    sparse_depths = []
    log.info("Extracting sparse depths for sparse scale alignment, this might take some time...")
    for curr_pose in tqdm(poses):
        sparse_depth = project_pcd_o3d(pcd, W, H, K, torch.linalg.inv(curr_pose), depth_max=max_d)

        # max_d might be with a different scale, need to limit and take the 10% of the depths
        if sparse_depth.max() < (0.5 * max_d):
            sparse_max = sparse_depth.max()
            new_max_d = sparse_max * 10 / 100
            sparse_depth = project_pcd_o3d(pcd, W, H, K, torch.linalg.inv(curr_pose), depth_max=new_max_d)

        sparse_depths.append(sparse_depth.squeeze())

    sparse_depths = torch.stack(sparse_depths, dim=0) # [num_frames, H, W]

    # Compute alignment
    mask = (sparse_depths > 0.01) & (dense_depths > min_d) & (dense_depths < max_d) & torch.logical_not(ignore_masks)
    # TODO remove option to compute scale and shift, only scale is better!
    scale, shift = least_squares_scale_and_shift(dense_depths, sparse_depths, mask)
    only_scale = least_squares_only_scale(dense_depths, sparse_depths, mask)
    scale_check, shift_check = compute_scale_and_shift(dense_depths, sparse_depths, mask) # to double check own results with the DN-splatter implementation

    save_scale_plot([(1 / (scale + 0.001)).clamp(-10,1000)], stamps, "Scales", filename="scale_and_shift", output_dir=log_dir)
    save_scale_plot([1 / (only_scale + 0.001)], stamps, "Scales", filename="only_scale", output_dir=log_dir)

    log.info("===============")
    log.info(f"FINISHED SPARSE SCALE ALIGNMENT, MEAN OF COMPUTED SCALES IS: { (1 / (only_scale + 0.001)).mean().item():0.4f}")
    log.info("===============")

    if return_only_scale:
        return only_scale, torch.zeros_like(only_scale)
    else:
        return scale, shift