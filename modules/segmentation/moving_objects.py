import logging
from pathlib import Path
from typing import List, Dict, Union

import torch
import matplotlib.pyplot as plt

from modules.core.visualization import visualize_flow
from modules.core.utils import format_intrinsics, get_correspondences_from_flow, shrink_bool_mask
from modules.io.utils import save_pcd_o3d, save_image_torch # useful for debug

log = logging.getLogger(__name__)

def compute_dists(
        depth_a: torch.Tensor,
        depth_b: torch.Tensor,
        pose_a: torch.Tensor,
        pose_b: torch.Tensor,
        flow: torch.Tensor,
        intrinsics: List[float],
        min_flow: float = 0.5,
        occlusion_mask: torch.Tensor = None,
        ) -> torch.Tensor:
    """
    Computes the 3D distances between corresponding backprojected pixels.
    Correspondences are determined based on optical flow (`flow`). Masks
    The result is returned as a `[N, 1, H, W]` tensor where each pixel in position
    `(i, j)` has the 3D distance from its backprojected 3D point (by `depth_a`)
    to the backprojected 3D point (by `depth_b`). Masked values are set to zero.
    """
    device = depth_a.device
    N, _, H, W = depth_a.shape

    U, V = torch.meshgrid((torch.arange(W), torch.arange(H)), indexing="xy") # pixel coords
    U = U.to(device)
    V = V.to(device)
    Z = torch.ones_like(U)

    # [1, 3, H, W], needed for backprojection, these are the pixel coordinates and unit depths
    UVZ = (torch.stack((U, V, Z), dim=0).unsqueeze(0).to(device))

    T_WA = pose_a # [N, 4, 4] tensors
    T_WB = pose_b
    T_AB = torch.inverse(T_WA) @ T_WB  # T_AW @ T_WB == T_AB

    # Backprojection
    K = format_intrinsics(intrinsics)  # [3 ,3]
    K = K.to(device)

    # [N, 3, num_points], in frame A
    pcd_a  = torch.matmul(torch.inverse(K), (depth_a * UVZ).reshape(N, 3, -1).double())

    # [N, 3, num_points], in frame B
    pcd_b = torch.matmul(torch.inverse(K), (depth_b * UVZ).reshape(N, 3, -1).double())

    # Coordinate transform
    ones = torch.ones((N, 1, H * W)).to(device) # for homogeneous coordinates
    pcd_b = torch.concat((pcd_b, ones), dim=1) # [N, 4, num_points]

    pcd_b_raw = pcd_b.clone()[:, :3, :] # for debug
    pcd_b = T_AB @ pcd_b # from frame B to frame A for distance comparison
    pcd_b = pcd_b[:, :3, :] # remove the row of ones
    pcd_a = pcd_a.reshape(N, 3, H, W) # we need to reshape the tensor for indexing later
    pcd_b = pcd_b.reshape(N, 3, H, W)

    # Compute dense pixel correspondences from flow
    correspondences = get_correspondences_from_flow(flow, H, W) # [N, 2, H, W]

    U_corr = correspondences[:, 0, :, :].int() # integer tensors for indexing
    V_corr = correspondences[:, 1, :, :].int()

    corr_mask = ((U_corr > 0) & (U_corr < (W - 1)) & (V_corr > 0) & (V_corr < (H - 1))).to(device)
    corr_mask = corr_mask[:, None, :, :] # used for indexing the pointcloud [N, 1, H, W]
    flow_norm = torch.linalg.norm(flow, dim=1, keepdims=True) # [N, 1, H, W]
    flow_mask = (flow_norm >= min_flow)

    # Occlusions cause spurious depth deviations that we want to ignore
    if occlusion_mask is None:
        occlusion_mask = torch.ones_like(flow_mask).bool()
    else:
        # Grows the False regions which represent the occluded pixels
        occlusion_mask = shrink_bool_mask(occlusion_mask, iterations=3)

    mask = corr_mask & flow_mask &  occlusion_mask # mask of pixels to include

    # TODO figure out a way to do this with batched tensors to increase speed, not too high priority
    # Batch-wise dense point matching and 3D distance computation
    dists = torch.zeros_like(depth_a)
    for batch in range(N):
        U_corr_batch = U_corr[batch, ...] # [H, W]
        V_corr_batch = V_corr[batch, ...] # [H, W]

        pcd_a_batch = pcd_a[batch, ...] # [3, H, W]
        pcd_b_batch = pcd_b[batch, ...] # [3, H, W]

        mask_batch = corr_mask[batch, 0, ...] # [H ,W]

        # Use the corresponding coordinate tensors to create a new (3, H, W) tensor to do dense point matching
        U_orig = U[mask_batch] # [num_matches], contains [i_1, i_2, ...]
        V_orig = V[mask_batch] # [num_matches] contains [j_1, j_2, ...]
        U_new = U_corr_batch[mask_batch] # [num_matches], contains [i_1_corr, i_2_corr, ...]
        V_new = V_corr_batch[mask_batch] # [num_matches], contains [j_1_corr, j_2_corr, ...]

        # Gets the entries from pcd_b_batch with indices (i_corr, j_corr) and
        # places them at (i, j) on a new tensor pcd_b_corr_batch, to match
        # 3d point coordinate entries at (i, j) in pcd_a_batch
        pcd_b_corr_batch = torch.zeros_like(pcd_b_batch)

        # [3, H, W], V is the row index, U is the column
        pcd_b_corr_batch[:, V_orig, U_orig] = pcd_b_batch[:, V_new, U_new]
        dists[batch, 0, :, :] = torch.linalg.norm((pcd_a_batch - pcd_b_corr_batch), dim=0)

    dists[torch.logical_not(mask)] = 0
    return dists


def segment_moving_objects(
    mask_moveable_a: torch.Tensor,
    mask_moveable_b: torch.Tensor,
    depth_a: torch.Tensor,
    depth_b: torch.Tensor,
    pose_a: torch.Tensor,
    pose_b: torch.Tensor,
    forward_flow: torch.Tensor,
    intrinsics: List,
    instance_masks: List,
    max_d: float = 50.0,
    min_flow: float = 0.0,
    delta_thresh: float = 1.75,
    percentage_thresh: float = 0.80,
    tracking_percentage_thresh: float = 0.80,
    prev_mask_moving: torch.Tensor = None,
    output_dir: Path = None,
    image_id: str = None,
    occlusion_mask: torch.Tensor = None,
    disocclusion_mask: torch.Tensor = None,
    debug: bool = False,
    original_image: torch.Tensor = None, # for debug purposes
):
    # TODO implement
    device = "cuda" if torch.cuda.is_available() else "cpu"
    N, _, H, W = depth_a.shape  # [N, 3, H, W] tensor

    # TODO implement
    assert N == 1, "Moving object masking not implemented for batched inputs properly, please pass a single image!"

    # Useful for plotting the flows
    if original_image is not None:
        original_image = original_image.to(device)

    # dists has the same shape as depths [N, 1, H, W]
    dists = compute_dists(depth_a, depth_b, pose_a, pose_b, forward_flow, intrinsics, min_flow, occlusion_mask)
    dists[dists == torch.inf] = 0

    # Auxilary tensor to only consider pixels from dists below the depth threshold to include in the mean computation
    dists_aux = dists.clone()

    # Both depth_a and max_d are with the depth model scale
    dists_aux[depth_a > max_d] = 0

    # Getting the mean of all non-zero elements below max_d
    dists_avg = torch.mean(dists[dists_aux.nonzero(as_tuple=True)])
    dists_avg_mask = dists > (dists_avg * delta_thresh) # will be compared with instance masks
    mask = torch.zeros_like(depth_a, dtype=bool)

    for i, instance_mask in enumerate(instance_masks):
        instance_mask_size = instance_mask.sum() # pixel count for current group
        intersection_size = (dists_avg_mask & instance_mask).sum()

        if prev_mask_moving is not None:
            intersection_prev_size = (prev_mask_moving & instance_mask).sum()

        # If the instance mask and the dists_mask agree, take the whole instance as a moving object
        if (intersection_size / instance_mask_size) > percentage_thresh:
            mask = torch.logical_or(instance_mask & mask_moveable_a, mask)

        # If the instance mask intersects the previous moving object mask above a threshold, we take it as moving object too
        if prev_mask_moving is not None and (intersection_prev_size / instance_mask_size) > tracking_percentage_thresh:
            mask = torch.logical_or(instance_mask & mask_moveable_a, mask)

        if debug:
            plt.imsave(f"{str(output_dir)}/{image_id}_debug_10_instance_{i}.png", (instance_mask * original_image).squeeze().permute(1,2,0).cpu().numpy())

    if debug:
        dists_mask = dists > delta_thresh  # just here for debug and comparison
        plt.imsave(f"{str(output_dir)}/{image_id}_debug_01_depth_a.png", (1/depth_a).squeeze().cpu().numpy())
        plt.imsave(f"{str(output_dir)}/{image_id}_debug_02_depth_b.png", (1/depth_b).squeeze().cpu().numpy())
        plt.imsave(f"{str(output_dir)}/{image_id}_debug_03_dists_naive_threshold.png", dists_mask.squeeze().cpu().numpy())
        plt.imsave(f"{str(output_dir)}/{image_id}_debug_04_dists_values.png", dists.squeeze().cpu().numpy())
        plt.imsave(f"{str(output_dir)}/{image_id}_debug_04_dists_inv_values.png", (1/(dists + 0.0001)).squeeze().clamp(0, 10).cpu().numpy()) # dists should be greater than 0.1
        plt.imsave(f"{str(output_dir)}/{image_id}_debug_05_dists_avg_mask.png", (dists_avg_mask).squeeze().cpu().numpy())
        plt.imsave(f"{str(output_dir)}/{image_id}_debug_06_dists_avg_values_masked.png", (dists * dists_avg_mask).squeeze().cpu().numpy())
        plt.imsave(f"{str(output_dir)}/{image_id}_debug_06_dists_avg_values_used.png", (dists_aux).squeeze().cpu().numpy())
        visualize_flow(forward_flow, original_image, output_path=f"{str(output_dir)}/{image_id}_debug_07_forward_flow")

        if occlusion_mask is not None:
            plt.imsave(f"{str(output_dir)}/{image_id}_debug_08_occlusion_mask.png", (occlusion_mask * original_image).squeeze().permute(1,2,0).cpu().numpy())

        if disocclusion_mask is not None:
            plt.imsave(f"{str(output_dir)}/{image_id}_debug_08_disocclusion_mask.png", (disocclusion_mask * original_image).squeeze().permute(1,2,0).cpu().numpy())

        plt.imsave(f"{str(output_dir)}/{image_id}_debug_09_prev_moving_mask.png", (prev_mask_moving * original_image).squeeze().permute(1,2,0).cpu().numpy())

    return mask