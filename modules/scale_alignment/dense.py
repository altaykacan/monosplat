import logging
from typing import Union, List, Tuple, Dict
from pathlib import Path


import torch
import torch.nn.functional as F
from tqdm import tqdm

from modules.core.models import RAFT
from modules.core.interfaces import BaseDataset, BaseModel
from modules.core.visualization import visualize_flow
from modules.core.utils import format_intrinsics, compute_occlusions
from modules.depth.models import PrecomputedDepthModel, KITTI360DepthModel
from modules.eval.utils import compute_mean_recursive
from modules.io.datasets import CombinedColmapDataset, KITTI360Dataset
from modules.io.utils import save_image_torch
from modules.segmentation.models import SegFormer, PrecomputedSegModel
from modules.segmentation.utils import combine_segmentation_masks
from modules.scale_alignment.visualization import save_histogram, save_scale_plot, plot_dense_alignment_results


log = logging.getLogger(__name__)

def do_dense_alignment(
    dataset: BaseDataset,
    flow_steps: List[int],
    t_z_thresh: float = 0.2, # was 0.5 before when we used it for ds01 and kitti360_0_mini
    min_d: float = 0.0,
    max_d: float = 50.0,
    min_flow: float = 0.2,
    mask_moveable_objects: bool = True,
    seg_model_type: str = "predict",
    mask_occlusions: bool = True,
    mask_disocclusions: bool = False,
    log_dir: Path = Path("./alignment_plots"),
    frame_idx_for_hist: int = 8,
    log_every: int = 50,
    debug: bool = False,
    depth_type: str = "gt",
    ) -> Dict:
    """
    Does dense scale alignment for each pair of images for multiple flow steps.
    Computes the dense correspondences usign optical flow. For each pair, a
    forward and a backward computation is carried out and averaged over.

    The means of the cleaned scales are saved in a dictionary structured like:
    ```
    scales:{
            flow_step: {
                frame_id_1: scalar_scale_value,
                frame_id_2: scalar_scale_value,
                ...,
                frame_id_n: scalar_scale_value,
            }
        }
    ```

    This allows you to create mean scale values vs. frame id plots as line plots.
    You can also visualize histograms to see the distribution of the scales.
    For that we save the per-frame cleaned scale values (not the mean) as
    a dictionary too:
    ```
    frame_scales_for_histogram: {
        flow_step: {
            frame_id_1: {
                forward: 1D_scale_tensor,
                backward: 1D_scale_tensor
                },
            frame_id_2: {
                forward: 1D scale-tensor,
                backward: 1D scale_hensor
                },
            ...,
            frame_id_n: {
                forward: 1D scale_tensor,
                backward: 1D scale_tensor
                }
    ```
    """
    scales = {}
    frame_scales_for_hist = {}
    scale_tensors_to_plot = {}
    tz_to_plot = {}
    std_to_plot = {}
    if debug:
        frame_ids_for_hist = [dataset.frame_ids[i] for i in [frame_idx_for_hist, 0, len(dataset) // 8, len(dataset) // 4]]
    else:
        frame_ids_for_hist = [dataset.frame_ids[i] for i in [frame_idx_for_hist, 0, len(dataset) // 4, len(dataset) // 2, - (len(dataset) // 4)]]
    flow_model = RAFT()

    if isinstance(dataset, KITTI360Dataset) and depth_type == "gt":
        depth_model = KITTI360DepthModel(dataset)
    else:
        depth_model = PrecomputedDepthModel(dataset)

    if mask_moveable_objects and seg_model_type == "precomputed":
        seg_model = PrecomputedSegModel(dataset)
        seg_classes = [] # just reads in the already computed masks
    elif mask_moveable_objects and seg_model_type == "predict":
        seg_model = SegFormer()
        seg_classes = ["car", "bus", "person", "truck", "bicycle", "motorcycle", "rider"]

    if isinstance(flow_steps, int):
        flow_steps = [flow_steps]

    # Compute pair-wise dense scale alignments for multiple flow steps
    log.info(f"Processing flow steps...")
    for flow_step in flow_steps:
        scales[flow_step] = {}
        frame_scales_for_hist[flow_step] = {} # for visualization
        scale_tensors_to_plot[flow_step] = {}
        tz_to_plot[flow_step] = {}
        std_to_plot[flow_step] = {}
        prev_scale = None # to catch outliers

        # TODO do this in a batched manner to speed up a lot (low prio)
        log.info(f"Processing each image pair for flow step {flow_step}. This might take some time...")
        for i, (target_id, target_image, target_pose) in enumerate(tqdm(dataset)):
            try:
                source_id, source_image, source_pose = dataset[i + flow_step]
            except IndexError:
                log.warning(f"The end of the dataset is reached while doing dense scale alignment")
                continue

            # Deal with combined datasets where frame ids have the video id as the first digit
            if isinstance(dataset, CombinedColmapDataset):
                target_vid_id = str(target_id)[0] # first digit is video id
                source_vid_id  = str(source_id)[0]

                # We don't want to compare frame pairs from different videos
                if target_vid_id != source_vid_id:
                    log.info(f"Reached the end of the current sub-trajectory while doing dense scale alignment, moving on to the next video")
                    continue

            target_pred = depth_model.predict({"frame_ids": [target_id]})
            source_pred = depth_model.predict({"frame_ids": [source_id]})
            target_depth = target_pred["depths"] # [N, 1, H, W]
            source_depth = source_pred["depths"]

            if mask_moveable_objects:
                seg_pred = seg_model.predict({
                    "images": torch.stack((target_image, source_image), dim=0),
                    "classes_to_segment": seg_classes,
                    "frame_ids": [target_id, source_id]
                    })
                seg_masks = combine_segmentation_masks(seg_pred["masks_dict"], seg_classes)
                target_mask = seg_masks[0].unsqueeze(0) # [1, 1, H, W]
                source_mask = seg_masks[1].unsqueeze(0)
            else:
                target_mask = torch.zeros_like(target_depth).bool()
                source_mask = torch.zeros_like(target_depth).bool()

            curr_scale, vis_tensor1, t_z = align_scale_for_image_pair(
                                                target_depth,
                                                source_depth,
                                                target_pose,
                                                source_pose,
                                                target_image,
                                                source_image,
                                                dataset.intrinsics,
                                                flow_model,
                                                target_mask,
                                                source_mask,
                                                min_d,
                                                max_d,
                                                min_flow,
                                                mask_occlusions,
                                                mask_disocclusions,
                                            )

            curr_scale_rev, vis_tensor2, t_z_rev = align_scale_for_image_pair(
                                                source_depth,
                                                target_depth,
                                                source_pose,
                                                target_pose,
                                                source_image,
                                                target_image,
                                                dataset.intrinsics,
                                                flow_model,
                                                source_mask,
                                                target_mask,
                                                min_d,
                                                max_d,
                                                min_flow,
                                                mask_occlusions,
                                                mask_disocclusions,
                                            )

            if i % log_every == 0:
                log.info(f"Computed forward scale {curr_scale.mean().item():0.4f} and backwards scale {curr_scale_rev.mean().item():0.4f} for images {target_id}-{source_id}, t_z: {t_z.item():0.4f}, scaled t_z: {curr_scale.mean().item() * t_z.item():0.4f}.")

            tz_to_plot[flow_step][target_id] = torch.abs(t_z).item()
            std_to_plot[flow_step][target_id] = (torch.std(curr_scale) + torch.std(curr_scale_rev)).mean().item()
            mean_scale = (curr_scale.mean() + curr_scale_rev.mean()).item() / 2

            if torch.abs(mean_scale * t_z) <= t_z_thresh:
                log.warning(f"Scaled up t_z value is less than the threshold ({torch.abs(mean_scale * t_z):0.4f} < {t_z_thresh}). Skipping images {target_id}-{source_id}")
                pass
            elif torch.abs(t_z) <= 0.001:
                log.warning(f"Computed t_z value is less than the threshold ({torch.abs(t_z):0.4f} < {0.2}). Skipping images {target_id}-{source_id}")
                pass
            else:
                scales[flow_step][target_id] = mean_scale

                # # If scale increased a lot we start tracking the frame id for detailed plots
                # if (prev_scale is not None) and abs(prev_scale - mean_scale) / prev_scale > 1.5:
                #     if target_id not in frame_ids_for_hist:
                #         log.warning(f"Scale changed too much compared to the last scale value ({prev_scale.item():0.4f} to {mean_scale.item():0.4f}). Tracking frame id {target_id}...")
                #         frame_ids_for_hist.append(target_id)

                # prev_scale = mean_scale

            # Save variables for histogram
            if target_id in frame_ids_for_hist:
                frame_scales_for_hist[flow_step][target_id] = {"forward": curr_scale, "backward": curr_scale_rev}
                scale_tensors_to_plot[flow_step][target_id] = {"forward": vis_tensor1, "backward": vis_tensor2}

            if debug and target_id > dataset.frame_ids[len(dataset) // 4]:
                log.warning(f"Debug mode is active, stopping dense alignment early at frame {target_id} for flow step {flow_step}")
                break

        # Save the mean of the scales to log
        mean_of_means = torch.tensor(list(scales[flow_step].values())).mean()
        std_of_means = torch.tensor(list(scales[flow_step].values())).std()
        log.info("===============")
        log.info(f"FINISHED FLOW STEP {flow_step}, MEAN OF CLEANED SCALES: {mean_of_means:0.4f}, STANDARD DEV: {std_of_means:0.4f}")
        log.info("===============")

    plot_dense_alignment_results(
        scales,
        frame_ids_for_hist,
        frame_scales_for_hist,
        scale_tensors_to_plot,
        tz_to_plot,
        std_to_plot,
        log_dir,
        dataset,
        )

    # Average over all flow steps, all frames, and both forward backward methods
    final_scale = compute_mean_recursive(scales)
    log.info(f"Final computed scale value to multiply the poses with: {final_scale:0.4f} (or multiply the depths with {1/final_scale:0.4f})")

    return 1 / final_scale


def align_scale_for_image_pair(
    target_depth: torch.Tensor,
    source_depth: torch.Tensor,
    target_pose: torch.Tensor,
    source_pose: torch.Tensor,
    target_image: torch.Tensor,
    source_image: torch.Tensor,
    intrinsics: Tuple[float],
    flow_model: BaseModel,
    target_mask: torch.Tensor = None,
    source_mask: torch.Tensor = None,
    min_depth: float = 0.0,
    max_depth: float = 100.0,
    min_flow: float = 0.2,
    mask_occlusions: bool = True,
    mask_disocclusions: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the scale of a depth model and the poses using a target and a source
    image using dense correspondences from optical flow. CURRENTLY DOES NOT SUPPORT
    BATCHED COMPUTATION.

    The target frame is referenced as coordinate frame A and the source frame as
    B. The frame B' is the auxilary coordinate frame that we use that has the
    same origin as frame B but the same rotation as frame A.
    """
    _, _, H, W = target_depth.shape # [N, 1, H, W] tensor
    device = target_depth.device
    K = format_intrinsics(intrinsics)
    K = K.to(device)

    # Flow prediction for correspondence computation and
    flow_pred = flow_model.predict({
        "image_a": target_image[None, :, :, :],
        "image_b": source_image[None, :, :, :]
        })
    flow = flow_pred["flow"] # [N, 2, H, W]

    if mask_occlusions or mask_disocclusions:
        reverse_flow_output = flow_model.predict({
            "image_a": source_image[None, :, :, :],
            "image_b": target_image[None, :, :, :],
            })
        reverse_flow = reverse_flow_output["flow"] # [N, 2, H, W]

        # Occluded (respectively disoccluded) parts are False in the masks, we want the opposite
        occlusion_mask, disocclusion_mask = compute_occlusions(flow, reverse_flow)

        if mask_occlusions:
            target_mask = target_mask | torch.logical_not(occlusion_mask)

        if mask_disocclusions:
            target_mask = target_mask | torch.logical_not(disocclusion_mask)

    # Define coordinate grids, pixel coords
    U, V = torch.meshgrid((torch.arange(W), torch.arange(H)), indexing="xy")
    Z = torch.ones_like(U)
    UVZ = torch.stack((U, V, Z), dim=0).unsqueeze(0).to(device)  # [1, 3, H, W] for backproj
    UV = torch.stack((U, V), dim=0).unsqueeze(0).to(device) # [1, 2, H, W] for grid sample

    # Compute t_z, z component of A_t_AB (A is the target, B is the source frame)
    T_WA = target_pose.clone()
    T_WB = source_pose.clone()
    T_AB = torch.inverse(T_WA) @ T_WB  # T_AW @ T_WB == T_AB
    t_z = T_AB[2, 3]  # z-component (in camera A's frame) of the transformation matrix

    # Compute transform and rotation from B to B'
    T_WBp = torch.eye(4, dtype=torch.double).to(device)
    T_WBp[:3, :3] = T_WA[:3, :3]  # B' has the same orientation as A
    T_WBp[:3, 3] = T_WB[:3, 3]  # B' has the same origin position as B

    T_BpB = torch.inverse(T_WBp) @ T_WB  # T_BpW @ T_WB == T_BpB
    R_BpB = T_BpB[:3, :3]

    # Backproject the points in frame B to compute depth_b_prime
    pointcloud = torch.matmul(torch.inverse(K), (source_depth * UVZ).reshape(1, 3, -1).double()) # [1, 3, num_points]
    pointcloud = torch.matmul(R_BpB, pointcloud)

    # We need only the Z coordinates of the points in B' for depth_b_prime
    depth_b_prime = pointcloud[:, 2, :]  # [1, num_points]
    depth_b_prime = depth_b_prime.reshape(1, 1, H, W)  # [1, 1, H, W]

    # Grid needs to be normalized to [-1, 1] and have shape [N, H, W, 2]
    grid = (UV + flow).permute(0, 2, 3, 1).double()

    # Solve for normalized coordinates: x_n: (x - 0) / (W - 0) == (x_n - (-1)) / (1 - (-1))
    grid[:, :, :, 0] = 2 * grid[:, :, :, 0] / (W - 1) - 1
    grid[:, :, :, 1] = 2 * grid[:, :, :, 1] / (H - 1) - 1

    # Given a correspondence in pixel coordinates from (i, j) in depth_a to (i', j') in depth_b_prime
    # depth_corr is a [1, 1, H , W] tensor where each position (1, 1, i, j) gives the
    # depth_b_prime value at (i', j') that corresponds to the (i, j) value in depth_a
    depth_corr = F.grid_sample(depth_b_prime, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

    # Get mask of pixels to ignore in scale computation
    if target_mask is not None:
        corr_mask = depth_corr <= 0 # mask of pixels that left the view (no corresponding depths)
        flow_mask = torch.linalg.norm(flow, dim=1, keepdim=True) <= min_flow
        depth_mask = (target_depth <= min_depth) | (target_depth >= max_depth)
        ignore_mask = target_mask | source_mask | corr_mask | flow_mask | depth_mask
        mask = torch.logical_not(ignore_mask)

    # Compute scale array for valid pixels for this target-source pair
    scales = torch.abs((target_depth[mask] - depth_corr[mask]) / t_z)
    # scales = 1 / scales # save the scale factor for the depths not the poses
    cleaned_scales, clean_mask = clean_scales(scales)

    # Visualizations for debugging
    scales_tensor = torch.zeros_like(target_depth).double()
    cleaned_scales_tensor = torch.zeros_like(target_depth).double()
    scales_tensor[mask] = scales
    cleaned_scales_tensor[mask] = scales * clean_mask

    return cleaned_scales, cleaned_scales_tensor, t_z


def clean_scales(
        scales: torch.Tensor,
        std_threshold: float = 1.5,
        max_cutoff: float = 500.0,
        min_cutoff=0.0
        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cleans the computed scales using heuristics to make the results more robust"""

    # Remove values below min and max, mask has the same shape as scales
    mask = (scales < max_cutoff) & (scales > min_cutoff)

    # Remove values that deviate by std_threshold * std away from the mean after thresholding
    mean = torch.mean(scales[mask])
    std = torch.std(scales[mask])

    mask = (
        mask
        & (scales < (mean + (std_threshold * std)))
        & (scales > (mean - (std_threshold * std)))
    )

    # mask has the same shape as scales, cleaned_scales should be a smaller tensor
    cleaned_scales = scales[mask]

    return cleaned_scales, mask