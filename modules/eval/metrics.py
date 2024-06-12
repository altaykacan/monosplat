"""Contains helper functions and classes to compute standard metrics"""
from typing import Dict, Union, Callable, Tuple, List

import torch
import open3d as o3d

from modules.eval.utils import build_pose_dict
from modules.eval.tum_rgbd_tools.evaluate_rpe import evaluate_trajectory_rpe

class AverageMetric:
    """Parent class for metrics"""
    def __init__(self, metric_fn: Callable):
        self.metric_fn = metric_fn
        self.reset()

    def reset(self):
        self._val = 0.0
        self._sum = 0.0
        self._count = 0.0
        self.avg = 0.0

class AverageDepthMetric(AverageMetric):
    """
    Class to keep track of averages of a given metric that is computed with
    a function `metric_fn` that expects batched predicted and ground truth
    depths as input and returns a float value.

    Implementation inspired from Metric3Dv2 authors: https://github.com/YvanYin/Metric3D/blob/main/mono/utils/avg_meter.py
    """

    def update(self, depth: torch.Tensor, gt_depth: torch.Tensor) -> torch.Tensor:
        if len(depth.shape) != 4:
            raise ValueError(f"Expected '[N, C, H, W]' batched input for predictions but got {depth.shape} shape")
        if len(gt_depth.shape) != 4:
            raise ValueError(f"Expected '[N, C, H, W]' batched input for labels but got {depth.shape} shape")

        self._val, err = self.metric_fn(depth, gt_depth) # err is to visualize errors
        self._sum = self._sum + self._val
        self._count += 1 # counts batches, not samples
        self.avg = self._sum / self._count # averages over batches
        return err


def compute_absrel(depth: torch.Tensor, gt_depth: torch.Tensor) -> Tuple[float, torch.Tensor]:
    err = torch.zeros_like(depth)
    mask = gt_depth > 0 # also avoids division by zero
    diff = torch.abs(depth[mask] - gt_depth[mask]) / gt_depth[mask]
    err[mask]= diff
    result = torch.mean(diff).item() # mean over all pixels in current batch
    return result, err


def compute_sqrel(depth: torch.Tensor, gt_depth: torch.Tensor) -> Tuple[float, torch.Tensor]:
    err = torch.zeros_like(depth)
    mask = gt_depth > 0
    diff = ((depth[mask] - gt_depth[mask]) ** 2) /  gt_depth[mask]
    err[mask] = diff
    result = torch.mean(diff).item()
    return result, err


# TODO do we take the sqrt of the total mean or do we take the mean of the sqrt values individually?
# i.e. is the sum over all pixels for a given image or is it ALL valid pixels in the WHOLE dataset
def compute_rmse(depth: torch.Tensor, gt_depth: torch.Tensor) -> Tuple[float, torch.Tensor]:
    N, C, H, W = depth.shape
    err = torch.zeros_like(depth) # [N, 1, H, W], useful for visualization
    diff = 0.0

    # Due to the square root, taking the mean over batched preds in one go doesn't work
    for batch_idx in range(N):
        depth_i = depth[batch_idx, :, :, :] # [1, H, W]
        gt_depth_i = gt_depth[batch_idx, : ,:, :] # [1, H, W]
        mask = gt_depth_i > 0 # per-batch sample pixel masking
        err[batch_idx][mask] = torch.sqrt((depth_i[mask] - gt_depth_i[mask])**2) # tensor
        diff += torch.sqrt(torch.mean((depth_i[mask] - gt_depth_i[mask])**2)).item() # scalar

    result = diff / N # take the mean over the current batch
    return result, err


def compute_rmse_log(depth: torch.Tensor, gt_depth: torch.Tensor) -> Tuple[float, torch.Tensor]:
    # Use log base 10 like Metric3D and add small number for numerical stability
    return compute_rmse(torch.log10(depth + 1e-10), torch.log10(gt_depth + 1e-10))


def accuracy_threshold(depth: torch.Tensor, gt_depth: torch.Tensor, i: int) -> Tuple[float, torch.Tensor]:
    mask = gt_depth > 0
    err = torch.zeros_like(depth) # [N, 1, H, W]
    depth_m = depth[mask]
    gt_depth_m = gt_depth[mask]

    diff = torch.maximum(depth_m / gt_depth_m, gt_depth_m / depth_m)
    err[mask] = diff
    result = torch.mean((diff < 1.25**i).float()).item() # mean over all the pixels in batch
    return result, err


def compute_accuracy_threshold_1(depth: torch.Tensor, gt_depth: torch.Tensor) -> Tuple[float, torch.Tensor]:
    return accuracy_threshold(depth, gt_depth, i=1)


def compute_accuracy_threshold_2(depth: torch.Tensor, gt_depth: torch.Tensor) -> Tuple[float, torch.Tensor]:
    return accuracy_threshold(depth, gt_depth, i=2)


def compute_accuracy_threshold_3(depth: torch.Tensor, gt_depth: torch.Tensor) -> Tuple[float, torch.Tensor]:
    return accuracy_threshold(depth, gt_depth, i=3)


def compute_ate(pred_traj: torch.Tensor, ref_traj: torch.Tensor, rot: torch.Tensor, trans: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Computes the absolute trajectory error. Trajectories should be torch tensors
    with shape `[3, N]`. Returns the computed translational errors per
    trajectory sample point with shape `[1, N]`
    """
    aligned_traj = scale * rot @ pred_traj + trans
    alignment_error = aligned_traj - ref_traj
    trans_error = torch.sqrt(torch.sum(alignment_error * alignment_error, dim=0))

    return trans_error


def compute_rpe(pred_poses: torch.Tensor, ref_poses: torch.Tensor, stamps: Union[List[float], List[int]], ref_stamps: Union[List[float], List[int]], scale: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the relative pose error. Poses should be torch tensors with
    shape `[N, 4, 4]`
    """
    # Convert pose tensors into the expected dictionary format as numpy arrays
    pred_pose_dict = build_pose_dict(pred_poses, stamps)
    ref_pose_dict = build_pose_dict(ref_poses, ref_stamps)

    result = evaluate_trajectory_rpe(ref_pose_dict,
                                     pred_pose_dict,
                                     param_max_pairs=10000,
                                     param_fixed_delta=True,
                                     param_delta_unit="f",
                                     param_offset=0,
                                     param_scale=scale
    )

    stamps_rpe = torch.tensor(result, dtype=torch.double)[:, 0]
    trans_error = torch.tensor(result, dtype=torch.double)[:, 4]
    rot_error = torch.tensor(result, dtype=torch.double)[:,5]
    return stamps_rpe, trans_error, rot_error


def compute_accuracy_pcd(pred_to_ref_dists: torch.Tensor) -> float:
    """
    Computes the accuracy metric for point cloud reconstruction evaluation using
    open3D.
    """
    dists = pred_to_ref_dists
    acc = torch.mean(dists).item() # mean over all the pred points
    return acc


def compute_completion_pcd(ref_to_pred_dists: torch.Tensor) -> float:
    """
    Computes the completion metric for point cloud reconstruction evaluation
    using open3D similar to `compute_accuracy_pcd()`.
    """
    # TODO do we need to mask the reference cloud (for missing ground truth?)
    dists = ref_to_pred_dists
    comp = torch.mean(dists).item() # mean over all the ground truth points
    return comp


def compute_chamfer_distance_pcd(accuracy: float, completion: float) -> float:
    """
    Computes chamfer distance which is the mean of the accuracy and completion
    metrics for point cloud evaluation.
    """
    return (accuracy + completion) / 2


def compute_precision_pcd(pred_to_ref_dists: torch.Tensor, thresh: float = 0.05) -> float:
    """
    Computes the precision metric for point cloud reconstruction evaluation
    using open3D. It computes the mean distance from points in `pred_pcd` to
    the closest point in `ref_pcd` for all point pairs that have a distance
    less than `thresh`. In the literature this value is set to be 5cm.
    """
    dists = pred_to_ref_dists
    valid_dists = dists < thresh
    prec = len(valid_dists) / len(dists)
    return prec


def compute_recall_pcd(ref_to_pred_dists: torch.Tensor, thresh: float = 0.05) -> float:
    """
    Computes the recall metric for point cloud reconstruction evaluation using
    open3D. Similar to `compute_precision_pcd()` but the distance computation
    is the other way around.
    """
    dists = ref_to_pred_dists
    valid_dists = dists < thresh
    recall = len(valid_dists) / len(dists)
    return recall


def compute_fscore_pcd(precision: float, recall: float) -> float:
    """
    Computes the f-score which is the harmonic mean of the precision and recall
    metrics.
    """
    return (2 * precision * recall) / (precision + recall)


