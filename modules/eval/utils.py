import copy
from pathlib import Path
from typing import Dict, List, Union, Tuple

import torch
import numpy as np
import open3d as o3d
from tqdm import tqdm
import matplotlib.pyplot as plt

from configs.data import SUPPORTED_DATASETS
from modules.eval.tum_rgbd_tools.associate import associate


def round_results(results: Dict, decimal_point: int) -> Dict:
    "Parses all elements of a dictionary and rounds them up to a given decimal point"
    for key, value in results.items():
        if isinstance(value, torch.Tensor) or isinstance(value, np.ndarray) or isinstance(value, str):
            results[key] = value
        elif isinstance(value, dict):
            round_results(value, decimal_point)
        else:
            results[key] = round(value, decimal_point)
    return results


def compute_mean_recursive(results: Dict) -> float:
    "Parses all elements of a numeric dictionary and recursively computes the mean"
    def compute_mean_helper(data: Dict, accumulator: List):
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                accumulator.append(value.mean().item())
            elif isinstance(value, dict):
                compute_mean_helper(value, accumulator)
            elif isinstance(value, (int, float)):
                accumulator.append(value) # ignores non-numeric types

    accumulator = []
    compute_mean_helper(results, accumulator)
    overall_mean = (sum(accumulator) / len(accumulator)) if accumulator else 0
    return overall_mean



def extract_translations_from_poses(poses: torch.Tensor) -> torch.Tensor:
    """
    Expects `[N, 4, 4]` poses for `N` points on the trajectories and
    returns the translation components (first three rows of the last column)
    as a `[3, N]` tensor.
    """
    traj = poses[:, :3, 3]
    traj = traj.T # [3, N]
    return traj


def get_dummy_stamps(traj: torch.Tensor) -> torch.Tensor:
    """
    Used to get dummy values for time stamps based on the `traj` tensor
    with shape `[3, N]` where `N` is the number of frames we have poses for.
    Simply enumerates the frames and returns a `[N]` tensor with the
    corresponding numbers.
    """
    _, N = traj.shape
    stamps = torch.arange(N)
    return stamps


def build_pose_dict(poses: torch.Tensor, stamps: List[int]) -> Dict:
    """
    Build the trajectory data structure TUM RGB-D dataset tools scripts expect
    to see, i.e. as `{"stamp": data, ...}` dictionaries. The poses are expected
    to be a `[N, 4, 4]` tensor. The returned poses are numpy arrays. Useful
    for relative pose error computation.
    """

    if len(stamps) != poses.shape[0]:
        raise RuntimeError(f"The number of your stamps ({len(stamps)}) does not match the number of trajectories you have ({poses.shape[1]}). Please check your poses!")

    pose_dict = {}
    for idx in range(poses.shape[0]): # iterate over all poses saved in first dimension
        pose_dict[stamps[idx]] = poses[idx].detach().cpu().numpy()

    return pose_dict


def get_pose_matches(poses: torch.Tensor, ref_poses: torch.Tensor, stamps: Union[List[float], List[int]], ref_stamps: Union[List[float], List[int]], dataset: str) -> Tuple[torch.Tensor, torch.Tensor]:
    # Extract translations from poses first
    traj = extract_translations_from_poses(poses) # [3, N]
    ref_traj = extract_translations_from_poses(ref_poses)

    # We need to form {stamp: data} dictionaries to use associate() from TUM RGB-D Tools
    if dataset == "tum_rgbd":
        traj_dict = {stamps[i]: traj[:, i] for i in range(len(stamps))}
        ref_traj_dict = {ref_stamps[i]: ref_traj[:, i] for i in range(len(ref_stamps))}

        matches = associate(traj_dict, ref_traj_dict, offset=0.0, max_difference=0.02)
        matched_traj = torch.stack([traj_dict[a] for a,b in matches], dim=1)
        matched_ref_traj = torch.stack([ref_traj_dict[b] for a,b in matches], dim=1)

    # Any other dataset should have exactly matching timestamps
    elif dataset in SUPPORTED_DATASETS:
        matched_traj = []
        matched_ref_traj = []
        for ref_idx, ref_stamp in enumerate(ref_stamps):
            try:
                idx = stamps.index(ref_stamp) # will throw value error if it can't find it
                matched_traj.append(traj[:, idx])
                matched_ref_traj.append(ref_traj[:, ref_idx])
            except ValueError:
                continue

        if len(matched_ref_traj) == 0 or len(matched_traj) == 0:
            raise RuntimeError(f"No matching timestamps have been found in the reference trajectory file. Please check your dataset and pose files!")

        matched_traj = torch.stack(matched_traj, dim=1) # [3, num_matches]
        matched_ref_traj = torch.stack(matched_ref_traj, dim=1)

    return matched_traj, matched_ref_traj



def save_pointclouds_src_target(source: o3d.geometry.PointCloud,
                                target: o3d.geometry.PointCloud,
                                downsample_voxel_size: float = -1.0,
                                T: Union[np.ndarray, torch.Tensor] = torch.eye(4),
                                filename: Union[Path, str] = "cloud.ply",
                                output_dir: Union[Path, str] = "./",
                                ):
    if isinstance(T, torch.Tensor):
        T = T.detach().cpu().numpy()
    if isinstance(filename, str):
        filename = Path(filename)
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    # Color the clouds to better see the alignment
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])

    source_temp.transform(T)

    # Downsample to save memory
    if downsample_voxel_size > 0.0:
        source_temp = source_temp.voxel_down_sample(voxel_size=downsample_voxel_size)
        target_temp = target_temp.voxel_down_sample(voxel_size=downsample_voxel_size)

    source_path = str(output_dir / filename.with_name(filename.stem + "_src" + filename.suffix))
    target_path = str(output_dir / filename.with_name(filename.stem + "_target" + filename.suffix))
    o3d.io.write_point_cloud(source_path, source_temp)
    o3d.io.write_point_cloud(target_path, target_temp)


def do_multiscale_colored_icp(source: o3d.geometry.PointCloud,
                              target: o3d.geometry.PointCloud,
                              voxel_radius: List[float] = [0.04, 0.02, 0.01],
                              max_iter: List[int] = [50, 30, 14]
                              ) -> np.ndarray:
    """
    Does colored point cloud ICP registration based on the method of
    J. Park, Q.-Y. Zhou, V. Koltun -- Colored Point Cloud Registration Revisited
    ICCV 2017. Returns the computed transform as a numpy array.

    Code inspired heavily from the Open3D tutorial on colored point cloud
    registration.
    """
    current_transformation = np.eye(4)
    for scale in tqdm(range(len(voxel_radius))):
        iter = max_iter[scale]
        radius = voxel_radius[scale]

        # Downsample point cloud for current scale
        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)

        # Estimate normals, necessary for point-to-plane ICP
        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

        # Do the actual registration
        result_icp = o3d.pipelines.registration.registration_colored_icp(
            source_down, target_down, radius, current_transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                                relative_rmse=1e-6,
                                                                max_iteration=iter))
        current_transformation = result_icp.transformation

    return np.array(current_transformation)


def do_point_to_plane_icp(source: o3d.geometry.PointCloud,
                          target: o3d.geometry.PointCloud,
                          threshold: float = 0.02,
                          ) ->  np.ndarray:
    # Initial guess of the transform source to target, assuming we did alignment with poses
    trans_init = np.eye(4)

    # TODO the threshold we use might be off if we have COLMAP clouds, reference cloud must be in metric scale
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
    current_transformation = reg_p2l.transformation

    return np.array(current_transformation)