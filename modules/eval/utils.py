from pathlib import Path
from typing import Dict, List, Union

import torch
import numpy as np
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
    for idx in range(poses.shape[0]):
        pose_dict[stamps[idx]] = poses[idx].detach().cpu().numpy()

    return pose_dict


def get_pose_matches(traj: torch.Tensor, ref_traj: torch.Tensor, stamps: Union[List[float], List[int]], ref_stamps: Union[List[float], List[int]], dataset: str):
    if dataset == "tum_rgbd":
        # We need to form {stamp: data} dictionaries to use associate() from the TUM dataset
        traj_dict = {stamps[i]: traj[:, i] for i in range(len(stamps))}
        ref_traj_dict = {ref_stamps[i]: ref_traj[:, i] for i in range(len(ref_stamps))}

        # Using the TUM RGB-D tools to get [3, num_matches] tensors
        matches = associate(traj_dict, ref_traj_dict, offset=0.0, max_difference=0.02)
        matched_traj = torch.stack([traj_dict[a] for a,b in matches], dim=1)
        matched_ref_traj = torch.stack([ref_traj_dict[b] for a,b in matches], dim=1)

    elif dataset in SUPPORTED_DATASETS:
        # Any other dataset should have exactly matching timestamps
        matched_traj = traj
        matched_ref_traj = []
        for ref_stamp in ref_stamps:
            try:
                idx = stamps.index(ref_stamp)
                matched_ref_traj.append(ref_traj[:, idx])
            except ValueError:
                continue

        if len(matched_ref_traj) == 0:
            raise RuntimeError(f"No matching timestamps have been found in the reference trajectory file. Please check your dataset and pose files!")

        matched_ref_traj = torch.stack(matched_ref_traj, dim=1) # [3, num_matches]

    return matched_traj, matched_ref_traj


def save_rpe_plot(trans_error: torch.Tensor, stamps: torch.Tensor, filename: Union[Path, str], output_dir: Union[Path, str]="."):
    """Draws and saves a relative positional error plot over time for pose evaluation"""
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(stamps - stamps[0],trans_error,'-',color="blue")
    ax.set_xlabel('time [s]')
    ax.set_ylabel('translational error [m]')
    plt.savefig(Path(output_dir) / Path(filename),dpi=300)