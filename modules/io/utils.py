"""Utility functions for data and input-output related operations"""
import logging
import shutil
import json
from pathlib import Path
from typing import Union, List, Callable, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt

from configs.data import SUPPORTED_DATASETS
from modules.io.datasets import CustomDataset, KITTI360Dataset, KITTIDataset, TUMRGBDDataset, COLMAPDataset

log = logging.getLogger(__name__)

def save_image_torch(tensor: torch.Tensor, name: str = "debug", output_dir: Union[str, Path] = "."):
    """
    Saves a torch tensor representing an image into disk, useful for debugging.
    Only saves the first sample in batched input
    """
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    if not output_dir.is_dir():
        logging.warning(f"Output directory for saving tensors '{str(output_dir)}' does not exist. Creating it...")
        output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / Path(f"{name}.png")

    if len(tensor.shape) == 4: # ignore rest of the batch
        tensor = tensor[0, :, :, :]
    tensor = tensor.squeeze()

    if len(tensor.shape) == 3: # rgb image
        plt.imsave(output_path, tensor.detach().cpu().permute(1, 2,0).numpy())
    if len(tensor.shape) == 2: # binary mask or 1-channel image
        plt.imsave(output_path, tensor.detach().cpu().numpy())


def find_latest_number_in_dir(dir_path: Union[str, Path]) -> int:
    """
    Finds and returns the largest number in a directory that contains
    enumerated subdirectories with the naming convention `xx_some_directory_name`
    where `xx` represents the numbering of the run. Useful for keeping track
    of reconstruction or gaussian splat training runs.
    """
    largest_idx = 0

    if isinstance(dir_path, str):
        dir_path = Path(dir_path)

    for path in dir_path.iterdir():
        path_str = str(path)

        # Don't count hidden files
        if path_str[0] == ".":
            continue

        # Expecting xx_some_directory_name as a naming convention where xx is a number
        idx = int(path_str.split("_")[0])
        if idx > largest_idx:
            largest_idx = idx

    return largest_idx


def ask_to_clear_dir(dir_path: Union[str, Path]) -> bool:
    """
    Asks the user to delete existing files in a directory if there are existing
    files. If the user types in 'y' the files are deleted.
    Useful for scripts that compute some values and save them for later usage.
    Returns a boolean representing whether the operation should be continued
    or not.
    """
    if isinstance(dir_path, str):
        dir_path = Path(dir_path)

    not_empty = any(dir_path.iterdir())
    should_continue = True

    if not_empty:
        while True:
            answer = input(f"The directory you specified {str(dir_path)} is not empty, do you want to delete existing files before continuing? [y/n]: ")

            if answer.lower() == "y":
                print(f"Deleting existing files at {str(dir_path)}...")
                for file in dir_path.iterdir():
                    if file.is_file():
                        file.unlink()
                should_continue = True
                break
            elif answer.lower() == "n":
                print(f"Not deleting existing files at {str(dir_path)} and aborting...")
                should_continue = False
                break

            print("Please type 'y' or 'n'")

    return should_continue


def find_all_files_in_dir(directory: Union[Path, str], extension: str = "") -> List:
    """
    Find all files in a directory (optionally with an extension `.png`, `.txt`
    etc.) and returns a list of the paths to each of the files.
    """
    if isinstance(directory, str):
        directory = Path(directory)

    # Extension needs to start with "."
    if extension != "" and extension[0] != ".":
        extension = "." + extension

    paths = []
    for f in directory.iterdir():
        if f.is_file():
            if extension == "":
                paths.append(f.absolute())
            if extension != "" and extension == f.suffix:
                paths.append(f.absolute())

    return paths

def get_parse_and_stamp_fn(dataset: str) -> Tuple[Callable, Callable]:
    """
    Returns the dataset-specific functions for parsing the 4x4 T_WC (cam to
    world) pose matrix and the frame id (or the timestamp) from the cleaned
    rows of the pose file.
    """
    if dataset == "custom":
        dataset_class = CustomDataset
    elif dataset == "kitti360":
        dataset_class = KITTI360Dataset
    elif dataset == "kitti":
        dataset_class = KITTIDataset
    elif dataset == "tum_rgbd":
        dataset_class = TUMRGBDDataset
    elif dataset  == "colmap":
        raise NotImplementedError("Will do this soon!")
    elif dataset not in SUPPORTED_DATASETS:
        raise NotImplementedError(f"The dataset you specified ({dataset}) is not supported yet, supported datasets are: {str(SUPPORTED_DATASETS)}")
    else:
        raise NotImplementedError("Please add the dataset you implemented to the if statement here")

    parse_fn = dataset_class.parse_pose
    stamp_fn = dataset_class.parse_frame_id

    return parse_fn, stamp_fn


def read_all_poses_and_stamps(pose_path: Union[Path, str], dataset: str = "custom") -> Tuple[torch.Tensor, List[int]]:
    """
    Reads all poses from the rows of `pose_path` depending on the dataset.
    Returns a `torch.Tensor` with shape `[N, 4, 4]` where `N` is the number of
    camera centers specified in the pose file and a list of all timestamps (or
    frame ids) with the associated poses.
    """
    if isinstance(pose_path, str):
        pose_path = Path(pose_path)

    parse_fn, stamp_fn = get_parse_and_stamp_fn(dataset)

    poses = []
    stamps = []
    with open(pose_path, "r") as p_file:
        all_pose_rows = p_file.readlines()

    for pose_row in all_pose_rows:
        pose_row = pose_row.strip()
        if pose_row[0] == "#":
            continue
        cols = pose_row.split(" ")

        pose = parse_fn(cols)
        stamp = stamp_fn(cols)
        poses.append(pose)
        stamps.append(stamp)

    poses = torch.stack(poses, dim=0)

    # If the stamps are None (no frame id in pose file), we simply enumerate them
    if stamps[0] is None:
        stamps = [i for i in range(len(poses))]

    return poses, stamps


def read_poses_from_stamps(pose_path: Union[Path, str], matching_stamps: List[int], dataset: str = "custom") -> torch.Tensor:
    """
    Reads poses from `pose_path` if and only if the timestamps of the poses
    match the ones in `matching_stamps`. This is relevant for the cases
    when the pose estimation system does not give you poses for every ground
    truth pose entry you have.
    """
    if isinstance(pose_path, str):
        pose_path = Path(pose_path)

    parse_fn, stamp_fn = get_parse_and_stamp_fn(dataset)

    poses = []
    stamp_count = 0 # counter for datasets that don't have frame ids in pose files
    with open(pose_path, "r") as p_file:
        all_pose_rows = p_file.readlines()

    for pose_row in all_pose_rows:
        pose_row = pose_row.strip()

        # Skip comments
        if pose_row[0] == "#":
            continue

        cols = pose_row.split(" ")
        pose = parse_fn(cols)
        stamp = stamp_fn(cols)

        if stamp is not None and stamp not in matching_stamps:
            continue
        poses.append(pose)

    poses = torch.stack(poses, dim=0)
    return poses


def read_deepscenario_sparse_cloud(filename: Union[str, Path]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reads in the ground truth reconstructions for the custom data from DeepScenario.
    Returns the `xyz` and `rgb` values as torch tensors with shape
    `[3, num_points]` to then be converted to any point cloud format.

    The input is expected to be a path to a file formatted as a json file
    as:
    ```
    [
      {
        "shots": {...},
        "cameras": {...},
        "points":{
            "point_id": {
                "color": [R, G, B],
                "coordinates": [x, y, z],
            },
            ...
        },
        "biases": {...},
        "rig_cameras": {...},
        "rig_instances": {...},
        "reference_lla": {...},
      }
    ]
    ```
    """
    if isinstance(filename, str):
        filename = Path(filename)

    with open(filename, "rb") as file:
        data = json.load(file) # might take some time, if the point cloud is too large use ijson

    points = data[0]["points"] # the whole json is saved as a list so we need to access the first element
    xyz = []
    rgb = []
    for point_id in points.keys():
        coords = points[point_id]["coordinates"] # both are lists with three elements
        colors = points[point_id]["color"]
        xyz.append(coords)
        rgb.append(colors)
    xyz = torch.tensor(xyz, dtype=torch.double).T # [3, num_points]
    rgb = torch.tensor(rgb, dtype=torch.uint8).T

    return xyz, rgb


