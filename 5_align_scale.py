# TODO add docstring
import json
import logging
import argparse
from pathlib import Path
from typing import NamedTuple
from datetime import datetime

import torch
import numpy as np
import open3d as o3d

from modules.segmentation.models import SegFormer
from modules.segmentation.utils import combine_segmentation_masks
from modules.core.models import RAFT
from modules.io.datasets import CustomDataset, COLMAPDataset
from modules.io.utils import read_all_poses_and_stamps, find_all_files_in_dir, create_scales_and_shifts_txt
from modules.scale_alignment.sparse import do_sparse_alignment
from modules.scale_alignment.dense import do_dense_alignment

# TODO implement this script!
def main(args):
    root_dir = args.root_dir
    pose_path = Path(args.pose_path)
    intrinsics = tuple(args.intrinsics)
    dataset_type = args.dataset.lower()
    target_size = tuple(args.target_size)
    sparse_cloud_path = args.sparse_cloud_path
    alignment_type = args.alignment_type.lower()
    mask_moveables = args.mask_moveables
    mask_occlusions = args.mask_occlusions
    flow_steps = args.flow_steps

    # TODO implement the reprojection error and the exhaustive scale computation thing?
    if alignment_type not in ["sparse", "dense", "dynamic", "reprojection"]:
        raise ValueError(f"Alignment type '{alignment_type}' not recognized. Please enter either 'sparse' or 'dense'.")

    colmap_dir = root_dir / Path("poses/colmap")
    depth_dir = root_dir / Path("data/depths/arrays") # as .npy arrays
    image_dir = root_dir / Path("data/rgb") # as png files
    if dataset_type == "colmap":
        log_dir = root_dir / Path("poses/colmap")
    else:
        log_dir = root_dir / Path("poses/slam")

    # Setup logging
    log_path = log_dir.absolute() / Path("log_scale_alignment.txt")
    log_time = datetime.now().strftime('%Y-%m-%d:%H-%M-%S')
    with open(log_path, 'w'): # to clear existing logs
        pass
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(levelname)s - %(message)s')
    logging.info(f"Log file for '5_align_scale.py', created at (year-month-day:hour-minute-second): {log_time}")
    # logging.info(f"Arguments: \n{json.dumps(vars(args), indent=4)}")
    logging.info(f"Arguments: \n{args}") # TODO remove after debug

    # TODO again, can use a dataset factory to clean up code
    # Read poses and precomputed depths
    if dataset_type == "colmap":
        dataset = COLMAPDataset(colmap_dir, pose_scale=1, target_size=target_size, orig_intrinsics=intrinsics, depth_dir=depth_dir, end=60)
    elif dataset_type == "custom":
        dataset = CustomDataset(image_dir, pose_path, pose_scale=1, target_size=target_size, orig_intrinsics=intrinsics, depth_dir=depth_dir, end=50)

    # Sparse alignment
    if alignment_type == "sparse":
        scales, shifts = do_sparse_alignment(
            dataset.poses,
            dataset.frame_ids,
            sparse_cloud_path,
            dataset.depth_paths,
            intrinsics,
            log_dir=log_dir,
            )

        create_scales_and_shifts_txt(dataset.depth_paths, scales, shifts, pose_path, dataset.frame_ids)

    # Dense alignment
    elif alignment_type == "dense":
        scales = do_dense_alignment(
            dataset,
            flow_steps,
            mask_moveable_objects=mask_moveables,
            mask_occlusions=mask_occlusions,
            log_dir=log_dir
            )

        # No shift calculation for dense alignment
        shifts = 0.0
        create_scales_and_shifts_txt(dataset.depth_paths, scales, shifts, pose_path, dataset.frame_ids)

    # Windowed dynamic dense alignment
    elif alignment_type == "dynamic":
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to align the scale predicted depths and poses. Computes scale and shift factors for depth values.")

    # class DebugArgsSparse(NamedTuple):
    #     root_dir: str = "/usr/stud/kaa/data/root/ds01"
    #     pose_path: str = "/usr/stud/kaa/data/root/ds01/poses/colmap/sparse/0_txt/images.txt" # path to poses
    #     sparse_cloud_path: str = "/usr/stud/kaa/data/root/ds01/poses/colmap/sparse/0_txt/points3D.ply"
    #     intrinsics: list = [1404.975, 1404.975, 1299.3, 769.45] # should make sure it matches the colmap/slam intrinsics for sparse alignment
    #     dataset: str = "colmap" # type of dataset
    #     alignment_type: str = "sparse" # either 'sparse', 'dense', or 'dynamic'
    #     mask_moveables: bool = True
    #     mask_occlusions: bool = True
    #     flow_step: int = 2 # step size for computing optical flow

    # args = DebugArgsSparse()

    class DebugArgsDense(NamedTuple):
        root_dir: str = "/usr/stud/kaa/data/root/ds01"
        pose_path: str = "/usr/stud/kaa/data/root/ds01/poses/colmap/sparse/0_txt/images.txt" # path to poses
        sparse_cloud_path: str = "/usr/stud/kaa/data/root/ds01/poses/colmap/sparse/0_txt/points3D.ply"
        # TODO these intrinsics do not match the ones we got for colmap, for colmap we were using different sized images, running colmap again
        intrinsics: list = [1404.975, 1404.975, 1299.3, 769.45] # should make sure it matches the colmap/slam intrinsics for sparse alignment
        dataset: str = "colmap" # type of dataset
        target_size: list = [576, 1024] # needs to be divisible by 8 for optical flow prediction
        alignment_type: str = "dense" # either 'sparse', 'dense', 'dynamic', 'reprojection'
        mask_moveables: bool = True
        mask_occlusions: bool = True
        flow_steps: list = [2, 4, 6] # step size for computing optical flow
        scales: list = [1, 2, 4] # scales to downsize images for dense scale alignment
    args = DebugArgsDense()

    main(args)


