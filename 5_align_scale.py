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
    pose_path = Path(args.pose_path) if args.pose_path is not None else None
    intrinsics = tuple(args.intrinsics)
    dataset_type = args.dataset.lower()
    target_size = tuple(args.target_size)
    alignment_type = args.alignment_type.lower()
    sparse_cloud_path = Path(args.sparse_cloud_path) if args.sparse_cloud_path is not None else None
    mask_moveables = not args.skip_mask_moveables
    mask_occlusions = not args.skip_mask_occlusions
    flow_steps = args.flow_steps
    max_d = args.max_d

    # TODO implement dynamic scale alignment (windowed average) and the reprojection error exhaustive scale computation thing?
    if alignment_type not in ["sparse", "dense", "dynamic", "reprojection"]:
        raise ValueError(f"Alignment type '{alignment_type}' not recognized. Please enter either 'sparse' or 'dense'.")

    colmap_dir = root_dir / Path("poses/colmap")
    depth_dir = root_dir / Path("data/depths/arrays") # as .npy arrays
    image_dir = root_dir / Path("data/rgb") # as png files
    if dataset_type == "colmap":
        log_dir = root_dir / Path("poses/colmap")
    else:
        log_dir = root_dir / Path("poses/slam")

    depth_dir_empty = not any(depth_dir.iterdir())
    if depth_dir_empty:
        raise ValueError(f"The depth directory '{depth_dir}' is empty! Scale alignment requires precomputed depths. Please run '3_precompute_depth_and_normals.py' for your current dataset first.")

    # Setup logging
    log_path = log_dir.absolute() / Path("log_scale_alignment.txt")
    log_time = datetime.now().strftime('%Y-%m-%d:%H-%M-%S')
    with open(log_path, 'w'): # to clear existing logs
        pass
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(levelname)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(f"Log file for '5_align_scale.py', created at (year-month-day:hour-minute-second): {log_time}")
    logging.info(f"Arguments: \n{json.dumps(vars(args), indent=4)}")

    # TODO again, can use a dataset factory to clean up code
    # Read poses and precomputed depths
    if dataset_type == "colmap":
        dataset = COLMAPDataset(colmap_dir, pose_scale=1, target_size=target_size, orig_intrinsics=intrinsics, depth_dir=depth_dir)
        pose_path = dataset.pose_path
    elif dataset_type == "custom":
        dataset = CustomDataset(image_dir, pose_path, pose_scale=1, target_size=target_size, orig_intrinsics=intrinsics, depth_dir=depth_dir)

    # Sparse alignment
    if alignment_type == "sparse":
        scales, shifts = do_sparse_alignment(
            dataset.poses,
            dataset.frame_ids,
            sparse_cloud_path,
            dataset.depth_paths,
            intrinsics,
            log_dir=log_dir,
            max_d=max_d,
            )

        create_scales_and_shifts_txt(dataset.depth_paths, scales, shifts, pose_path, dataset.frame_ids)

    # Dense alignment
    elif alignment_type == "dense":
        scales = do_dense_alignment(
            dataset,
            flow_steps,
            mask_moveable_objects=mask_moveables,
            mask_occlusions=mask_occlusions,
            log_dir=log_dir,
            max_d=max_d,
            )

        # No shift calculation for dense alignment
        shifts = 0.0
        create_scales_and_shifts_txt(dataset.depth_paths, scales, shifts, pose_path, dataset.frame_ids)

    # Windowed dynamic dense alignment
    elif alignment_type == "dynamic":
        pass

    # Exhaustive scale search with reprojection error
    elif alignment_type == "reprojection":
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to align the scale predicted depths and poses. Computes scale and shift factors for depth values.")
    parser.add_argument("--root_dir", "-r", type=str, help="The root directory of your dataset")
    parser.add_argument("--pose_path", "-p", type=str, default=None, help="The path to your poses. Not needed if your dataset is a COLMAP dataset.")
    parser.add_argument("--intrinsics", type=float, nargs=4, help="Pinhole camera intrinsics. Provide as '--intrinsics fx fy cx cy'")
    parser.add_argument("--alignment_type", "-a", type=str, default="dense", help="The type of scale alignment. Use either 'sparse' or 'dense'.")
    parser.add_argument("--sparse_cloud_path", type=str, default=None, help="The path to the sparse point cloud. Only needed if you are using sparse alignment.")
    parser.add_argument("--target_size", type=int, nargs=2, default=(), help="The target (H, W) to resize the images to. Leave empty to use the original image sizes from extracted frames. Provide input as '--target_size H W'")
    parser.add_argument("--dataset", type=str, default="custom", help="Dataset type to use. Choose either 'colmap', 'custom' or 'kitti360'. If you choose 'kitti360' the ground truth poses and depths will be used. This is useful to validate the scale alignment methods.")
    parser.add_argument("--skip_mask_moveables", action="store_true", help="Flag to SKIP masking pixels belonging moveable objects. Not relevant for sparse alignment")
    parser.add_argument("--skip_mask_occlusions", action="store_true", help="Flag to SKIP masking pixels in occluded regions. Not relevant for sparse alignment.")
    parser.add_argument("--flow_steps", type=int, nargs="+", default=[2, 4], help="Step sizes for computing optical flow. Dense scale alignment is done for each of the integers you provide and the scale factors are averaged. Provide input as '--flow_step 1 2 3 ...'.")
    parser.add_argument("--max_d", type=float, default=50.0, help="Maximum depth to use for scale alignment")

    args = parser.parse_args()
    main(args)


