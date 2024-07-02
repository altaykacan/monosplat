# TODO add docstring
"""Script to apply either sparse or dense scale alignment to monocular SLAM poses or COLMAP poses. Aligns the scale of up-to-scale poses with the internal scale of a """
import json
import logging
import argparse
from pathlib import Path
from typing import NamedTuple
from datetime import datetime

import torch
import numpy as np
import open3d as o3d

from configs.data import PADDED_IMG_NAME_LENGTH
from modules.segmentation.models import SegFormer
from modules.segmentation.utils import combine_segmentation_masks
from modules.core.models import RAFT
from modules.io.datasets import CustomDataset, ColmapDataset, CombinedColmapDataset, KITTI360Dataset
from modules.io.utils import read_all_poses_and_stamps, find_all_files_in_dir, create_scales_and_shifts_txt, ask_to_clear_dir
from modules.scale_alignment.sparse import do_sparse_alignment
from modules.scale_alignment.dense import do_dense_alignment

def main(args):
    root_dir = Path(args.root_dir)
    pose_path = Path(args.pose_path) if args.pose_path is not None else None
    intrinsics = tuple(args.intrinsics)
    dataset_type = args.dataset.lower()
    target_size = tuple(args.target_size)
    alignment_type = args.alignment_type.lower()
    sparse_cloud_path = Path(args.sparse_cloud_path) if args.sparse_cloud_path is not None else None
    mask_moveables = not args.skip_mask_moveables
    seg_model_type = args.seg_model_type
    mask_occlusions = not args.skip_mask_occlusions
    flow_steps = args.flow_steps
    max_d = args.max_d
    cam_id = args.kitti360_cam_id
    seq_id = args.kitti360_seq_id
    start_id = args.start_id
    end_id = args.end_id
    depth_type = args.depth_type
    debug = args.debug
    exp_name = args.exp_name
    padded_img_name_length = args.padded_img_name_length

    if not root_dir.exists():
        raise RuntimeError(f"Your root_dir at '{str(root_dir)}' does not exist! Please make sure you give the right path.")

    # TODO implement dynamic scale alignment (windowed average) and the reprojection error exhaustive scale computation thing?
    if alignment_type not in ["sparse", "dense", "dynamic", "reprojection"]:
        raise ValueError(f"Alignment type '{alignment_type}' not recognized. Please enter either 'sparse' or 'dense'.")

    colmap_dir = root_dir / Path("poses/colmap")
    depth_dir = root_dir / Path("data/depths/arrays") # as .npy arrays
    image_dir = root_dir / Path("data/rgb") # as png files

    if mask_moveables:
        mask_dir = root_dir / "data" / "masks_moveable"
        if not mask_dir.exists():
            raise ValueError(f"You specified '{str(mask_dir)}' as your mask directory but it does not exist. Please check your data!")
        if not any(mask_dir.iterdir()):
            raise ValueError(f"You specified '{str(mask_dir)}' as your mask directory but it is empty. Please check your data!")
    else:
        mask_dir = None

    if "colmap" in dataset_type:
        log_dir = root_dir / Path("poses/colmap")
    elif "custom" in dataset_type:
        log_dir = root_dir / Path("poses/slam")
    else:
        log_dir = root_dir/ Path("eval/scale_validation")

    if exp_name is not None:
        log_dir = log_dir / Path(exp_name)

    log_dir = log_dir / Path("alignment_plots")
    log_dir.mkdir(exist_ok=True, parents=True )

    depth_dir_empty = not any(depth_dir.iterdir())
    if depth_dir_empty:
        raise ValueError(f"The depth directory '{depth_dir}' is empty! Scale alignment requires precomputed depths. Please run '3_precompute_depth_and_normals.py' for your current dataset first.")

    # Setup logging
    log_path = log_dir.parent / Path("log_scale_alignment.txt")
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
        dataset = ColmapDataset(colmap_dir, pose_scale=1, target_size=target_size, orig_intrinsics=intrinsics, depth_dir=depth_dir, start=start_id, end=end_id, mask_dir=mask_dir)
        pose_path = dataset.pose_path
    if dataset_type == "combined_colmap":
        dataset = CombinedColmapDataset(colmap_dir, pose_scale=1, target_size=target_size, orig_intrinsics=intrinsics, depth_dir=depth_dir, start=start_id, end=end_id, mask_dir=mask_dir)
        pose_path = dataset.pose_path
    elif dataset_type == "custom":
        dataset = CustomDataset(image_dir, pose_path, pose_scale=1, target_size=target_size, orig_intrinsics=intrinsics, depth_dir=depth_dir, start=start_id, end=end_id, mask_dir=mask_dir, padded_img_name_length=padded_img_name_length)
    elif dataset_type == "kitti360":
        dataset = KITTI360Dataset(seq_id, cam_id, pose_scale=1, target_size=target_size, depth_dir=depth_dir, start=start_id, end=end_id, mask_dir=mask_dir)
        pose_path = log_dir

    # Sparse alignment
    if alignment_type == "sparse":
        should_continue = ask_to_clear_dir(log_dir / Path("tensors"))
        if not should_continue:
            logging.warning(f"The directory '{str(log_dir / Path('tensors'))}' is not empty. Aborting.")
            return -1

        should_continue = ask_to_clear_dir(log_dir)
        if not should_continue:
            logging.warning(f"The directory '{str(log_dir)}' is not empty. Aborting.")
            return -1

        if isinstance(dataset, KITTI360Dataset) and depth_type == "gt":
            depth_paths = dataset.gt_depth_paths
        elif depth_type == "precomputed":
            depth_paths = dataset.depth_paths

        # By default we only compute only scales, shift factors are all 0
        scales, shifts = do_sparse_alignment(
            dataset.poses,
            dataset.frame_ids,
            sparse_cloud_path,
            depth_paths,
            dataset.intrinsics,
            log_dir=log_dir,
            max_d=max_d,
            )

        create_scales_and_shifts_txt(depth_paths, scales, shifts, pose_path, dataset.frame_ids)

    # Dense alignment
    elif alignment_type == "dense":
        should_continue = ask_to_clear_dir(log_dir / Path("tensors"))
        if not should_continue:
            logging.warning(f"The directory '{str(log_dir / Path('tensors'))}' is not empty and you chose to not delete it. Aborting.")
            return -1

        should_continue = ask_to_clear_dir(log_dir)
        if not should_continue:
            logging.warning(f"The directory '{str(log_dir)}' is not empty and you chose to not delete it. Aborting.")
            return -1

        scales = do_dense_alignment(
            dataset,
            flow_steps,
            mask_moveable_objects=mask_moveables,
            seg_model_type=seg_model_type,
            mask_occlusions=mask_occlusions,
            log_dir=log_dir,
            max_d=max_d,
            debug=debug,
            depth_type=depth_type,
            )

        # No shift calculation for dense alignment
        shifts = 0.0
        if isinstance(dataset, KITTI360Dataset) and depth_type == "gt":
            depth_paths = dataset.gt_depth_paths
        elif isinstance(dataset, KITTI360Dataset) and depth_type == "precomputed":
            depth_paths = dataset.depth_paths
        else:
            depth_paths = dataset.depth_paths
        create_scales_and_shifts_txt(depth_paths, scales, shifts, pose_path, dataset.frame_ids)

    # Windowed dynamic dense alignment
    elif alignment_type == "dynamic":
        pass

    # Exhaustive scale search with reprojection error
    elif alignment_type == "reprojection":
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to align the scale predicted depths and poses. Computes scale and shift factors for depth values. Works only with precomputed depths and poses!")
    parser.add_argument("--root_dir", "-r", type=str, help="The root directory of your dataset")
    parser.add_argument("--pose_path", "-p", type=str, default=None, help="The path to your poses. Not needed if your dataset is a COLMAP dataset (i.e. if you give '--dataset colmap' or '--dataset combined_colmap').")
    parser.add_argument("--intrinsics", type=float, nargs=4, help="Pinhole camera intrinsics. Provide as '--intrinsics fx fy cx cy'")
    parser.add_argument("--alignment_type", "-a", type=str, default="dense", choices=["dense", "sparse"], help="The type of scale alignment. Use either 'sparse' or 'dense'.")
    parser.add_argument("--sparse_cloud_path", type=str, default=None, help="The path to the sparse point cloud. Only needed if you are using sparse alignment with '--alignment_type sparse'.")
    parser.add_argument("--target_size", type=int, nargs=2, default=(), help="The target (H, W) to resize the images to. Leave empty to use the original image sizes from extracted frames. Provide input as '--target_size H W'")
    parser.add_argument("--dataset", type=str, default="custom", help="Dataset type to use. Choose either 'colmap', 'combined_colmap', 'custom' or 'kitti360'. If you choose 'kitti360' the ground truth poses and depths will be used and you need to provide '--cam_id' and '--seq_id' options. This is useful to validate the scale alignment methods.")
    parser.add_argument("--skip_mask_moveables", action="store_true", help="Flag to SKIP masking pixels belonging moveable objects. Not relevant for sparse alignment")
    parser.add_argument("--seg_model_type", type=str, default="predict", choices=["predict", "precomputed"], help="The type of segmentation model you want to use for dense scale alignment. Give either 'predict' or 'precomputed'. Only relevant for '--alignment_type dense'")
    parser.add_argument("--skip_mask_occlusions", action="store_true", help="Flag to SKIP masking pixels in occluded regions. Not relevant for sparse alignment.")
    parser.add_argument("--flow_steps", type=int, nargs="+", default=[2, 4, 6], help="Step sizes for computing optical flow. Dense scale alignment is done for each of the integers you provide and the scale factors are averaged. Provide input as '--flow_step 1 2 3 ...'.")
    parser.add_argument("--max_d", type=float, default=30.0, help="Maximum depth to use for scale alignment")
    parser.add_argument("--start_id", type=int, default=0, help="The first id to take from the dataset of your choice. Useful for validating with mini datasets.")
    parser.add_argument("--end_id", type=int, default=-1, help="The last id to take from the dataset of your choice. Useful for validating with mini datasets.")
    parser.add_argument("--kitti360_cam_id", type=int, default=None, help="Camera id for KITTI360. Only relevant if you provide '--dataset kitti360'")
    parser.add_argument("--kitti360_seq_id", type=int, default=None, help="Sequence id for KITTI360. Only relevant if you provide '--dataset kitti360'")
    parser.add_argument("--depth_type", type=str, default="precomputed", choices=["gt", "precomputed"], help="Whether to use ground truth depths or precomputed depths to do scale alignment with your dataset. Only KITTI360 datasets have ground truth depths so far. Useful for validating sparse and dense alignment give the same results (using dense depth preds with ground truth poses)")
    parser.add_argument("--debug", action="store_true", help="Debug flag, uses a subset of the images to avoid long waiting times when debugging.")
    parser.add_argument("--exp_name", type=str, default=None, help="Optional experiment name to save results into a different directory. Useful for evaluation and validation.")
    parser.add_argument("--padded_img_name_length", type=int, default=PADDED_IMG_NAME_LENGTH, help="Total image name length. The integer frame id will be prepended with zeros until it reaches this length. For colmap datasets use 5, for KITTI use 6, for KITTI360 use 10")

    args = parser.parse_args()
    main(args)