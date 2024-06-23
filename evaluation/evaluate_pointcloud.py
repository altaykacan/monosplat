"""
A script to compute standard metrics for an estimated pointcloud and a
reference ground truth point cloud (or mesh). Uses poses to align the
coordinate frames of the two clouds. An alternative is to use some global
registration algorithm to get a rough alignment between the point clouds.
Using poses to align the coordinate frames is much more scalable for
large clouds (it is independent of cloud size) given that we have poses
for both clouds.

The registration results are evaluated with (from open3D ICP tutorial):
- fitness, which measures the overlapping area (# of inlier correspondences / # of points in target). The higher the better.
- inlier_rmse, which measures the RMSE of all inlier correspondences. The lower the better.
"""
import json
import logging
import datetime
import argparse
from pathlib import Path
from typing import NamedTuple

import torch
import numpy as np
import open3d as o3d

from modules.pose.alignment import umeyama_alignment
from modules.pose.utils import get_transformation_matrix
from modules.io.utils import read_all_poses_and_stamps, save_pcd_o3d
from modules.eval.metrics import (
    compute_accuracy_pcd,
    compute_completion_pcd,
    compute_chamfer_distance_pcd,
    compute_recall_pcd,
    compute_precision_pcd,
    compute_fscore_pcd,
    )
from modules.eval.utils import (
    round_results,
    extract_translations_from_poses,
    get_pose_matches,
    save_pointclouds_src_target,
    do_point_to_plane_icp,
    do_multiscale_colored_icp,
    )


def main(args):
    pred_path = args.pred_path
    ref_path = args.ref_path
    pred_pose_path = args.pred_pose_path
    ref_pose_path = args.ref_pose_path
    dataset = args.dataset.lower()
    ref_dataset = args.ref_dataset.lower()
    output_dir = Path(args.output_dir)
    align_scale = args.align_scale
    register_clouds = args.register_clouds
    use_color_registration = args.use_color_registration
    save_clouds = args.save_clouds
    crop_pred_cloud = args.crop_pred_cloud
    crop_ref_cloud = args.crop_ref_cloud
    vis_downsample_voxel_size = args.vis_downsample_voxel_size

    # Setup logging
    now = datetime.datetime.now()
    timestamp = now.strftime("%m-%d_%H-%M-%S")
    output_dir = output_dir / Path(f"pcd_{timestamp}")
    log_path = output_dir.absolute() / Path("log.txt")
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(levelname)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(f"Arguments: \n{json.dumps(vars(args), indent=4)}")

    # Nested dictionary to store results
    results = {
        "acc": 0.0,
        "comp": 0.0,
        "chamfer": 0.0,
        "prec": 0.0,
        "recall": 0.0,
        "fscore": 0.0,
    }

    # Load in pointclouds using open3d
    pred_pcd = o3d.io.read_point_cloud(pred_path)
    ref_pcd = o3d.io.read_point_cloud(ref_path)
    logging.info(f"Read predicted point cloud with: {len(pred_pcd.points)} points and reference point cloud with {len(ref_pcd.points)} points.")

    # Read in predicted and reference pose information for coordinate frame alignment.
    poses, stamps = read_all_poses_and_stamps(pred_pose_path, dataset)
    ref_poses, ref_stamps = read_all_poses_and_stamps(ref_pose_path, ref_dataset)

    # Get matched trajectories (matching translations of valid frames)
    traj_m, ref_traj_m = get_pose_matches(
        poses,
        ref_poses,
        stamps,
        ref_stamps,
        dataset,
        )

    # Align the two point clouds using poses
    rot, t, s = umeyama_alignment(traj_m, ref_traj_m, align_scale)
    T =  get_transformation_matrix(rot, t)
    logging.info(f"The computed coordinate frame alignment transformation is:\n{T.numpy()}")

    # Save clouds before alignment
    if save_clouds:
        save_pointclouds_src_target(
            pred_pcd,
            ref_pcd,
            vis_downsample_voxel_size,
            filename="pcd_pre_align.ply",
            output_dir=output_dir,
            )

    # Align and scale
    pred_pcd = pred_pcd.transform(T)
    pred_pcd = pred_pcd.scale(scale=s, center=pred_pcd.get_center())

    # Save clouds after alignment
    if save_clouds:
        save_pointclouds_src_target(
            pred_pcd,
            ref_pcd,
            vis_downsample_voxel_size,
            filename="pcd_pre_register.ply",
            output_dir=output_dir,
            )

    # Crop the clouds using the bounding box of the other (for fair metric comparison)
    if crop_pred_cloud:
        logging.info("Cropping the predicted cloud to be within the bounding box of the reference cloud...")
        bounding_box = ref_pcd.get_axis_aligned_bounding_box()
        pred_pcd = pred_pcd.crop(bounding_box)

        if save_clouds:
            save_pointclouds_src_target(
                pred_pcd,
                ref_pcd,
                vis_downsample_voxel_size,
                filename="pcd_after_pred_crop.ply",
                output_dir=output_dir,
                )

    if crop_ref_cloud:
        logging.info("Cropping reference cloud to be within the bounding box of the predicted cloud...")
        bounding_box = pred_pcd.get_axis_aligned_bounding_box()
        ref_pcd = ref_pcd.crop(bounding_box)

        if save_clouds:
            save_pointclouds_src_target(
                pred_pcd,
                ref_pcd,
                vis_downsample_voxel_size,
                filename="pcd_after_ref_crop.ply",
                output_dir=output_dir,
                )

    # Register the two clouds
    if register_clouds:
        eval_pre = o3d.pipelines.registration.evaluate_registration(
            pred_pcd,
            ref_pcd,
            max_correspondence_distance=0.02, # TODO this distance might not be metric/meaningful if we use colmap clouds as reference
            )
        logging.info(f"Point cloud agreement before registration:\n fitness: {eval_pre.fitness}\n inlier rmse: {eval_pre.inlier_rmse}\n correspondences: {len(eval_pre.correspondence_set)}")

        if use_color_registration:
            logging.info("Registering clouds using multiscale colored ICP, this might take some time...")
            icp_trans = do_multiscale_colored_icp(pred_pcd, ref_pcd)
        else:
            logging.info("Registering clouds using the point-to-plane ICP algorithm...")
            icp_trans = do_point_to_plane_icp(pred_pcd, ref_pcd)

        pred_pcd.transform(icp_trans)
        logging.info(f"Transformed the predicted cloud to register it with the reference cloud with transform:\n{icp_trans}.")

        eval_post = o3d.pipelines.registration.evaluate_registration(
            pred_pcd,
            ref_pcd,
            max_correspondence_distance=0.02,
            )
        logging.info(f"Point cloud agreement after registration:\n fitness: {eval_post.fitness}\n inlier rmse: {eval_post.inlier_rmse}\n correspondences: {len(eval_post.correspondence_set)}")

        if save_clouds:
            save_pointclouds_src_target(
                pred_pcd,
                ref_pcd,
                vis_downsample_voxel_size,
                filename="pcd_post_register.ply",
                output_dir=output_dir,
                )

    # Compute metrics
    pred_to_ref_dists = torch.tensor(pred_pcd.compute_point_cloud_distance(ref_pcd)).double()
    ref_to_pred_dists = torch.tensor(ref_pcd.compute_point_cloud_distance(pred_pcd)).double()
    results["acc"] = compute_accuracy_pcd(pred_to_ref_dists)
    results["comp"] = compute_completion_pcd(ref_to_pred_dists)
    results["chamfer"] = compute_chamfer_distance_pcd(results["acc"], results["comp"])
    results["prec"] = compute_precision_pcd(pred_to_ref_dists)
    results["recall"] = compute_recall_pcd(ref_to_pred_dists)
    results["fscore"] = compute_fscore_pcd(results["prec"], results["recall"])
    logging.info(f"Results: \n{json.dumps(results, indent=4)}")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="A script to compute standard metrics for an estimated pointcloud and a reference ground truth point cloud (or mesh)")

    parser.add_argument("--pred_path", type=str, help="The path to the .ply file to be evaluated.")
    parser.add_argument("--ref_path", type=str, help="The path to the .ply file to be used as reference (ground truth) for evaluation")
    parser.add_argument("--pred_pose_path", type=str, help="The path to the pose file associated with the predicted point cloud. Needs to match the format specified by '--dataset'")
    parser.add_argument("--ref_pose_path", type=str, help="The path to the pose file associated with the reference point cloud. Needs to match the format specified by '--ref_dataset'")
    parser.add_argument("--dataset", type=str, help="Name of the dataset that the pose files are formatted for. See './configs/data.py' for a list of available formats.")
    parser.add_argument("--ref_dataset", type=str, help="Name of the dataset that the reference pose files are formatted for. See './configs/data.py' for a list of available formats.")
    parser.add_argument("--output_dir", type=str, default="./evaluation/eval_results", help="The directory where the logs and the outputs will be saved. Outputs are timestamped.")
    parser.add_argument("--align_scale", action="store_true", help="Flag to whether align the scales of the two point clouds when aligning the coordinate frames using the poses.")
    parser.add_argument("--register_clouds", action="store_true", help="Flag to whether register the two point clouds after alignment.")
    parser.add_argument("--use_color_registration", action="store_true", help="Flag to whether use multi-scale colored ICP registration instead of the purely geometry based variant.")
    parser.add_argument("--crop_pred_cloud", action="store_true", help="Flag to whether crop the predicted cloud to fit into the bounding box of the reference cloud for metric compuation.")
    parser.add_argument("--crop_ref_cloud", action="store_true", help="Flag to whether crop the reference cloud to fit into the bounding box of the predicted cloud for metric compuation.")
    parser.add_argument("--save_clouds", action="store_true", help="Flag to whether save intermediate representations of the clouds, useful for debugging.")
    parser.add_argument("--vis_downsample_voxel_size", type=float, default=0.05, help="Voxel size used for downsampling the saved clouds. Useful for reducing memory footprint of debugging output.")

    args = parser.parse_args()
    main(args)