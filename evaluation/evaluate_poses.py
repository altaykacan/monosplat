"""
Script for evaluating average metrics over multiple pose predictions and
comparing it to one reference pose.
"""
import json
import argparse
import logging
import datetime
import pprint
from pathlib import Path
from collections import OrderedDict

import torch
import numpy as np
from tqdm import tqdm

from modules.pose.alignment import umeyama_alignment
from modules.pose.visualization import save_traj
from modules.eval.utils import round_results, extract_translations_from_poses, get_pose_matches, save_rpe_plot
from modules.eval.metrics import compute_ate, compute_rpe
from modules.eval.tum_rgbd_tools.associate import associate
from modules.eval.tum_rgbd_tools.evaluate_ate import align
from modules.io.utils import find_all_files_in_dir, read_all_poses_and_stamps
from configs.data import EVAL_DECIMAL_POINTS

def main(args):
    pose_dir = Path(args.pose_dir)
    ref_pose_path = Path(args.ref_pose_path)
    output_dir = Path(args.output_dir)
    dataset = args.dataset.lower()
    ref_dataset = args.ref_dataset.lower()
    align_scale = args.align_scale

    # Setup logging
    now = datetime.datetime.now()
    timestamp = now.strftime("%m-%d_%H-%M-%S")
    output_dir = output_dir / Path(f"pose_{timestamp}")
    log_path = output_dir.absolute() / Path("log.txt")
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(levelname)s - %(message)s')
    logging.info(f"Arguments: \n{json.dumps(vars(args), indent=4)}")

    # Nested dictionary to store results, keys are metrics with dictionaries for sequences
    results = OrderedDict()
    results["ate"] = OrderedDict()
    results["rpe"] = OrderedDict()

    # Read in paths to all estimated pose files
    pose_paths = find_all_files_in_dir(pose_dir, extension=".txt")

    if len(pose_paths) == 0:
        raise RuntimeError(f"No poses have been found, please check your pose_dir input: {pose_dir}")

    # Iterate over the multiple pose files
    for i, pose_path in tqdm(enumerate(pose_paths)):
        logging.info(f"Processing pose file {pose_path.name}")

        # Get timestamps for which we have predicted poses
        poses, stamps = read_all_poses_and_stamps(pose_path, dataset)
        ref_poses, ref_stamps = read_all_poses_and_stamps(ref_pose_path, ref_dataset)

        # Get matched trajectories (translations of valid poses)
        traj_m, ref_traj_m = get_pose_matches(poses,
                                              ref_poses,
                                              stamps,
                                              ref_stamps,
                                              dataset
                                              )

        # Align rotation, translation, and (optionally) the scale
        rot, t, s = umeyama_alignment(traj_m, ref_traj_m, align_scale)
        rot_tum, t_tum, trans_error_tum = align(traj_m.numpy(), ref_traj_m.numpy())

        # Compute errors
        trans_error_ate = compute_ate(traj_m, ref_traj_m, rot, t, s)
        stamps_rpe, trans_error_rpe, rot_error_rpe = compute_rpe(poses, ref_poses, stamps, ref_stamps, s)

        # Save results
        results["ate"][f"traj_{i}"] = OrderedDict([
            ("pose_path", str(pose_path.name)),
            ("pose_pairs", len(trans_error_ate)),
            ("rmse", torch.sqrt(torch.dot(trans_error_ate, trans_error_ate) / len(trans_error_ate)).item()),
            ("mean", torch.mean(trans_error_ate).item()),
            ("median", torch.median(trans_error_ate).item()),
            ("std", torch.std(trans_error_ate).item()),
            ("min", torch.min(trans_error_ate).item()),
            ("max", torch.max(trans_error_ate).item()),
        ])

        results["rpe"][f"traj_{i}"] = OrderedDict([
            ("pose_pairs", len(trans_error_rpe)),
            ("trans_rmse", torch.sqrt(torch.dot(trans_error_rpe, trans_error_rpe) / len(trans_error_rpe)).item()),
            ("trans_mean", torch.mean(trans_error_rpe).item()),
            ("trans_median", torch.median(trans_error_rpe).item()),
            ("trans_std", torch.std(trans_error_rpe).item()),
            ("trans_min", torch.min(trans_error_rpe).item()),
            ("trans_max", torch.max(trans_error_rpe).item()),

            ("rot_rmse", (torch.sqrt(torch.dot(rot_error_rpe, rot_error_rpe) / len(rot_error_rpe)) * 180 / torch.pi).item()),
            ("rot_mean", (torch.mean(rot_error_rpe) * 180 / torch.pi).item()),
            ("rot_median", (torch.median(rot_error_rpe) * 180 / torch.pi).item()),
            ("rot_std", (torch.std(rot_error_rpe) * 180 / torch.pi).item()),
            ("rot_min", (torch.min(rot_error_rpe) * 180 / torch.pi).item()),
            ("rot_max", (torch.max(rot_error_rpe) * 180 / torch.pi).item()),
        ])

        # Save plots of the trajectories
        save_traj([s * rot @ traj_m + t, ref_traj_m],
                    labels=["aligned", "ref"],
                    filename=f"{pose_path.stem}_1_umeyama_alignment.png",
                    output_dir=output_dir
        )
        save_traj([traj_m, ref_traj_m],
                    labels=["pred", "ref"],
                    filename=f"{pose_path.stem}_2_no_alignment.png",
                    output_dir=output_dir,
        )
        save_traj(
            [torch.Tensor(rot_tum).double() @ traj_m + torch.Tensor(t_tum).double(), ref_traj_m],
            labels=["aligned", "ref"],
            filename=f"{pose_path.stem}_3_tum_aligned.png",
            output_dir=output_dir,
            )
        save_traj([s * rot @ traj_m + t, ref_traj_m, traj_m],
                    labels=["aligned", "ref", "pred"],
                    filename=f"{pose_path.stem}_4_umeyama_alignment_with_pred.png",
                    output_dir=output_dir,
        )

        save_traj([s * rot @ traj_m + t, ref_traj_m],
                    labels=["aligned", "ref"],
                    filename=f"{pose_path.stem}_5_umeyama_alignment_with_diff.png",
                    output_dir=output_dir,
                    show_diff=True,
        )

        save_rpe_plot(trans_error_rpe,
                        stamps_rpe,
                        filename=f"{pose_path.stem}_6_rpe_plot.png",
                        output_dir=output_dir,
        )

    # Average over the means and rmse values of different sequences
    traj_keys = results["ate"].keys()
    num_traj = len(traj_keys)
    ate_metrics = ["rmse", "mean"]
    rpe_metrics = ["trans_rmse", "trans_mean", "rot_rmse", "rot_mean"]
    totals = {"ate": {}, "rpe": {}} # to keep track of total values accross sequences

    for traj_key in traj_keys:
        for metric in ate_metrics:
            curr_total = totals["ate"].get(metric, 0)
            curr_total += results["ate"][traj_key][metric]
            totals["ate"][metric] = curr_total

        for metric in rpe_metrics:
            curr_total = totals["rpe"].get(metric, 0)
            curr_total += results["rpe"][traj_key][metric]
            totals["rpe"][metric] = curr_total

    results["ate"]["avg"] = OrderedDict()
    results["rpe"]["avg"] = OrderedDict()

    for metric in ate_metrics:
        results["ate"]["avg"][metric] = totals["ate"][metric] / num_traj
    for metric in rpe_metrics:
        results["rpe"]["avg"][metric] = totals["rpe"][metric] / num_traj

    # Round all results to a fixed decimal point
    results = round_results(results, EVAL_DECIMAL_POINTS)

    logging.info(f"Results: \n{json.dumps(results, indent=4)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script to evaluate multiple predicted poses and compare it to one ground truth (or reference) pose. It reports metrics averaged over the multiple predicted poses.")

    parser.add_argument("--pose_dir", type=str, help="The path to where the predicted pose files are.")
    parser.add_argument("--ref_pose_path", type=str, help="The path to where the ground truth/reference pose file is.")
    parser.add_argument("--output_dir", type=str, default="./evaluation/eval_results", help="The directory where the logs and the outputs will be saved. Outputs are timestamped.")
    parser.add_argument("--dataset", type=str, help="Dataset format of the predicted pose files.")
    parser.add_argument("--ref_dataset", type=str, help="Dataset format of the ground truth/reference files.")
    parser.add_argument("--align_scale", action="store_true", help="Flag to specify whether in addition to rotations and translations, a scale factor for alignment is computed.")

    args = parser.parse_args()

    main(args)