#!/usr/bin/env bash

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

###################
# kitti360_0_mini
###################
python evaluation/evaluate_poses.py -d /usr/stud/kaa/data/root/kitti360_0_mini/eval/poses_rgbd -g /usr/stud/kaa/data/root/kitti360_0_mini/eval/gt_poses.txt -o /usr/stud/kaa/data/root/kitti360_0_mini/eval/results --dataset custom --ref_dataset kitti360 --align_scale --exp_name rgbd_with_scale

python evaluation/evaluate_poses.py -d /usr/stud/kaa/data/root/kitti360_0_mini/eval/poses_rgbd -g /usr/stud/kaa/data/root/kitti360_0_mini/eval/gt_poses.txt -o /usr/stud/kaa/data/root/kitti360_0_mini/eval/results --dataset custom --ref_dataset kitti360  --exp_name rgbd_no_scale

python evaluation/evaluate_poses.py -d /usr/stud/kaa/data/root/kitti360_0_mini/eval/poses_mono -g /usr/stud/kaa/data/root/kitti360_0_mini/eval/gt_poses.txt -o /usr/stud/kaa/data/root/kitti360_0_mini/eval/results --dataset custom --ref_dataset kitti360  --exp_name mono_no_scale

python evaluation/evaluate_poses.py -d /usr/stud/kaa/data/root/kitti360_0_mini/eval/poses_mono -g /usr/stud/kaa/data/root/kitti360_0_mini/eval/gt_poses.txt -o /usr/stud/kaa/data/root/kitti360_0_mini/eval/results --dataset custom --ref_dataset kitti360 --align_scale --exp_name mono_scale

python evaluation/evaluate_poses.py -d /usr/stud/kaa/data/root/kitti360_0_mini/eval/poses_colmap -g /usr/stud/kaa/data/root/kitti360_0_mini/eval/gt_poses.txt -o /usr/stud/kaa/data/root/kitti360_0_mini/eval/results --dataset colmap --ref_dataset kitti360 --align_scale --exp_name colmap_scale

###################
# kitti 07
###################
# rgbd with scale
python evaluation/evaluate_poses.py -d /usr/stud/kaa/data/root/kitti07/eval/poses_rgbd -g /usr/stud/kaa/data/root/kitti07/eval/gt_poses.txt -o /usr/stud/kaa/data/root/kitti07/eval/results --dataset custom --ref_dataset kitti --align_scale --exp_name rgbd_with_scale

#rgbd no scale
python evaluation/evaluate_poses.py -d /usr/stud/kaa/data/root/kitti07/eval/poses_rgbd -g /usr/stud/kaa/data/root/kitti07/eval/gt_poses.txt -o /usr/stud/kaa/data/root/kitti07/eval/results --dataset custom --ref_dataset kitti  --exp_name rgbd_no_scale

# mono
python evaluation/evaluate_poses.py -d /usr/stud/kaa/data/root/kitti07/eval/poses_mono -g /usr/stud/kaa/data/root/kitti07/eval/gt_poses.txt -o /usr/stud/kaa/data/root/kitti07/eval/results --dataset custom --ref_dataset kitti --align_scale --exp_name mono_with_scale

# colmap exhaustive
python evaluation/evaluate_poses.py -d /usr/stud/kaa/data/root/kitti07/eval/poses_colmap -g /usr/stud/kaa/data/root/kitti07/eval/gt_poses.txt -o /usr/stud/kaa/data/root/kitti07/eval/results --dataset colmap --ref_dataset kitti --align_scale --exp_name colmap_with_scale