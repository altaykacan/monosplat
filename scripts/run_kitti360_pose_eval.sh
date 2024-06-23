#!/usr/bin/env bash

# KITTI360 0 Mini
python evaluation/evaluate_poses.py -d /usr/stud/kaa/data/root/kitti360_0_mini/eval/poses_rgbd -g /usr/stud/kaa/data/root/kitti360_0_mini/eval/gt_poses.txt -o /usr/stud/kaa/data/root/kitti360_0_mini/eval/results --dataset custom --ref_dataset kitti360 --align_scale --exp_name rgbd_with_scale

python evaluation/evaluate_poses.py -d /usr/stud/kaa/data/root/kitti360_0_mini/eval/poses_rgbd -g /usr/stud/kaa/data/root/kitti360_0_mini/eval/gt_poses.txt -o /usr/stud/kaa/data/root/kitti360_0_mini/eval/results --dataset custom --ref_dataset kitti360  --exp_name rgbd_no_scale

python evaluation/evaluate_poses.py -d /usr/stud/kaa/data/root/kitti360_0_mini/eval/poses_mono -g /usr/stud/kaa/data/root/kitti360_0_mini/eval/gt_poses.txt -o /usr/stud/kaa/data/root/kitti360_0_mini/eval/results --dataset custom --ref_dataset kitti360  --exp_name mono_no_scale

python evaluation/evaluate_poses.py -d /usr/stud/kaa/data/root/kitti360_0_mini/eval/poses_mono -g /usr/stud/kaa/data/root/kitti360_0_mini/eval/gt_poses.txt -o /usr/stud/kaa/data/root/kitti360_0_mini/eval/results --dataset custom --ref_dataset kitti360 --align_scale  --exp_name mono_scale