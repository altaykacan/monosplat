#!/usr/bin/env bash

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

########
# ds01
########

# rgbd
python evaluation/evaluate_poses.py -d /usr/stud/kaa/data/root/ds01/eval/poses_rgbd -g /usr/stud/kaa/data/root/ds01/eval/colmap_poses.txt -o /usr/stud/kaa/data/root/ds01/eval/results --dataset custom --ref_dataset colmap --align_scale --exp_name rgbd

# mono
python evaluation/evaluate_poses.py -d /usr/stud/kaa/data/root/ds01/eval/poses_mono -g /usr/stud/kaa/data/root/ds01/eval/colmap_poses.txt -o /usr/stud/kaa/data/root/ds01/eval/results --dataset custom --ref_dataset colmap --align_scale --exp_name mono