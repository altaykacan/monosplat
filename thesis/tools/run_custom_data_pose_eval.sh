#!/usr/bin/env bash

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

########
# ds01
########

# rgbd
python evaluation/evaluate_poses.py -d /usr/stud/kaa/data/root/ds01/eval/poses_rgbd -g /usr/stud/kaa/data/root/ds01/eval/colmap_poses.txt -o /usr/stud/kaa/data/root/ds01/eval/results --dataset custom --ref_dataset colmap --align_scale --exp_name rgbd

# mono
python evaluation/evaluate_poses.py -d /usr/stud/kaa/data/root/ds01/eval/poses_mono -g /usr/stud/kaa/data/root/ds01/eval/colmap_poses.txt -o /usr/stud/kaa/data/root/ds01/eval/results --dataset custom --ref_dataset colmap --align_scale --exp_name mono

########
# ds02
########

# rgbd
python evaluation/evaluate_poses.py -d /usr/stud/kaa/data/root/ds02/eval/poses_rgbd -g /usr/stud/kaa/data/root/ds02/poses/colmap/sparse/0/images.txt -o /usr/stud/kaa/data/root/ds02/eval/results --dataset custom --ref_dataset colmap --align_scale --exp_name rgbd

# mono
python evaluation/evaluate_poses.py -d /usr/stud/kaa/data/root/ds02/eval/poses_mono -g /usr/stud/kaa/data/root/ds02/poses/colmap/sparse/0/images.txt -o /usr/stud/kaa/data/root/ds02/eval/results --dataset custom --ref_dataset colmap --align_scale --exp_name mono



########
# muc02
########

# rgbd
python evaluation/evaluate_poses.py -d /usr/stud/kaa/data/root/muc02/eval/poses_rgbd -g /usr/stud/kaa/data/root/muc02/poses/colmap/sparse/0/images.txt -o /usr/stud/kaa/data/root/muc02/eval/results --dataset custom --ref_dataset colmap --align_scale --exp_name rgbd

# mono
python evaluation/evaluate_poses.py -d /usr/stud/kaa/data/root/muc02/eval/poses_mono -g /usr/stud/kaa/data/root/muc02/poses/colmap/sparse/0/images.txt -o /usr/stud/kaa/data/root/muc02/eval/results --dataset custom --ref_dataset colmap --align_scale --exp_name mono

########
# ostspange
########

# rgbd
python evaluation/evaluate_poses.py -d /usr/stud/kaa/data/root/ostspange/eval/poses_rgbd -g /usr/stud/kaa/data/root/ostspange/eval/poses_colmap/colmap_poses.txt -o /usr/stud/kaa/data/root/ostspange/eval/results --dataset custom --ref_dataset colmap --align_scale --exp_name rgbd

# mono
python evaluation/evaluate_poses.py -d /usr/stud/kaa/data/root/ostspange/eval/poses_mono -g /usr/stud/kaa/data/root/ostspange/eval/poses_colmap/colmap_poses.txt -o /usr/stud/kaa/data/root/ostspange/eval/results --dataset custom --ref_dataset colmap --align_scale --exp_name mono

########
# ottendichler
########

# rgbd
python evaluation/evaluate_poses.py -d /usr/stud/kaa/data/root/ottendichler/eval/poses_rgbd -g /usr/stud/kaa/data/root/ottendichler/poses/colmap/sparse/0/images.txt -o /usr/stud/kaa/data/root/ottendichler/eval/results --dataset custom --ref_dataset colmap --align_scale --exp_name rgbd

# mono
python evaluation/evaluate_poses.py -d /usr/stud/kaa/data/root/ottendichler/eval/poses_mono -g /usr/stud/kaa/data/root/ottendichler/poses/colmap/sparse/0/images.txt -o /usr/stud/kaa/data/root/ottendichler/eval/results --dataset custom --ref_dataset colmap --align_scale --exp_name mono