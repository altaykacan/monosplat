#!/usr/bin/env bash

export PYTHONPATH="${PYTHONPATH}:$(pwd)"


# ds01

# colmap poses dense
python 5_align_scale.py -r /usr/stud/kaa/data/root/ds01/ -p /usr/stud/kaa/data/root/ds01/poses/colmap/sparse/0/images.txt --intrinsics 534.045 534.045 512 288 -a dense --dataset colmap --seg_model_type precomputed --flow_steps 2 4 6 8 --exp_name scale_colmap

# orb mono
python 5_align_scale.py -r /usr/stud/kaa/data/root/ds01/ -p /usr/stud/kaa/data/root/ds01/poses/slam/1_mono_KeyFrameTrajectory.txt --intrinsics 534.045 534.045 512 288 -a dense --dataset custom --seg_model_type precomputed --flow_steps 2 4 6 8 --exp_name scale_mono_keyframe

# orb rgbd
python 5_align_scale.py -r /usr/stud/kaa/data/root/ds01/ -p /usr/stud/kaa/data/root/ds01/poses/slam/1_rgbd_CameraTrajectory.txt --intrinsics 534.045 534.045 512 288 -a dense --dataset custom --seg_model_type precomputed --flow_steps 2 4 6 8 --exp_name scale_rgbd_all

python 5_align_scale.py -r /usr/stud/kaa/data/root/ds01/ -p /usr/stud/kaa/data/root/ds01/poses/slam/1_rgbd_KeyFrameTrajectory.txt --intrinsics 534.045 534.045 512 288 -a dense --dataset custom --seg_model_type precomputed --flow_steps 2 4 6 8 --exp_name scale_rgbd_keyframe