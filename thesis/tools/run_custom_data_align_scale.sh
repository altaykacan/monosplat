#!/usr/bin/env bash

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

########
# ds01
########

# colmap poses dense
python 5_align_scale.py -r /usr/stud/kaa/data/root/ds01/ -p /usr/stud/kaa/data/root/ds01/poses/colmap/sparse/0/images.txt --intrinsics 534.045 534.045 512 288 -a dense --dataset colmap --seg_model_type precomputed --flow_steps 2 4 6 8 --exp_name scale_colmap

# orb mono
# python 5_align_scale.py -r /usr/stud/kaa/data/root/ds01/ -p /usr/stud/kaa/data/root/ds01/poses/slam/1_mono_KeyFrameTrajectory.txt --intrinsics 534.045 534.045 512 288 -a dense --dataset custom --seg_model_type precomputed --flow_steps 2 4 6 8 --exp_name scale_mono_keyframe

python 5_align_scale.py -r /usr/stud/kaa/data/root/ds01/ -p /usr/stud/kaa/data/root/ds01/poses/slam/1_mono_CameraTrajectory.txt --intrinsics 534.045 534.045 512 288 -a dense --dataset custom --seg_model_type precomputed --flow_steps 2 4 6 8 --exp_name scale_mono

# orb rgbd, no need run actually, it has to be already 1.0
python 5_align_scale.py -r /usr/stud/kaa/data/root/ds01/ -p /usr/stud/kaa/data/root/ds01/poses/slam/1_rgbd_CameraTrajectory.txt --intrinsics 534.045 534.045 512 288 -a dense --dataset custom --seg_model_type precomputed --flow_steps 2 4 6 8 --exp_name scale_rgbd

# python 5_align_scale.py -r /usr/stud/kaa/data/root/ds01/ -p /usr/stud/kaa/data/root/ds01/poses/slam/1_rgbd_KeyFrameTrajectory.txt --intrinsics 534.045 534.045 512 288 -a dense --dataset custom --seg_model_type precomputed --flow_steps 2 4 6 8 --exp_name scale_rgbd_keyframe


########
# ds02
########
# colmap poses dense
python 5_align_scale.py -r /usr/stud/kaa/data/root/ds02 -p /usr/stud/kaa/data/root/ds02/poses/colmap/sparse/0/images.txt --intrinsics 534.045 534.045 512 288 -a dense --dataset colmap --seg_model_type precomputed --flow_steps 2 4 6 8 --exp_name scale_dense_colmap

# orb mono
python 5_align_scale.py -r /usr/stud/kaa/data/root/ds02 -p /usr/stud/kaa/data/root/ds02/poses/slam/1_mono_CameraTrajectory.txt --intrinsics 534.045 534.045 512 288 -a dense --dataset custom --seg_model_type precomputed --flow_steps 2 4 6 8 --exp_name scale_dense_mono


########
# muc02
########

# colmap poses dense
python 5_align_scale.py -r /usr/stud/kaa/data/root/muc02 -p /usr/stud/kaa/data/root/muc02/poses/colmap/sparse/0/images.txt --intrinsics 535.0 535.0 512 288 -a dense --dataset colmap --seg_model_type precomputed --flow_steps 2 4 6 8 --exp_name scale_dense_colmap

# orb mono
python 5_align_scale.py -r /usr/stud/kaa/data/root/muc02 -p  /usr/stud/kaa/data/root/muc02/poses/slam/1_mono_CameraTrajectory.txt --intrinsics 535.0 535.0 512 288 -a dense --dataset custom --seg_model_type precomputed --flow_steps 2 4 6 8 --exp_name scale_dense_mono


########
# ottendichler
########

# colmap poses dense
python 5_align_scale.py -r /usr/stud/kaa/data/root/ottendichler -p /usr/stud/kaa/data/root/ottendichler/poses/colmap/sparse/0/images.txt --intrinsics 525.75 525.75 512 288 -a dense --dataset colmap --seg_model_type precomputed --flow_steps 2 4 6 8 --exp_name scale_dense_colmap

# orb mono
python 5_align_scale.py -r /usr/stud/kaa/data/root/ottendichler -p /usr/stud/kaa/data/root/ottendichler/poses/slam/1_mono_CameraTrajectory.txt --intrinsics 525.75 525.75 512 288 -a dense --dataset custom --seg_model_type precomputed --flow_steps 2 4 6 8 --exp_name scale_dense_mono

########
# ostspange
########

# colmap poses dense
python 5_align_scale.py -r /usr/stud/kaa/data/root/ostspange -p /usr/stud/kaa/data/root/ostspange/poses/colmap/sparse/0/images.txt --intrinsics 533.72 533.72 512 288 -a dense --dataset colmap --seg_model_type precomputed --flow_steps 2 4 6 8 --exp_name scale_dense_colmap

# orb mono
python 5_align_scale.py -r /usr/stud/kaa/data/root/ostspange -p /usr/stud/kaa/data/root/ostspange/poses/slam/1_mono_CameraTrajectory.txt  --intrinsics 533.72 533.72 512 288 -a dense --dataset custom --seg_model_type precomputed --flow_steps 2 4 6 8 --exp_name scale_dense_mono