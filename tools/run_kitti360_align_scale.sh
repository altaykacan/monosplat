#!/usr/bin/env bash

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

###################
# kitti360_0_mini
###################

# COLMAP pose sparse scale est
python /usr/stud/kaa/thesis/DEN-Splatting/5_align_scale.py -r /usr/stud/kaa/data/root/kitti360_0_mini --intrinsics 552.55 552.55 682.05 238.77 --dataset colmap --alignment_type sparse --sparse_cloud_path /usr/stud/kaa/data/root/kitti360_0_mini/poses/colmap/sparse/0/points3D.txt --exp_name sparse_scale_colmap --start_id 500 --end_id 1500 --depth_type precomputed

# SLAM pose sparse scale est --> no cloud


# COLMAP pose dense scale est
python /usr/stud/kaa/thesis/DEN-Splatting/5_align_scale.py -r /usr/stud/kaa/data/root/kitti360_0_mini --intrinsics 552.55 552.55 682.05 238.77 --dataset colmap --alignment_type dense --exp_name dense_scale_colmap --start_id 500 --end_id 1500 --depth_type precomputed --seg_model_type precomputed --flow_steps 1 2 4 6

# SLAM pose dense scale est --> mono
python /usr/stud/kaa/thesis/DEN-Splatting/5_align_scale.py -r /usr/stud/kaa/data/root/kitti360_0_mini --pose_path /usr/stud/kaa/data/root/kitti360_0_mini/poses/slam/1_mono_CameraTrajectory.txt --intrinsics 552.55 552.55 682.05 238.77 --dataset custom --alignment_type dense --exp_name dense_scale_orb_mono --start_id 500 --end_id 1500 --depth_type precomputed --seg_model_type precomputed --flow_steps 1 2 4 6 --padded_img_name_length 10

# SLAM pose dense scale est --> rgbd, should give 1.0
python /usr/stud/kaa/thesis/DEN-Splatting/5_align_scale.py -r /usr/stud/kaa/data/root/kitti360_0_mini --pose_path /usr/stud/kaa/data/root/kitti360_0_mini/poses/slam/1_rgbd_CameraTrajectory.txt --intrinsics 552.55 552.55 682.05 238.77 --dataset custom --alignment_type dense --exp_name dense_scale_orb_rgbd --start_id 500 --end_id 1500 --depth_type precomputed --seg_model_type precomputed --flow_steps 1 2 4 6 --padded_img_name_length 10
