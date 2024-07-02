#!/usr/bin/env bash

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

###################
# kitti360_0_mini
###################

# Colmap sparse alignment scale
python 6_create_pointcloud.py --root_dir "/usr/stud/kaa/data/root/kitti360_0_mini" --recon_name "colmap_sparse_scale" --reconstructor "simple" --backprojector "simple" --dataset "colmap" --depth_model precomputed --normal_model precomputed --batch_size 4 --intrinsics 552.55 552.55 682.05 238.77  --use_every_nth 5 --max_d 20 --dropout 0.90 --pose_scale 41.6 --padded_img_name_length 10


# Colmap dense alignment scales
python 6_create_pointcloud.py --root_dir "/usr/stud/kaa/data/root/kitti360_0_mini" --recon_name "colmap_dense_scale_flow_step_2_47" --reconstructor "simple" --backprojector "simple" --dataset "colmap" --depth_model precomputed --normal_model precomputed --batch_size 4 --intrinsics 552.55 552.55 682.05 238.77  --use_every_nth 5  --max_d 20 --dropout 0.90 --pose_scale 47.0  --padded_img_name_length 10

python 6_create_pointcloud.py --root_dir "/usr/stud/kaa/data/root/kitti360_0_mini" --recon_name "colmap_dense_scale_flow_step_6_42" --reconstructor "simple" --backprojector "simple" --dataset "colmap" --depth_model precomputed --normal_model precomputed --batch_size 4 --intrinsics 552.55 552.55 682.05 238.77  --use_every_nth 5 --max_d 20 --dropout 0.90 --pose_scale 42.0  --padded_img_name_length 10

python 6_create_pointcloud.py --root_dir "/usr/stud/kaa/data/root/kitti360_0_mini" --recon_name "colmap_dense_scale_flow_step_avg" --reconstructor "simple" --backprojector "simple" --dataset "colmap" --depth_model precomputed --normal_model precomputed --batch_size 4 --intrinsics 552.55 552.55 682.05 238.77  --use_every_nth 5 --max_d 20 --dropout 0.90 --pose_scale 45.7 --padded_img_name_length 10


# Visual ORB-SLAM3 dense scales
python 6_create_pointcloud.py --root_dir "/usr/stud/kaa/data/root/kitti360_0_mini" --pose_path /usr/stud/kaa/data/root/kitti360_0_mini/poses/slam/1_mono_CameraTrajectory.txt --recon_name "orb_rgbd" --reconstructor "simple" --backprojector "simple" --dataset "custom" --depth_model precomputed --normal_model precomputed --batch_size 4 --intrinsics 552.55 552.55 682.05 238.77  --use_every_nth 5 --max_d 20 --dropout 0.90 --pose_scale 4.13 --padded_img_name_length 10

# RGBD ORB-SLAM3
python 6_create_pointcloud.py --root_dir "/usr/stud/kaa/data/root/kitti360_0_mini" --pose_path /usr/stud/kaa/data/root/kitti360_0_mini/poses/slam/1_rgbd_CameraTrajectory.txt --recon_name "orb_rgbd" --reconstructor "simple" --backprojector "simple" --dataset "custom" --depth_model precomputed --normal_model precomputed --batch_size 4 --intrinsics 552.55 552.55 682.05 238.77  --use_every_nth 5 --max_d 20 --dropout 0.90 --pose_scale 1.0 --padded_img_name_length 10
