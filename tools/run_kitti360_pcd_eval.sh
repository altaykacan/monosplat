#!/usr/bin/env bash

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

###################
# kitti360_0_mini
###################

# orb mono
python evaluation/evaluate_pointcloud.py --pred_path /usr/stud/kaa/data/root/kitti360_0_mini/reconstructions/8_orb_mono/cloud.ply --ref_path /usr/stud/kaa/data/root/kitti360_0_mini/eval/gt_map.ply --pred_pose_path /usr/stud/kaa/data/root/kitti360_0_mini/reconstructions/8_orb_mono/poses.txt --ref_pose_path /usr/stud/kaa/data/root/kitti360_0_mini/eval/gt_poses.txt  --dataset custom --ref_dataset kitti360 --output_dir /usr/stud/kaa/data/root/kitti360_0_mini/eval/results --crop_ref_cloud --crop_pred_cloud --save_clouds --pose_scale 4.13 --exp_name orb_mono

python evaluation/evaluate_pointcloud.py --pred_path /usr/stud/kaa/data/root/kitti360_0_mini/reconstructions/8_orb_mono/cloud.ply --ref_path /usr/stud/kaa/data/root/kitti360_0_mini/eval/gt_map.ply --pred_pose_path /usr/stud/kaa/data/root/kitti360_0_mini/reconstructions/8_orb_mono/poses.txt --ref_pose_path /usr/stud/kaa/data/root/kitti360_0_mini/eval/gt_poses.txt  --dataset custom --ref_dataset kitti360 --output_dir /usr/stud/kaa/data/root/kitti360_0_mini/eval/results --crop_ref_cloud --crop_pred_cloud --save_clouds --pose_scale 4.13 --exp_name orb_mono_with_scale --align_scale

# orb rgbd
python evaluation/evaluate_pointcloud.py --pred_path /usr/stud/kaa/data/root/kitti360_0_mini/reconstructions/13_orb_rgbd/cloud.ply --ref_path /usr/stud/kaa/data/root/kitti360_0_mini/eval/gt_map.ply --pred_pose_path /usr/stud/kaa/data/root/kitti360_0_mini/reconstructions/13_orb_rgbd/poses.txt --ref_pose_path /usr/stud/kaa/data/root/kitti360_0_mini/eval/gt_poses.txt  --dataset custom --ref_dataset kitti360 --output_dir /usr/stud/kaa/data/root/kitti360_0_mini/eval/results  --crop_ref_cloud --crop_pred_cloud --save_clouds --pose_scale 1.0 --exp_name orb_rgbd_with_scale --align_scale

python evaluation/evaluate_pointcloud.py --pred_path /usr/stud/kaa/data/root/kitti360_0_mini/reconstructions/13_orb_rgbd/cloud.ply  --ref_path /usr/stud/kaa/data/root/kitti360_0_mini/eval/gt_map.ply --pred_pose_path  /usr/stud/kaa/data/root/kitti360_0_mini/reconstructions/13_orb_rgbd/poses.txt --ref_pose_path /usr/stud/kaa/data/root/kitti360_0_mini/eval/gt_poses.txt  --dataset custom --ref_dataset kitti360 --output_dir /usr/stud/kaa/data/root/kitti360_0_mini/eval/results  --crop_ref_cloud --crop_pred_cloud --save_clouds --pose_scale 1.0 --exp_name orb_rgbd


# colmap dense avg
python evaluation/evaluate_pointcloud.py --pred_path /usr/stud/kaa/data/root/kitti360_0_mini/reconstructions/10_colmap_dense_scale_flow_step_avg/cloud.ply --ref_path /usr/stud/kaa/data/root/kitti360_0_mini/eval/gt_map.ply --pred_pose_path /usr/stud/kaa/data/root/kitti360_0_mini/reconstructions/10_colmap_dense_scale_flow_step_avg/colmap_poses.txt --ref_pose_path /usr/stud/kaa/data/root/kitti360_0_mini/eval/gt_poses.txt  --dataset colmap --ref_dataset kitti360 --output_dir /usr/stud/kaa/data/root/kitti360_0_mini/eval/results  --crop_ref_cloud --crop_pred_cloud --save_clouds --pose_scale 45.7 --exp_name colmap_dense

python evaluation/evaluate_pointcloud.py --pred_path /usr/stud/kaa/data/root/kitti360_0_mini/reconstructions/10_colmap_dense_scale_flow_step_avg/cloud.ply --ref_path /usr/stud/kaa/data/root/kitti360_0_mini/eval/gt_map.ply --pred_pose_path /usr/stud/kaa/data/root/kitti360_0_mini/reconstructions/10_colmap_dense_scale_flow_step_avg/colmap_poses.txt --ref_pose_path /usr/stud/kaa/data/root/kitti360_0_mini/eval/gt_poses.txt  --dataset colmap --ref_dataset kitti360 --output_dir /usr/stud/kaa/data/root/kitti360_0_mini/eval/results  --crop_ref_cloud --crop_pred_cloud --save_clouds --pose_scale 45.7 --exp_name colmap_dense_with_scale --align_scale


# colmap sparse
python evaluation/evaluate_pointcloud.py --pred_path /usr/stud/kaa/data/root/kitti360_0_mini/reconstructions/11_colmap_sparse_scale/cloud.ply --ref_path /usr/stud/kaa/data/root/kitti360_0_mini/eval/gt_map.ply --pred_pose_path /usr/stud/kaa/data/root/kitti360_0_mini/reconstructions/11_colmap_sparse_scale/colmap_poses.txt --ref_pose_path /usr/stud/kaa/data/root/kitti360_0_mini/eval/gt_poses.txt  --dataset colmap --ref_dataset kitti360 --output_dir /usr/stud/kaa/data/root/kitti360_0_mini/eval/results  --crop_ref_cloud --crop_pred_cloud --save_clouds --pose_scale 41.6 --exp_name colmap_sparse

python evaluation/evaluate_pointcloud.py --pred_path /usr/stud/kaa/data/root/kitti360_0_mini/reconstructions/11_colmap_sparse_scale/cloud.ply --ref_path /usr/stud/kaa/data/root/kitti360_0_mini/eval/gt_map.ply --pred_pose_path /usr/stud/kaa/data/root/kitti360_0_mini/reconstructions/11_colmap_sparse_scale/colmap_poses.txt --ref_pose_path /usr/stud/kaa/data/root/kitti360_0_mini/eval/gt_poses.txt  --dataset colmap --ref_dataset kitti360 --output_dir /usr/stud/kaa/data/root/kitti360_0_mini/eval/results  --crop_ref_cloud --crop_pred_cloud --save_clouds --pose_scale 41.6 --exp_name colmap_sparse_with_scale --align_scale


# colmap baseline
python evaluation/evaluate_pointcloud.py --pred_path /usr/stud/kaa/data/root/kitti360_0_mini/poses/colmap/sparse/0/points3D.ply --ref_path /usr/stud/kaa/data/root/kitti360_0_mini/eval/gt_map.ply --pred_pose_path /usr/stud/kaa/data/root/kitti360_0_mini/poses/colmap/sparse/0/images.txt --ref_pose_path /usr/stud/kaa/data/root/kitti360_0_mini/eval/gt_poses.txt  --dataset colmap --ref_dataset kitti360 --output_dir /usr/stud/kaa/data/root/kitti360_0_mini/eval/results  --crop_ref_cloud --crop_pred_cloud --save_clouds --pose_scale 1.0 --exp_name colmap_baseline_with_scale --align_scale