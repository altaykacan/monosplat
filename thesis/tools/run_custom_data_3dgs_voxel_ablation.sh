#!/usr/bin/env bash

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export ROOTPATH="/usr/stud/kaa/data/root"

########
# ds01
########

####
# Downsample voxel size ablation, all colmap pose, masked, with init & skydome, no reg
####
# 0.01
python 7_train_gaussians.py -r $ROOTPATH/ds01 --recon_name 16_colmap_denser_cloud_voxel_0_01 -o colmap_voxel_0_01 --use_mask --mask_path $ROOTPATH/ds01/data/masks_moveable --port 6661

# 0.025
python 7_train_gaussians.py -r $ROOTPATH/ds01 --recon_name 17_colmap_denser_cloud_voxel_0_025 -o colmap_voxel_0_025 --use_mask --mask_path $ROOTPATH/ds01/data/masks_moveable --port 6661

# 0.05
python 7_train_gaussians.py -r $ROOTPATH/ds01 --recon_name 18_colmap_denser_cloud_voxel_0_05 -o colmap_voxel_0_05 --use_mask --mask_path $ROOTPATH/ds01/data/masks_moveable --port 6661

# 0.1
python 7_train_gaussians.py -r $ROOTPATH/ds01 --recon_name 19_colmap_denser_cloud_voxel_0_1 -o colmap_voxel_0_1 --use_mask --mask_path $ROOTPATH/ds01/data/masks_moveable --port 6661

# 1.0
python 7_train_gaussians.py -r $ROOTPATH/ds01 --recon_name 20_colmap_denser_cloud_voxel_1 -o colmap_voxel_1 --use_mask --mask_path $ROOTPATH/ds01/data/masks_moveable --port 6661

# 5.0
python 7_train_gaussians.py -r $ROOTPATH/ds01 --recon_name 21_colmap_denser_cloud_voxel_5 -o colmap_voxel_5 --use_mask --mask_path $ROOTPATH/ds01/data/masks_moveable --port 6661

# 10.0
python 7_train_gaussians.py -r $ROOTPATH/ds01 --recon_name 22_colmap_denser_cloud_voxel_10 -o colmap_voxel_10 --use_mask --mask_path $ROOTPATH/ds01/data/masks_moveable --port 6661

# ####
# # Downsample voxel size ablation, all colmap pose, masked, with init & skydome, depth reg
# ####
# # 0.01
# python 7_train_gaussians.py -r $ROOTPATH/ds01 --recon_name 16_colmap_denser_cloud_voxel_0_01 -o colmap_voxel_0_01_d_reg --use_mask --mask_path $ROOTPATH/ds01/data/masks_moveable --port 6661 --use_gt_depth

# # 0.025
# python 7_train_gaussians.py -r $ROOTPATH/ds01 --recon_name 17_colmap_denser_cloud_voxel_0_025 -o colmap_voxel_0_025_d_reg --use_mask --mask_path $ROOTPATH/ds01/data/masks_moveable --port 6661 --use_gt_depth

# # 0.05
# python 7_train_gaussians.py -r $ROOTPATH/ds01 --recon_name 18_colmap_denser_cloud_voxel_0_05 -o colmap_voxel_0_05_d_reg --use_mask --mask_path $ROOTPATH/ds01/data/masks_moveable --port 6661 --use_gt_depth

# # 0.1
# python 7_train_gaussians.py -r $ROOTPATH/ds01 --recon_name 19_colmap_denser_cloud_voxel_0_1 -o colmap_voxel_0_1_d_reg --use_mask --mask_path $ROOTPATH/ds01/data/masks_moveable --port 6661 --use_gt_depth

# # 1.0
# python 7_train_gaussians.py -r $ROOTPATH/ds01 --recon_name 20_colmap_denser_cloud_voxel_1 -o colmap_voxel_1_d_reg --use_mask --mask_path $ROOTPATH/ds01/data/masks_moveable --port 6661 --use_gt_depth

# # 5.0
# python 7_train_gaussians.py -r $ROOTPATH/ds01 --recon_name 21_colmap_denser_cloud_voxel_5 -o colmap_voxel_5_d_reg --use_mask --mask_path $ROOTPATH/ds01/data/masks_moveable --port 6661 --use_gt_depth

# # 10.0
# python 7_train_gaussians.py -r $ROOTPATH/ds01 --recon_name 22_colmap_denser_cloud_voxel_10 -o colmap_voxel_10_d_reg --use_mask --mask_path $ROOTPATH/ds01/data/masks_moveable --port 6661 --use_gt_depth