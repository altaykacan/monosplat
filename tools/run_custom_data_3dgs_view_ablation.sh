#!/usr/bin/env bash

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

########
# ds01
########

####
# View sparsity ablation, all with colmap poses, masked, init cloud and skydome
####

# no reg
# val block size 1
python 7_train_gaussians.py -r /usr/stud/kaa/data/root/ds01 --recon_name 15_colmap_dense_cloud_with_init_and_skydome -o colmap_view_sparsity_block_1_no_reg --use_mask --mask_path /usr/stud/kaa/data/root/ds01/data/masks_moveable --port 6661 ---consecutive_val_block_size 1

# val block size 4
python 7_train_gaussians.py -r /usr/stud/kaa/data/root/ds01 --recon_name 15_colmap_dense_cloud_with_init_and_skydome -o colmap_view_sparsity_block_4_no_reg --use_mask --mask_path /usr/stud/kaa/data/root/ds01/data/masks_moveable --port 6661 --consecutive_val_block_size 4

# val block size 8
python 7_train_gaussians.py -r /usr/stud/kaa/data/root/ds01 --recon_name 15_colmap_dense_cloud_with_init_and_skydome -o colmap_view_sparsity_block_8_no_reg --use_mask --mask_path /usr/stud/kaa/data/root/ds01/data/masks_moveable --port 6661 --consecutive_val_block_size 8

# val block size 16
python 7_train_gaussians.py -r /usr/stud/kaa/data/root/ds01 --recon_name 15_colmap_dense_cloud_with_init_and_skydome -o colmap_view_sparsity_block_16_no_reg --use_mask --mask_path /usr/stud/kaa/data/root/ds01/data/masks_moveable --port 6666 --consecutive_val_block_size 16

# depth reg
# val block size 1
python 7_train_gaussians.py -r /usr/stud/kaa/data/root/ds01 --recon_name 15_colmap_dense_cloud_with_init_and_skydome -o colmap_view_sparsity_block_1_d_reg --use_mask --mask_path /usr/stud/kaa/data/root/ds01/data/masks_moveable --port 6666 --consecutive_val_block_size 1 --use_gt_depth

# val block size 4
python 7_train_gaussians.py -r /usr/stud/kaa/data/root/ds01 --recon_name 15_colmap_dense_cloud_with_init_and_skydome -o colmap_view_sparsity_block_4_d_reg --use_mask --mask_path /usr/stud/kaa/data/root/ds01/data/masks_moveable --port 6666 --consecutive_val_block_size 4 --use_gt_depth

# val block size 8
python 7_train_gaussians.py -r /usr/stud/kaa/data/root/ds01 --recon_name 15_colmap_dense_cloud_with_init_and_skydome -o colmap_view_sparsity_block_8_d_reg --use_mask --mask_path /usr/stud/kaa/data/root/ds01/data/masks_moveable --port 6666 --consecutive_val_block_size 8 --use_gt_depth

# val block size 16
python 7_train_gaussians.py -r /usr/stud/kaa/data/root/ds01 --recon_name 15_colmap_dense_cloud_with_init_and_skydome -o colmap_view_sparsity_block_16_d_reg --use_mask --mask_path /usr/stud/kaa/data/root/ds01/data/masks_moveable --port 6666 --consecutive_val_block_size 16 --use_gt_depth

# depth and normal reg
# val block size 1
python 7_train_gaussians.py -r /usr/stud/kaa/data/root/ds01 --recon_name 15_colmap_dense_cloud_with_init_and_skydome -o colmap_view_sparsity_block_1_dn_reg --use_mask --mask_path /usr/stud/kaa/data/root/ds01/data/masks_moveable --port 6661 --consecutive_val_block_size 1 --use_gt_depth --use_gt_normal

# val block size 4
python 7_train_gaussians.py -r /usr/stud/kaa/data/root/ds01 --recon_name 15_colmap_dense_cloud_with_init_and_skydome -o colmap_view_sparsity_block_4_dn_reg --use_mask --mask_path /usr/stud/kaa/data/root/ds01/data/masks_moveable --port 6661 --consecutive_val_block_size 4 --use_gt_depth --use_gt_normal

# val block size 8
python 7_train_gaussians.py -r /usr/stud/kaa/data/root/ds01 --recon_name 15_colmap_dense_cloud_with_init_and_skydome -o colmap_view_sparsity_block_8_dn_reg --use_mask --mask_path /usr/stud/kaa/data/root/ds01/data/masks_moveable --port 6661 --consecutive_val_block_size 8 --use_gt_depth --use_gt_normal
