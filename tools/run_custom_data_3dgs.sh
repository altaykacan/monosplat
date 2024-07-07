#!/usr/bin/env bash

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

########
# ds01
########

####
# Visual quality experiments
####
# colmap baseline
python 7_train_gaussians.py -r /usr/stud/kaa/data/root/ds01 --colmap_dir /usr/stud/kaa/data/root/ds01/poses/colmap -o colmap_baseline --port 6661

# colmap baseline masked
python 7_train_gaussians.py -r /usr/stud/kaa/data/root/ds01 --colmap_dir /usr/stud/kaa/data/root/ds01/poses/colmap -o colmap_baseline_masked --port 6661 --use_mask --mask_path /usr/stud/kaa/data/root/ds01/data/masks_moveable

# Dense cloud with colmap poses
python 7_train_gaussians.py -r /usr/stud/kaa/data/root/ds01 --recon_name 24_colmap_dense_cloud -o colmap_dense  --port 6661

python 7_train_gaussians.py -r /usr/stud/kaa/data/root/ds01 --recon_name 24_colmap_dense_cloud -o colmap_dense_masked --use_mask --mask_path /usr/stud/kaa/data/root/ds01/data/masks_moveable --port 6661

# Dense cloud with colmap poses and init cloud
python 7_train_gaussians.py -r /usr/stud/kaa/data/root/ds01 --recon_name 14_colmap_dense_cloud_with_init -o colmap_dense_masked_with_init --use_mask --mask_path /usr/stud/kaa/data/root/ds01/data/masks_moveable --port 6661

# Dense cloud with colmap poses and skydome
python 7_train_gaussians.py -r /usr/stud/kaa/data/root/ds01 --recon_name 27_colmap_dense_cloud_with_skydome -o colmap_dense_masked_with_skydome --use_mask --mask_path /usr/stud/kaa/data/root/ds01/data/masks_moveable --port 6666

# Dense cloud with colmap poses, init cloud and skydome
python 7_train_gaussians.py -r /usr/stud/kaa/data/root/ds01 --recon_name 15_colmap_dense_cloud_with_init_and_skydome -o colmap_dense_masked_with_init_and_skydome --use_mask --mask_path /usr/stud/kaa/data/root/ds01/data/masks_moveable --port 6661

# Orb rgb-d
python 7_train_gaussians.py -r /usr/stud/kaa/data/root/ds01 --recon_name 28_rgbd_debug -o orb_rgbd --use_mask --mask_path /usr/stud/kaa/data/root/ds01/data/masks_moveable --port 6666

####
# Depth reg
####
# Dense cloud with colmap poses + depth reg
python 7_train_gaussians.py -r /usr/stud/kaa/data/root/ds01 --recon_name 24_colmap_dense_cloud -o colmap_dense_masked_depth_reg --use_mask --mask_path /usr/stud/kaa/data/root/ds01/data/masks_moveable --port 6661 --use_gt_depth

# Dense cloud with colmap poses, init cloud and sky dome + depth reg
python 7_train_gaussians.py -r /usr/stud/kaa/data/root/ds01 --recon_name 15_colmap_dense_cloud_with_init_and_skydome -o colmap_dense_masked_with_init_and_skydome_depth_reg --use_mask --mask_path /usr/stud/kaa/data/root/ds01/data/masks_moveable --port 6661 --use_gt_depth

# orb rgb-d, depth reg
python 7_train_gaussians.py -r /usr/stud/kaa/data/root/ds01 --recon_name 28_rgbd_debug -o orb_rgbd_d_reg --use_mask --mask_path /usr/stud/kaa/data/root/ds01/data/masks_moveable --port 6666 --use_gt_depth


####
# Normal reg
####

# orb rgb-d, normal reg
python 7_train_gaussians.py -r /usr/stud/kaa/data/root/ds01 --recon_name 28_rgbd_debug -o orb_rgbd_n_reg --use_mask --mask_path /usr/stud/kaa/data/root/ds01/data/masks_moveable --port 6666 --use_gt_normal
python 7_train_gaussians.py -r /usr/stud/kaa/data/root/ds01 --recon_name 28_rgbd_debug -o orb_rgbd_n_reg_lambda_0_1 --use_mask --mask_path /usr/stud/kaa/data/root/ds01/data/masks_moveable --port 6666 --use_gt_normal --lambda_normal 0.1
python 7_train_gaussians.py -r /usr/stud/kaa/data/root/ds01 --recon_name 28_rgbd_debug -o orb_rgbd_n_reg_lambda_0_4 --use_mask --mask_path /usr/stud/kaa/data/root/ds01/data/masks_moveable --port 6666 --use_gt_normal --lambda_normal 0.4


####
# Depth and normal reg
####
# # Dense cloud with colmap poses, init cloud and sky dome + depth reg
# python 7_train_gaussians.py -r /usr/stud/kaa/data/root/ds01 --recon_name 15_colmap_dense_cloud_with_init_and_skydome -o colmap_dense_masked_with_init_and_skydome_dn_reg --use_mask --mask_path /usr/stud/kaa/data/root/ds01/data/masks_moveable --port 6661 --use_gt_depth --use_gt_normal


########
# ds_combined
########
# no reg
python 7_train_gaussians.py -r /usr/stud/kaa/data/root/ds_combined --recon_name 25_colmap_dense_test -o colmap_dense_masked_with_init_and_skydome --port 6676 --use_mask --mask_path /usr/stud/kaa/data/root/ds_combined/data/masks_moveable --test_iterations 1000 5000 7000 10000 15000 20000 25000 30000 40000 50000 --save_iterations 7000 30000 50000

# depth reg
python 7_train_gaussians.py -r /usr/stud/kaa/data/root/ds_combined --recon_name 25_colmap_dense_test -o colmap_dense_masked_with_init_and_skydome_d_reg --port 6676  --use_mask --mask_path /usr/stud/kaa/data/root/ds_combined/data/masks_moveable --use_gt_depth --test_iterations 1000 5000 7000 10000 15000 20000 25000 30000 40000 50000 --save_iterations 7000 30000 50000

# normal reg
python 7_train_gaussians.py -r /usr/stud/kaa/data/root/ds_combined --recon_name 25_colmap_dense_test -o colmap_dense_masked_with_init_and_skydome_n_reg --port 6676  --use_mask --mask_path /usr/stud/kaa/data/root/ds_combined/data/masks_moveable --use_gt_normal --test_iterations 1000 5000 7000 10000 15000 20000 25000 30000 40000 50000 --save_iterations 7000 30000 50000

python 7_train_gaussians.py -r /usr/stud/kaa/data/root/ds_combined --recon_name 25_colmap_dense_test -o colmap_dense_masked_with_init_and_skydome_dn_reg --port 6676  --use_mask --mask_path /usr/stud/kaa/data/root/ds_combined/data/masks_moveable --use_gt_depth --use_gt_normal --test_iterations 1000 5000 7000 10000 15000 20000 25000 30000 40000 50000 --save_iterations 7000 30000 50000

