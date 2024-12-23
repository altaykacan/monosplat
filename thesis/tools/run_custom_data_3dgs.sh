#!/usr/bin/env bash

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export ROOTPATH="/usr/stud/kaa/data/root"

########
# ds01
########

####
# Visual quality experiments
####
# colmap baseline
python 7_train_gaussians.py -r $ROOTPATH/ds01 --colmap_dir $ROOTPATH/ds01/poses/colmap -o colmap_baseline --port 6661

# colmap baseline masked
python 7_train_gaussians.py -r $ROOTPATH/ds01 --colmap_dir $ROOTPATH/ds01/poses/colmap -o colmap_baseline_masked --port 6661 --use_mask --mask_path $ROOTPATH/ds01/data/masks_moveable

# Dense cloud with colmap poses, default and masked
python 7_train_gaussians.py -r $ROOTPATH/ds01 --recon_name 24_colmap_dense_cloud -o colmap_dense  --port 6661

python 7_train_gaussians.py -r $ROOTPATH/ds01 --recon_name 24_colmap_dense_cloud -o colmap_dense_masked --use_mask --mask_path $ROOTPATH/ds01/data/masks_moveable --port 6661

# Dense cloud with colmap poses and init cloud
python 7_train_gaussians.py -r $ROOTPATH/ds01 --recon_name 14_colmap_dense_cloud_with_init -o colmap_dense_masked_with_init --use_mask --mask_path $ROOTPATH/ds01/data/masks_moveable --port 6661

# Dense cloud with colmap poses and skydome
python 7_train_gaussians.py -r $ROOTPATH/ds01 --recon_name 27_colmap_dense_cloud_with_skydome -o colmap_dense_masked_with_skydome --use_mask --mask_path $ROOTPATH/ds01/data/masks_moveable --port 6666

# Dense cloud with colmap poses, init cloud and skydome
python 7_train_gaussians.py -r $ROOTPATH/ds01 --recon_name 15_colmap_dense_cloud_with_init_and_skydome -o colmap_dense_masked_with_init_and_skydome --use_mask --mask_path $ROOTPATH/ds01/data/masks_moveable --port 6661

# Orb rgb-d
python 7_train_gaussians.py -r $ROOTPATH/ds01 --recon_name 28_rgbd_debug -o orb_rgbd --use_mask --mask_path $ROOTPATH/ds01/data/masks_moveable --port 6666

####
# Depth reg
####
# Dense cloud with colmap poses, init cloud and sky dome + depth reg
python 7_train_gaussians.py -r $ROOTPATH/ds01 --recon_name 15_colmap_dense_cloud_with_init_and_skydome -o colmap_dense_masked_with_init_and_skydome_depth_reg --use_mask --mask_path $ROOTPATH/ds01/data/masks_moveable --port 6661 --use_gt_depth

# orb rgb-d, depth reg
python 7_train_gaussians.py -r $ROOTPATH/ds01 --recon_name 28_rgbd_debug -o orb_rgbd_d_reg --use_mask --mask_path $ROOTPATH/ds01/data/masks_moveable --port 6666 --use_gt_depth

####
# Normal reg
####
# Dense cloud with colmap poses, init cloud and sky dome + normal reg
python 7_train_gaussians.py -r $ROOTPATH/ds01 --recon_name 15_colmap_dense_cloud_with_init_and_skydome -o colmap_dense_masked_with_init_and_skydome_n_reg --use_mask --mask_path $ROOTPATH/ds01/data/masks_moveable --port 6661 --use_gt_normal

# orb rgb-d, normal reg, different lambdas
python 7_train_gaussians.py -r $ROOTPATH/ds01 --recon_name 28_rgbd_debug -o orb_rgbd_n_reg --use_mask --mask_path $ROOTPATH/ds01/data/masks_moveable --port 6666 --use_gt_normal

python 7_train_gaussians.py -r $ROOTPATH/ds01 --recon_name 28_rgbd_debug -o orb_rgbd_n_reg_lambda_0_1 --use_mask --mask_path $ROOTPATH/ds01/data/masks_moveable --port 6666 --use_gt_normal --lambda_normal 0.1

python 7_train_gaussians.py -r $ROOTPATH/ds01 --recon_name 28_rgbd_debug -o orb_rgbd_n_reg_lambda_0_4 --use_mask --mask_path $ROOTPATH/ds01/data/masks_moveable --port 6666 --use_gt_normal --lambda_normal 0.4

####
# Depth and normal reg
####
# Dense cloud with colmap poses, init cloud and sky dome + depth reg
python 7_train_gaussians.py -r $ROOTPATH/ds01 --recon_name 15_colmap_dense_cloud_with_init_and_skydome -o colmap_dense_masked_with_init_and_skydome_dn_reg --use_mask --mask_path $ROOTPATH/ds01/data/masks_moveable --port 6661 --use_gt_depth --use_gt_normal

# orb rgb-d, depth reg
python 7_train_gaussians.py -r $ROOTPATH/ds01 --recon_name 28_rgbd_debug -o orb_rgbd_d_reg --use_mask --mask_path $ROOTPATH/ds01/data/masks_moveable --port 6666 --use_gt_depth --use_gt_normal


########
# ds02
########
# no reg
python 7_train_gaussians.py -r $ROOTPATH/ds02  --colmap_dir $ROOTPATH/ds02/poses/colmap -o colmap_baseline_masked --port 6661 --use_mask --mask_path $ROOTPATH/ds02/data/masks_moveable

python 7_train_gaussians.py -r $ROOTPATH/ds02 --recon_name 1_colmap_dense_cloud -o colmap_dense_masked --port 6661 --use_mask --mask_path $ROOTPATH/ds02/data/masks_moveable

python 7_train_gaussians.py -r $ROOTPATH/ds02 --recon_name 2_colmap_dense_cloud_with_init -o colmap_dense_masked_with_init --port 6661 --use_mask --mask_path $ROOTPATH/ds02/data/masks_moveable

python 7_train_gaussians.py -r $ROOTPATH/ds02 --recon_name 3_colmap_dense_cloud_with_skydome -o colmap_dense_masked_with_skydome --port 6661 --use_mask --mask_path $ROOTPATH/ds02/data/masks_moveable

python 7_train_gaussians.py -r $ROOTPATH/ds02 --recon_name 4_colmap_dense_cloud_with_init_and_skydome -o colmap_dense_masked_with_init_and_skydome --port 6661 --use_mask --mask_path $ROOTPATH/ds02/data/masks_moveable

python 7_train_gaussians.py -r $ROOTPATH/ds02 --recon_name 6_mono_dense_skydome -o mono_dense_masked_with_skydome --port 6661 --use_mask --mask_path $ROOTPATH/ds02/data/masks_moveable

# d reg
python 7_train_gaussians.py -r $ROOTPATH/ds02 --recon_name 4_colmap_dense_cloud_with_init_and_skydome -o colmap_dense_masked_with_init_and_skydome_d_reg --port 6661 --use_mask --mask_path $ROOTPATH/ds02/data/masks_moveable --use_gt_depth

python 7_train_gaussians.py -r $ROOTPATH/ds02 --recon_name 6_mono_dense_skydome -o mono_dense_masked_with_skydome_d_reg --port 6661 --use_mask --mask_path $ROOTPATH/ds02/data/masks_moveable --use_gt_depth

# n reg
python 7_train_gaussians.py -r $ROOTPATH/ds02 --recon_name 4_colmap_dense_cloud_with_init_and_skydome -o colmap_dense_masked_with_init_and_skydome_n_reg --port 6661 --use_mask --mask_path $ROOTPATH/ds02/data/masks_moveable --use_gt_normal

#dn reg
python 7_train_gaussians.py -r $ROOTPATH/ds02 --recon_name 4_colmap_dense_cloud_with_init_and_skydome -o colmap_dense_masked_with_init_and_skydome_dn_reg --port 6661 --use_mask --mask_path $ROOTPATH/ds02/data/masks_moveable --use_gt_depth --use_gt_normal

########
# muc02
########
# no reg
python 7_train_gaussians.py -r $ROOTPATH/muc02  --colmap_dir $ROOTPATH/muc02/poses/colmap -o colmap_baseline_masked --port 6661 --use_mask --mask_path $ROOTPATH/muc02/data/masks_moveable

python 7_train_gaussians.py -r $ROOTPATH/muc02 --recon_name 1_colmap_dense_cloud -o colmap_dense_masked --port 6661 --use_mask --mask_path $ROOTPATH/muc02/data/masks_moveable

python 7_train_gaussians.py -r $ROOTPATH/muc02 --recon_name 2_colmap_dense_cloud_with_init -o colmap_dense_masked_with_init --port 6661 --use_mask --mask_path $ROOTPATH/muc02/data/masks_moveable

python 7_train_gaussians.py -r $ROOTPATH/muc02 --recon_name 3_colmap_dense_cloud_with_skydome -o colmap_dense_masked_with_skydome --port 6661 --use_mask --mask_path $ROOTPATH/muc02/data/masks_moveable

python 7_train_gaussians.py -r $ROOTPATH/muc02 --recon_name 4_colmap_dense_cloud_with_init_and_skydome -o colmap_dense_masked_with_init_and_skydome --port 6661 --use_mask --mask_path $ROOTPATH/muc02/data/masks_moveable

python 7_train_gaussians.py -r $ROOTPATH/muc02 --recon_name 5_rgbd_dense_skydome -o rgbd_dense_masked_with_skydome --port 6661 --use_mask --mask_path $ROOTPATH/muc02/data/masks_moveable


########
# ostspange
########
# no reg
python 7_train_gaussians.py -r $ROOTPATH/ostspange  --colmap_dir $ROOTPATH/ostspange/poses/colmap -o colmap_baseline_masked --port 6661 --use_mask --mask_path $ROOTPATH/ostspange/data/masks_moveable

python 7_train_gaussians.py -r $ROOTPATH/ostspange --recon_name 1_colmap_dense_cloud -o colmap_dense_masked --port 6767 --use_mask --mask_path $ROOTPATH/ostspange/data/masks_moveable

python 7_train_gaussians.py -r $ROOTPATH/ostspange --recon_name 2_colmap_dense_cloud_with_init -o colmap_dense_masked_with_init --port 6767 --use_mask --mask_path $ROOTPATH/ostspange/data/masks_moveable

python 7_train_gaussians.py -r $ROOTPATH/ostspange --recon_name 3_colmap_dense_cloud_with_skydome -o colmap_dense_masked_with_skydome --port 6767 --use_mask --mask_path $ROOTPATH/ostspange/data/masks_moveable

python 7_train_gaussians.py -r $ROOTPATH/ostspange --recon_name 4_colmap_dense_cloud_with_init_and_skydome -o colmap_dense_masked_with_init_and_skydome --port 6767 --use_mask --mask_path $ROOTPATH/ostspange/data/masks_moveable

python 7_train_gaussians.py -r $ROOTPATH/ostspange --recon_name 6_rgbd_dense_skydome_sparser -o rgbd_dense_masked_skydome --port 6767 --use_mask --mask_path $ROOTPATH/ostspange/data/masks_moveable


########
# ottendichler
########
# no reg
python 7_train_gaussians.py -r $ROOTPATH/ottendichler  --colmap_dir $ROOTPATH/ottendichler/poses/colmap -o colmap_baseline_masked --port 6767 --use_mask --mask_path $ROOTPATH/ottendichler/data/masks_moveable

python 7_train_gaussians.py -r $ROOTPATH/ottendichler --recon_name 1_colmap_dense_cloud -o colmap_dense_masked --port 6767 --use_mask --mask_path $ROOTPATH/ottendichler/data/masks_moveable

python 7_train_gaussians.py -r $ROOTPATH/ottendichler --recon_name 2_colmap_dense_cloud_with_init -o colmap_dense_masked_with_init --port 6767 --use_mask --mask_path $ROOTPATH/ottendichler/data/masks_moveable

python 7_train_gaussians.py -r $ROOTPATH/ottendichler --recon_name 3_colmap_dense_cloud_with_skydome -o colmap_dense_masked_with_skydome --port 6767 --use_mask --mask_path $ROOTPATH/ottendichler/data/masks_moveable

python 7_train_gaussians.py -r $ROOTPATH/ottendichler --recon_name 4_colmap_dense_cloud_with_init_and_skydome -o colmap_dense_masked_with_init_and_skydome --port 6767 --use_mask --mask_path $ROOTPATH/ottendichler/data/masks_moveable

python 7_train_gaussians.py -r $ROOTPATH/ottendichler --recon_name 6_rgbd_dense_skydome_sparser -o rgbd_dense_masked_withskydome --port 6767 --use_mask --mask_path $ROOTPATH/ottendichler/data/masks_moveable

# d-reg
python 7_train_gaussians.py -r $ROOTPATH/ottendichler --recon_name 4_colmap_dense_cloud_with_init_and_skydome -o colmap_dense_masked_with_init_and_skydome_d_reg --port 6767 --use_mask --mask_path $ROOTPATH/ottendichler/data/masks_moveable --use_gt_depth

python 7_train_gaussians.py -r $ROOTPATH/ottendichler --recon_name 6_rgbd_dense_skydome_sparser -o rgbd_dense_masked_with_skydome_d_reg --port 6767 --use_mask --mask_path $ROOTPATH/ottendichler/data/masks_moveable --use_gt_depth

# n-reg
python 7_train_gaussians.py -r $ROOTPATH/ottendichler --recon_name 4_colmap_dense_cloud_with_init_and_skydome -o colmap_dense_masked_with_init_and_skydome_n_reg --port 6767 --use_mask --mask_path $ROOTPATH/ottendichler/data/masks_moveable --use_gt_normal

python 7_train_gaussians.py -r $ROOTPATH/ottendichler --recon_name 6_rgbd_dense_skydome_sparser -o rgbd_dense_masked_with_skydome_n_reg --port 6767 --use_mask --mask_path $ROOTPATH/ottendichler/data/masks_moveable --use_gt_normal

# dn-reg
python 7_train_gaussians.py -r $ROOTPATH/ottendichler --recon_name 4_colmap_dense_cloud_with_init_and_skydome -o colmap_dense_masked_with_init_and_skydome_dn_reg --port 6767 --use_mask --mask_path $ROOTPATH/ottendichler/data/masks_moveable --use_gt_depth --use_gt_normal

python 7_train_gaussians.py -r $ROOTPATH/ottendichler --recon_name 6_rgbd_dense_skydome_sparser -o rgbd_dense_masked_with_skydome_dn_reg --port 6767 --use_mask --mask_path $ROOTPATH/ottendichler/data/masks_moveable --use_gt_depth --use_gt_normal


########
# ds_combined
########
# no reg
python 7_train_gaussians.py -r $ROOTPATH/ds_combined --recon_name 25_colmap_dense_test -o colmap_dense_masked_with_init_and_skydome --port 6676 --use_mask --mask_path $ROOTPATH/ds_combined/data/masks_moveable --test_iterations 1000 5000 7000 10000 15000 20000 25000 30000 40000 50000 --save_iterations 7000 30000 50000

# depth reg
python 7_train_gaussians.py -r $ROOTPATH/ds_combined --recon_name 25_colmap_dense_test -o colmap_dense_masked_with_init_and_skydome_d_reg --port 6676  --use_mask --mask_path $ROOTPATH/ds_combined/data/masks_moveable --use_gt_depth --test_iterations 1000 5000 7000 10000 15000 20000 25000 30000 40000 50000 --save_iterations 7000 30000 50000

# normal reg
python 7_train_gaussians.py -r $ROOTPATH/ds_combined --recon_name 25_colmap_dense_test -o colmap_dense_masked_with_init_and_skydome_n_reg --port 6676  --use_mask --mask_path $ROOTPATH/ds_combined/data/masks_moveable --use_gt_normal --test_iterations 1000 5000 7000 10000 15000 20000 25000 30000 40000 50000 --save_iterations 7000 30000 50000

# depth and normal reg
python 7_train_gaussians.py -r $ROOTPATH/ds_combined --recon_name 25_colmap_dense_test -o colmap_dense_masked_with_init_and_skydome_dn_reg --port 6676  --use_mask --mask_path $ROOTPATH/ds_combined/data/masks_moveable --use_gt_depth --use_gt_normal --test_iterations 1000 5000 7000 10000 15000 20000 25000 30000 40000 50000 --save_iterations 7000 30000 50000

