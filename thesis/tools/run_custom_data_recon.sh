#!/usr/bin/env bash

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

########
# ds01
########

####
# Visual quality experiments
####
# colmap dense
python 6_create_pointcloud.py -r /usr/stud/kaa/data/root/ds01 --intrinsics 534.045 534.045 512 288 --reconstructor simple --backprojector simple --dataset colmap --max_d 50 --depth_model precomputed --seg_model segformer --pose_scale 1.0 --use_every_nth 5 --batch_size 2 --dropout 0.99 --recon_name colmap_dense_cloud --clean_pointcloud --depth_scale 0.0501002
# 1/ 19.96

# colmap dense with init cloud
python 6_create_pointcloud.py -r /usr/stud/kaa/data/root/ds01 --intrinsics 534.045 534.045 512 288 --reconstructor simple --backprojector simple --dataset colmap --max_d 50 --depth_model precomputed --seg_model segformer --pose_scale 1.0 --use_every_nth 5 --batch_size 2 --dropout 0.99 --recon_name colmap_dense_cloud_with_init --init_cloud_path /usr/stud/kaa/data/root/ds01/poses/colmap/sparse/0/points3D.ply --clean_pointcloud --depth_scale 0.0501002
# 1/ 19.96

# colmap dense with skydome
python 6_create_pointcloud.py -r /usr/stud/kaa/data/root/ds01 --intrinsics 534.045 534.045 512 288 --reconstructor simple --backprojector simple --dataset colmap --max_d 50 --depth_model precomputed --seg_model segformer --pose_scale 1.0 --use_every_nth 5 --batch_size 2 --dropout 0.99 --recon_name colmap_dense_cloud_with_skydome --add_skydome --clean_pointcloud --depth_scale 0.0501002
# 1/ 19.96

# colmap dense with init cloud and skydome
python 6_create_pointcloud.py -r /usr/stud/kaa/data/root/ds01 --intrinsics 534.045 534.045 512 288 --reconstructor simple --backprojector simple --dataset colmap --max_d 50 --depth_model precomputed --seg_model segformer --pose_scale 1.0 --use_every_nth 5 --batch_size 2 --dropout 0.99 --recon_name colmap_dense_cloud_with_init_and_skydome --init_cloud_path /usr/stud/kaa/data/root/ds01/poses/colmap/sparse/0/points3D.ply --add_skydome --clean_pointcloud --depth_scale 0.0501002
# 1/ 19.96

# colmap denser with downsampling
python 6_create_pointcloud.py -r /usr/stud/kaa/data/root/ds01 --intrinsics 534.045 534.045 512 288 --reconstructor simple --backprojector simple --dataset colmap --max_d 50 --depth_model precomputed --seg_model segformer --pose_scale 1.0 --use_every_nth 3 --batch_size 2 --dropout 0.90 --downsample_pointcloud_voxel_size 0.05 --recon_name colmap_denser_cloud_with_downsample --clean_pointcloud --depth_scale 0.0501002
# 1/ 19.96

# colmap denser with downsampling and init cloud
python 6_create_pointcloud.py -r /usr/stud/kaa/data/root/ds01 --intrinsics 534.045 534.045 512 288 --reconstructor simple --backprojector simple --dataset colmap --max_d 50 --depth_model precomputed --seg_model segformer --pose_scale 1.0 --use_every_nth 3 --batch_size 2 --dropout 0.90 --downsample_pointcloud_voxel_size 0.05 --init_cloud_path /usr/stud/kaa/data/root/ds01/poses/colmap/sparse/0/points3D.ply --recon_name colmap_denser_cloud_with_downsample_and_init --clean_pointcloud --depth_scale 0.0501002
# 1/ 19.96

# colmap denser with downsampling and init cloud and skydome
python 6_create_pointcloud.py -r /usr/stud/kaa/data/root/ds01 --intrinsics 534.045 534.045 512 288 --reconstructor simple --backprojector simple --dataset colmap --max_d 50 --depth_model precomputed --seg_model segformer --pose_scale 1.0 --use_every_nth 3 --batch_size 2 --dropout 0.90 --downsample_pointcloud_voxel_size 0.05 --init_cloud_path /usr/stud/kaa/data/root/ds01/poses/colmap/sparse/0/points3D.ply --add_skydome --recon_name colmap_denser_cloud_with_downsample_and_init_and_skydome --clean_pointcloud --depth_scale 0.0501002
# 1 / 19.96


# rgbd
python 6_create_pointcloud.py -r /usr/stud/kaa/data/root/ds01 --intrinsics 534.045 534.045 512 288 --pose_path /usr/stud/kaa/data/root/ds01/poses/slam/1_rgbd_CameraTrajectory.txt --reconstructor simple --backprojector depth_dropout --dataset custom --max_d 100 --depth_model precomputed --seg_model segformer --pose_scale 1.0 --use_every_nth 3 --batch_size 2 --dropout 0.98 --downsample_pointcloud_voxel_size 0.05 --add_skydome --recon_name rgbd_dense_skydome --clean_pointcloud --depth_scale 1.0

####
# For creating nice videos
####
python 6_create_pointcloud.py -r /usr/stud/kaa/data/root/ds01 --intrinsics 534.045 534.045 512 288 --reconstructor simple --backprojector simple --dataset colmap --max_d 40 --depth_model precomputed --seg_model segformer --pose_scale 1.0 --use_every_nth 3 --batch_size 2 --dropout 0.95 --recon_name colmap_dense_cloud_with_skydome_vid --add_skydome --clean_pointcloud --depth_scale 0.0501002 --downsample_pointcloud_voxel_size 0.05

python 6_create_pointcloud.py -r /usr/stud/kaa/data/root/ds01 --intrinsics 534.045 534.045 512 288 --pose_path /usr/stud/kaa/data/root/ds01/poses/slam/1_rgbd_CameraTrajectory.txt --reconstructor simple --backprojector simple --dataset custom --max_d 40 --depth_model precomputed --seg_model segformer --pose_scale 1.0 --use_every_nth 3 --batch_size 2 --dropout 0.95 --downsample_pointcloud_voxel_size 0.05 --add_skydome --recon_name rgbd_dense_skydome_vid --clean_pointcloud --depth_scale 1.0


####
# Downsample density experiments, with colmap poses
####
# voxel size 0.01
python 6_create_pointcloud.py -r /usr/stud/kaa/data/root/ds01 --intrinsics 534.045 534.045 512 288 --reconstructor simple --backprojector simple --dataset colmap --max_d 50 --depth_model precomputed --seg_model segformer --pose_scale 1.0 --use_every_nth 3 --batch_size 2 --dropout 0.90 --downsample_pointcloud_voxel_size 0.01 --init_cloud_path /usr/stud/kaa/data/root/ds01/poses/colmap/sparse/0/points3D.ply --add_skydome --recon_name colmap_denser_cloud_voxel_0_01 --clean_pointcloud --depth_scale 0.0501002

# voxel size 0.025
python 6_create_pointcloud.py -r /usr/stud/kaa/data/root/ds01 --intrinsics 534.045 534.045 512 288 --reconstructor simple --backprojector simple --dataset colmap --max_d 50 --depth_model precomputed --seg_model segformer --pose_scale 1.0 --use_every_nth 3 --batch_size 2 --dropout 0.90 --downsample_pointcloud_voxel_size 0.025 --init_cloud_path /usr/stud/kaa/data/root/ds01/poses/colmap/sparse/0/points3D.ply --add_skydome --recon_name colmap_denser_cloud_voxel_0_025 --clean_pointcloud --depth_scale 0.0501002

# voxel size 0.05
python 6_create_pointcloud.py -r /usr/stud/kaa/data/root/ds01 --intrinsics 534.045 534.045 512 288 --reconstructor simple --backprojector simple --dataset colmap --max_d 50 --depth_model precomputed --seg_model segformer --pose_scale 1.0 --use_every_nth 3 --batch_size 2 --dropout 0.90 --downsample_pointcloud_voxel_size 0.05 --init_cloud_path /usr/stud/kaa/data/root/ds01/poses/colmap/sparse/0/points3D.ply --add_skydome --recon_name colmap_denser_cloud_voxel_0_05 --clean_pointcloud --depth_scale 0.0501002

# voxel size 0.1
python 6_create_pointcloud.py -r /usr/stud/kaa/data/root/ds01 --intrinsics 534.045 534.045 512 288 --reconstructor simple --backprojector simple --dataset colmap --max_d 50 --depth_model precomputed --seg_model segformer --pose_scale 1.0 --use_every_nth 3 --batch_size 2 --dropout 0.90 --downsample_pointcloud_voxel_size 0.1 --init_cloud_path /usr/stud/kaa/data/root/ds01/poses/colmap/sparse/0/points3D.ply --add_skydome --recon_name colmap_denser_cloud_voxel_0_1 --clean_pointcloud --depth_scale 0.0501002

# voxel size 1.0
python 6_create_pointcloud.py -r /usr/stud/kaa/data/root/ds01 --intrinsics 534.045 534.045 512 288 --reconstructor simple --backprojector simple --dataset colmap --max_d 50 --depth_model precomputed --seg_model segformer --pose_scale 1.0 --use_every_nth 3 --batch_size 2 --dropout 0.90 --downsample_pointcloud_voxel_size 1.0 --init_cloud_path /usr/stud/kaa/data/root/ds01/poses/colmap/sparse/0/points3D.ply --add_skydome --recon_name colmap_denser_cloud_voxel_1 --clean_pointcloud --depth_scale 0.0501002

# voxel size 5.0
python 6_create_pointcloud.py -r /usr/stud/kaa/data/root/ds01 --intrinsics 534.045 534.045 512 288 --reconstructor simple --backprojector simple --dataset colmap --max_d 50 --depth_model precomputed --seg_model segformer --pose_scale 1.0 --use_every_nth 3 --batch_size 2 --dropout 0.90 --downsample_pointcloud_voxel_size 5.0 --init_cloud_path /usr/stud/kaa/data/root/ds01/poses/colmap/sparse/0/points3D.ply --add_skydome --recon_name colmap_denser_cloud_voxel_5 --clean_pointcloud --depth_scale 0.0501002

# voxel size 10
python 6_create_pointcloud.py -r /usr/stud/kaa/data/root/ds01 --intrinsics 534.045 534.045 512 288 --reconstructor simple --backprojector simple --dataset colmap --max_d 50 --depth_model precomputed --seg_model segformer --pose_scale 1.0 --use_every_nth 3 --batch_size 2 --dropout 0.90 --downsample_pointcloud_voxel_size 10 --init_cloud_path /usr/stud/kaa/data/root/ds01/poses/colmap/sparse/0/points3D.ply --add_skydome --recon_name colmap_denser_cloud_voxel_10 --clean_pointcloud --depth_scale 0.0501002

########
# ds02
########
# colmap dense
python 6_create_pointcloud.py -r /usr/stud/kaa/data/root/ds02 --intrinsics 534.045 534.045 512 288 --reconstructor simple --backprojector simple --dataset colmap --max_d 50 --depth_model precomputed --seg_model segformer --pose_scale 1.0 --use_every_nth 5 --batch_size 2 --dropout 0.99 --recon_name colmap_dense_cloud --clean_pointcloud --depth_scale 0.0461

# colmap dense with init cloud
python 6_create_pointcloud.py -r /usr/stud/kaa/data/root/ds02 --intrinsics 534.045 534.045 512 288 --reconstructor simple --backprojector simple --dataset colmap --max_d 50 --depth_model precomputed --seg_model segformer --pose_scale 1.0 --use_every_nth 5 --batch_size 2 --dropout 0.99 --recon_name colmap_dense_cloud_with_init --init_cloud_path /usr/stud/kaa/data/root/ds02/poses/colmap/sparse/0/points3D.ply --clean_pointcloud --depth_scale 0.0461

# colmap dense with skydome
python 6_create_pointcloud.py -r /usr/stud/kaa/data/root/ds02 --intrinsics 534.045 534.045 512 288 --reconstructor simple --backprojector simple --dataset colmap --max_d 50 --depth_model precomputed --seg_model segformer --pose_scale 1.0 --use_every_nth 5 --batch_size 2 --dropout 0.99 --recon_name colmap_dense_cloud_with_skydome --add_skydome --clean_pointcloud --depth_scale 0.0461

# colmap dense with init cloud and skydome
python 6_create_pointcloud.py -r /usr/stud/kaa/data/root/ds02 --intrinsics 534.045 534.045 512 288 --reconstructor simple --backprojector simple --dataset colmap --max_d 50 --depth_model precomputed --seg_model segformer --pose_scale 1.0 --use_every_nth 5 --batch_size 2 --dropout 0.99 --recon_name colmap_dense_cloud_with_init_and_skydome --init_cloud_path /usr/stud/kaa/data/root/ds02/poses/colmap/sparse/0/points3D.ply --add_skydome --clean_pointcloud --depth_scale 0.0461

# rgbd
python 6_create_pointcloud.py -r /usr/stud/kaa/data/root/ds02 --intrinsics 534.045 534.045 512 288 --pose_path /usr/stud/kaa/data/root/ds02/poses/slam/1_rgbd_CameraTrajectory.txt --reconstructor simple --backprojector depth_dropout --dataset custom --max_d 100 --depth_model precomputed --seg_model segformer --pose_scale 1.0 --use_every_nth 3 --batch_size 2 --dropout 0.98 --downsample_pointcloud_voxel_size 0.05 --add_skydome --recon_name rgbd_dense_skydome --clean_pointcloud --depth_scale 1.0

# mono
python 6_create_pointcloud.py -r /usr/stud/kaa/data/root/ds02 --intrinsics 534.045 534.045 512 288 --pose_path /usr/stud/kaa/data/root/ds02/poses/slam/1_mono_CameraTrajectory.txt --reconstructor simple --backprojector simple --dataset custom --max_d 50 --depth_model precomputed --seg_model segformer --pose_scale 1.0 --use_every_nth 3 --batch_size 2 --dropout 0.99 --downsample_pointcloud_voxel_size 0.05 --add_skydome --recon_name mono_dense_skydome --clean_pointcloud --depth_scale 0.0212

# mono, depth dropout
python 6_create_pointcloud.py -r /usr/stud/kaa/data/root/ds02 --intrinsics 534.045 534.045 512 288 --pose_path /usr/stud/kaa/data/root/ds02/poses/slam/1_mono_CameraTrajectory.txt --reconstructor simple --backprojector depth_dropout --dataset custom --max_d 90 --depth_model precomputed --seg_model segformer --pose_scale 1.0 --use_every_nth 3 --batch_size 2 --dropout 0.99 --downsample_pointcloud_voxel_size 0.1 --add_skydome --recon_name mono_dense_skydome_depth_dropout --clean_pointcloud --depth_scale 0.0212

########
# muc02
########
# colmap dense
python 6_create_pointcloud.py -r /usr/stud/kaa/data/root/muc02 --intrinsics 535.0 535.0 512 288 --reconstructor simple --backprojector simple --dataset colmap --max_d 20 --depth_model precomputed --seg_model segformer --pose_scale 1.0 --use_every_nth 5 --batch_size 2 --dropout 0.98 --recon_name colmap_dense_cloud --clean_pointcloud --depth_scale 0.0630 --downsample_pointcloud_voxel_size 0.05

# colmap dense with init cloud
python 6_create_pointcloud.py -r /usr/stud/kaa/data/root/muc02 --intrinsics 535.0 535.0 512 288 --reconstructor simple --backprojector simple --dataset colmap --max_d 20 --depth_model precomputed --seg_model segformer --pose_scale 1.0 --use_every_nth 5 --batch_size 2 --dropout 0.98 --recon_name colmap_dense_cloud_with_init --init_cloud_path /usr/stud/kaa/data/root/muc02/poses/colmap/sparse/0/points3D.ply --clean_pointcloud --depth_scale 0.0630 --downsample_pointcloud_voxel_size 0.05

# colmap dense with skydome
python 6_create_pointcloud.py -r /usr/stud/kaa/data/root/muc02 --intrinsics 535.0 535.0 512 288 --reconstructor simple --backprojector simple --dataset colmap --max_d 20 --depth_model precomputed --seg_model segformer --pose_scale 1.0 --use_every_nth 5 --batch_size 2 --dropout 0.98 --recon_name colmap_dense_cloud_with_skydome --add_skydome --clean_pointcloud --depth_scale 0.0630 --downsample_pointcloud_voxel_size 0.05

# colmap dense with init cloud and skydome
python 6_create_pointcloud.py -r /usr/stud/kaa/data/root/muc02 --intrinsics 535.0 535.0 512 288 --reconstructor simple --backprojector simple --dataset colmap --max_d 20 --depth_model precomputed --seg_model segformer --pose_scale 1.0 --use_every_nth 5 --batch_size 2 --dropout 0.98 --recon_name colmap_dense_cloud_with_init_and_skydome --init_cloud_path /usr/stud/kaa/data/root/muc02/poses/colmap/sparse/0/points3D.ply --add_skydome --clean_pointcloud --depth_scale 0.0630 --downsample_pointcloud_voxel_size 0.05

# rgbd
python 6_create_pointcloud.py -r /usr/stud/kaa/data/root/muc02 --intrinsics 535.0 535.0 512 288 --pose_path /usr/stud/kaa/data/root/muc02/poses/slam/1_rgbd_CameraTrajectory.txt --reconstructor simple --backprojector depth_dropout --dataset custom --max_d 40 --depth_model precomputed --seg_model segformer --pose_scale 1.0 --use_every_nth 3 --batch_size 2 --dropout 0.98 --downsample_pointcloud_voxel_size 0.05 --add_skydome --recon_name rgbd_dense_skydome --clean_pointcloud --depth_scale 1.0

########
# ostspange
########
# colmap dense
python 6_create_pointcloud.py -r /usr/stud/kaa/data/root/ostspange --intrinsics 533.72 533.72 512 288 --reconstructor simple --backprojector simple --dataset colmap --max_d 30 --depth_model precomputed --seg_model segformer --pose_scale 1.0 --use_every_nth 5 --batch_size 2 --dropout 0.98 --recon_name colmap_dense_cloud --clean_pointcloud --depth_scale 0.0666 --downsample_pointcloud_voxel_size 0.05

# colmap dense with init cloud
python 6_create_pointcloud.py -r /usr/stud/kaa/data/root/ostspange --intrinsics 533.72 533.72 512 288 --reconstructor simple --backprojector simple --dataset colmap --max_d 30 --depth_model precomputed --seg_model segformer --pose_scale 1.0 --use_every_nth 5 --batch_size 2 --dropout 0.98 --recon_name colmap_dense_cloud_with_init --init_cloud_path /usr/stud/kaa/data/root/ostspange/poses/colmap/sparse/0/points3D.ply --clean_pointcloud --depth_scale 0.0666 --downsample_pointcloud_voxel_size 0.05

# colmap dense with skydome
python 6_create_pointcloud.py -r /usr/stud/kaa/data/root/ostspange --intrinsics 533.72 533.72 512 288 --reconstructor simple --backprojector simple --dataset colmap --max_d 30 --depth_model precomputed --seg_model segformer --pose_scale 1.0 --use_every_nth 5 --batch_size 2 --dropout 0.98 --recon_name colmap_dense_cloud_with_skydome --add_skydome --clean_pointcloud --depth_scale 0.0666 --downsample_pointcloud_voxel_size 0.05

# colmap dense with init cloud and skydome
python 6_create_pointcloud.py -r /usr/stud/kaa/data/root/ostspange --intrinsics 533.72 533.72 512 288 --reconstructor simple --backprojector simple --dataset colmap --max_d 30 --depth_model precomputed --seg_model segformer --pose_scale 1.0 --use_every_nth 5 --batch_size 2 --dropout 0.98 --recon_name colmap_dense_cloud_with_init_and_skydome --init_cloud_path /usr/stud/kaa/data/root/ostspange/poses/colmap/sparse/0/points3D.ply --add_skydome --clean_pointcloud --depth_scale 0.0666 --downsample_pointcloud_voxel_size 0.05

# rgbd, too big file size, incrasing voxel size
# python 6_create_pointcloud.py -r /usr/stud/kaa/data/root/ostspange --intrinsics  533.72 533.72 512 288 --pose_path /usr/stud/kaa/data/root/ostspange/poses/slam/1_rgbd_CameraTrajectory.txt --reconstructor simple --backprojector depth_dropout --dataset custom --max_d 40 --depth_model precomputed --seg_model segformer --pose_scale 1.0 --use_every_nth 3 --batch_size 2 --dropout 0.98 --downsample_pointcloud_voxel_size 0.05 --add_skydome --recon_name rgbd_dense_skydome --clean_pointcloud --depth_scale 1.0
python 6_create_pointcloud.py -r /usr/stud/kaa/data/root/ostspange --intrinsics  533.72 533.72 512 288 --pose_path /usr/stud/kaa/data/root/ostspange/poses/slam/1_rgbd_CameraTrajectory.txt --reconstructor simple --backprojector depth_dropout --dataset custom --max_d 40 --depth_model precomputed --seg_model segformer --pose_scale 1.0 --use_every_nth 3 --batch_size 2 --dropout 0.98 --downsample_pointcloud_voxel_size 0.1 --add_skydome --recon_name rgbd_dense_skydome_sparser --clean_pointcloud --depth_scale 1.0

########
# ottendichler
########
# colmap dense
python 6_create_pointcloud.py -r /usr/stud/kaa/data/root/ottendichler --intrinsics 525.75 525.75 512 288 --reconstructor simple --backprojector simple --dataset colmap --max_d 30 --depth_model precomputed --seg_model segformer --pose_scale 1.0 --use_every_nth 5 --batch_size 2 --dropout 0.98 --recon_name colmap_dense_cloud --clean_pointcloud --depth_scale  0.0525 --downsample_pointcloud_voxel_size 0.05

# colmap dense with init cloud
python 6_create_pointcloud.py -r /usr/stud/kaa/data/root/ottendichler --intrinsics 525.75 525.75 512 288 --reconstructor simple --backprojector simple --dataset colmap --max_d 30 --depth_model precomputed --seg_model segformer --pose_scale 1.0 --use_every_nth 5 --batch_size 2 --dropout 0.98 --recon_name colmap_dense_cloud_with_init --init_cloud_path /usr/stud/kaa/data/root/ottendichler/poses/colmap/sparse/0/points3D.ply --clean_pointcloud --depth_scale 0.0525 --downsample_pointcloud_voxel_size 0.05

# colmap dense with skydome
python 6_create_pointcloud.py -r /usr/stud/kaa/data/root/ottendichler --intrinsics 525.75 525.75 512 288 --reconstructor simple --backprojector simple --dataset colmap --max_d 30 --depth_model precomputed --seg_model segformer --pose_scale 1.0 --use_every_nth 5 --batch_size 2 --dropout 0.98 --recon_name colmap_dense_cloud_with_skydome --add_skydome --clean_pointcloud --depth_scale 0.0525 --downsample_pointcloud_voxel_size 0.05

# colmap dense with init cloud and skydome
python 6_create_pointcloud.py -r /usr/stud/kaa/data/root/ottendichler --intrinsics 525.75 525.75 512 288 --reconstructor simple --backprojector simple --dataset colmap --max_d 30 --depth_model precomputed --seg_model segformer --pose_scale 1.0 --use_every_nth 5 --batch_size 2 --dropout 0.98 --recon_name colmap_dense_cloud_with_init_and_skydome --init_cloud_path /usr/stud/kaa/data/root/ottendichler/poses/colmap/sparse/0/points3D.ply --add_skydome --clean_pointcloud --depth_scale 0.0525 --downsample_pointcloud_voxel_size 0.05

# rgbd
# python 6_create_pointcloud.py -r /usr/stud/kaa/data/root/ottendichler --intrinsics 525.75 525.75 512 288 --pose_path /usr/stud/kaa/data/root/ottendichler/poses/slam/1_rgbd_CameraTrajectory.txt --reconstructor simple --backprojector depth_dropout --dataset custom --max_d 40 --depth_model precomputed --seg_model segformer --pose_scale 1.0 --use_every_nth 3 --batch_size 2 --dropout 0.98 --downsample_pointcloud_voxel_size 0.05 --add_skydome --recon_name rgbd_dense_skydome --clean_pointcloud --depth_scale 1.0
python 6_create_pointcloud.py -r /usr/stud/kaa/data/root/ottendichler --intrinsics 525.75 525.75 512 288 --pose_path /usr/stud/kaa/data/root/ottendichler/poses/slam/1_rgbd_CameraTrajectory.txt --reconstructor simple --backprojector depth_dropout --dataset custom --max_d 40 --depth_model precomputed --seg_model segformer --pose_scale 1.0 --use_every_nth 3 --batch_size 2 --dropout 0.98 --downsample_pointcloud_voxel_size 0.1 --add_skydome --recon_name rgbd_dense_skydome_sparser --clean_pointcloud --depth_scale 1.0

# mono
python 6_create_pointcloud.py -r /usr/stud/kaa/data/root/ottendichler --intrinsics 525.75 525.75 512 288 --pose_path /usr/stud/kaa/data/root/ottendichler/poses/slam/1_mono_CameraTrajectory.txt --reconstructor simple --backprojector depth_dropout --dataset custom --max_d 40 --depth_model precomputed --seg_model segformer --pose_scale 1.0 --use_every_nth 5 --batch_size 2 --dropout 0.98  --downsample_pointcloud_voxel_size 0.05 --add_skydome --recon_name mono_dense_skydome --clean_pointcloud --depth_scale 0.0531


# mono to compare to colmap, without depth-based dropout
python 6_create_pointcloud.py -r /usr/stud/kaa/data/root/ottendichler --intrinsics 525.75 525.75 512 288 --pose_path /usr/stud/kaa/data/root/ottendichler/poses/slam/1_mono_CameraTrajectory.txt --reconstructor simple --backprojector simple --dataset custom --max_d 30 --depth_model precomputed --seg_model segformer --pose_scale 1.0 --use_every_nth 5 --batch_size 2 --dropout 0.95  --downsample_pointcloud_voxel_size 0.05 --add_skydome --recon_name mono_dense_skydome_denser --clean_pointcloud --depth_scale 0.0531

# colmap dense to compare to mono, dense
python 6_create_pointcloud.py -r /usr/stud/kaa/data/root/ottendichler --intrinsics 525.75 525.75 512 288 --reconstructor simple --backprojector simple --dataset colmap --max_d 30 --depth_model precomputed --seg_model segformer --pose_scale 1.0 --use_every_nth 5 --batch_size 2 --dropout 0.95 --recon_name colmap_dense_cloud_with_init_and_skydome_denser --init_cloud_path /usr/stud/kaa/data/root/ottendichler/poses/colmap/sparse/0/points3D.ply --add_skydome --clean_pointcloud --depth_scale 0.0525 --downsample_pointcloud_voxel_size 0.05