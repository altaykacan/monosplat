#!/usr/bin/env bash

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

###################
# kitti360_0_mini
###################

# sparse validation
python /usr/stud/kaa/thesis/DEN-Splatting/5_align_scale.py -r /usr/stud/kaa/data/root/kitti360_0_mini --intrinsics 552.55 552.55 682.05 238.77 --dataset kitti360 --alignment_type sparse --sparse_cloud_path /usr/stud/kaa/data/root/kitti360_0_mini/eval/gt_map.ply --debug --exp_name sparse_debug_sparse_val_300_500 --kitti360_cam_id 0 --kitti360_seq_id 0 --start_id 300 --end_id 500

python /usr/stud/kaa/thesis/DEN-Splatting/5_align_scale.py -r /usr/stud/kaa/data/root/kitti360_0_mini --intrinsics 552.55 552.55 682.05 238.77 --dataset kitti360 --alignment_type sparse --sparse_cloud_path /usr/stud/kaa/data/root/kitti360_0_mini/eval/gt_map.ply --debug --exp_name sparse_debug_sparse_val_800_1100 --kitti360_cam_id 0 --kitti360_seq_id 0 --start_id 800 --end_id 1100

python /usr/stud/kaa/thesis/DEN-Splatting/5_align_scale.py -r /usr/stud/kaa/data/root/kitti360_0_mini --intrinsics 552.55 552.55 682.05 238.77 --dataset kitti360 --alignment_type sparse --sparse_cloud_path /usr/stud/kaa/data/root/kitti360_0_mini/eval/gt_map.ply --debug --exp_name sparse_debug_sparse_val_full --kitti360_cam_id 0 --kitti360_seq_id 0 --start_id 300 --end_id 1500

# dense validation (dense)
python /usr/stud/kaa/thesis/DEN-Splatting/5_align_scale.py -r /usr/stud/kaa/data/root/kitti360_0_mini --intrinsics 552.55 552.55 682.05 238.77 --dataset kitti360 --alignment_type dense --exp_name dense_debug_dense_val_300_500 --kitti360_cam_id 0 --kitti360_seq_id 0 --start_id 300 --end_id 500 --depth_type precomputed --seg_model_type precomputed --flow_steps 1 2 4 6

python /usr/stud/kaa/thesis/DEN-Splatting/5_align_scale.py -r /usr/stud/kaa/data/root/kitti360_0_mini --intrinsics 552.55 552.55 682.05 238.77 --dataset kitti360 --alignment_type dense --exp_name dense_debug_dense_val_800_1100 --kitti360_cam_id 0 --kitti360_seq_id 0 --start_id 800 --end_id 1100 --depth_type precomputed --seg_model_type precomputed --flow_steps 1 2 4 6

python /usr/stud/kaa/thesis/DEN-Splatting/5_align_scale.py -r /usr/stud/kaa/data/root/kitti360_0_mini --intrinsics 552.55 552.55 682.05 238.77 --dataset kitti360 --alignment_type dense --exp_name dense_debug_dense_val_full --kitti360_cam_id 0 --kitti360_seq_id 0 --start_id 500 --end_id 1500 --depth_type precomputed --seg_model_type precomputed --flow_steps 1 2 4 6

# dense validation (sparse)
python /usr/stud/kaa/thesis/DEN-Splatting/5_align_scale.py -r /usr/stud/kaa/data/root/kitti360_0_mini --intrinsics 552.55 552.55 682.05 238.77 --dataset kitti360 --alignment_type sparse --sparse_cloud_path /usr/stud/kaa/data/root/kitti360_0_mini/eval/gt_map.ply --exp_name sparse_debug_dense_val_300_to_500 --kitti360_cam_id 0 --kitti360_seq_id 0 --start_id 300 --end_id 500 --depth_type precomputed

python /usr/stud/kaa/thesis/DEN-Splatting/5_align_scale.py -r /usr/stud/kaa/data/root/kitti360_0_mini --intrinsics 552.55 552.55 682.05 238.77 --dataset kitti360 --alignment_type sparse --sparse_cloud_path /usr/stud/kaa/data/root/kitti360_0_mini/eval/gt_map.ply --exp_name sparse_debug_dense_val_800_to_1100 --kitti360_cam_id 0 --kitti360_seq_id 0 --start_id 800 --end_id 1100 --depth_type precomputed

python /usr/stud/kaa/thesis/DEN-Splatting/5_align_scale.py -r /usr/stud/kaa/data/root/kitti360_0_mini --intrinsics 552.55 552.55 682.05 238.77 --dataset kitti360 --alignment_type sparse --sparse_cloud_path /usr/stud/kaa/data/root/kitti360_0_mini/eval/gt_map.ply --exp_name sparse_debug_dense_val_full --kitti360_cam_id 0 --kitti360_seq_id 0 --start_id 500 --end_id 1500 --depth_type precomputed



###################
# kitti360_4_mini
###################

# dense validation (dense)
python /usr/stud/kaa/thesis/DEN-Splatting/5_align_scale.py -r /usr/stud/kaa/data/root/kitti360_4_mini --intrinsics 552.55 552.55 682.05 238.77 --dataset kitti360 --alignment_type dense --exp_name dense_debug_dense_val_300_500 --kitti360_cam_id 0 --kitti360_seq_id 4 --start_id 300 --end_id 500 --depth_type precomputed --seg_model_type precomputed --flow_steps 1 2 4 6

python /usr/stud/kaa/thesis/DEN-Splatting/5_align_scale.py -r /usr/stud/kaa/data/root/kitti360_0_mini --intrinsics 552.55 552.55 682.05 238.77 --dataset kitti360 --alignment_type dense --exp_name dense_debug_dense_val_800_1100 --kitti360_cam_id 0 --kitti360_seq_id 4 --start_id 800 --end_id 1100 --depth_type precomputed --seg_model_type precomputed --flow_steps 1 2 4 6

python /usr/stud/kaa/thesis/DEN-Splatting/5_align_scale.py -r /usr/stud/kaa/data/root/kitti360_4_mini --intrinsics 552.55 552.55 682.05 238.77 --dataset kitti360 --alignment_type dense --exp_name dense_debug_dense_val_full --kitti360_cam_id 0 --kitti360_seq_id 4 --start_id 500 --end_id 1000 --depth_type precomputed --seg_model_type precomputed --flow_steps 1 2 4 6

# dense validation (sparse)
python /usr/stud/kaa/thesis/DEN-Splatting/5_align_scale.py -r /usr/stud/kaa/data/root/kitti360_4_mini --intrinsics 552.55 552.55 682.05 238.77 --dataset kitti360 --alignment_type sparse --sparse_cloud_path /usr/stud/kaa/data/root/kitti360_0_mini/eval/gt_map.ply --exp_name sparse_debug_dense_val_300_to_500 --kitti360_cam_id 0 --kitti360_seq_id 4 --start_id 300 --end_id 500 --depth_type precomputed

python /usr/stud/kaa/thesis/DEN-Splatting/5_align_scale.py -r /usr/stud/kaa/data/root/kitti360_0_mini --intrinsics 552.55 552.55 682.05 238.77 --dataset kitti360 --alignment_type sparse --sparse_cloud_path /usr/stud/kaa/data/root/kitti360_0_mini/eval/gt_map.ply --exp_name sparse_debug_dense_val_800_to_1100 --kitti360_cam_id 0 --kitti360_seq_id 4 --start_id 800 --end_id 1100 --depth_type precomputed

python /usr/stud/kaa/thesis/DEN-Splatting/5_align_scale.py -r /usr/stud/kaa/data/root/kitti360_4_mini --intrinsics 552.55 552.55 682.05 238.77 --dataset kitti360 --alignment_type sparse --sparse_cloud_path /usr/stud/kaa/data/root/kitti360_4_mini/eval/gt_map.ply --exp_name sparse_debug_dense_val_full --kitti360_cam_id 0 --kitti360_seq_id 4 --start_id 300 --end_id 1000 --depth_type precomputed
