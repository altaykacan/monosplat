#!/usr/bin/env bash
# main script to run scale estimation evaluation and validation on kitti360

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

###################
# kitti360_0_mini
###################

# sparse validation
python /usr/stud/kaa/thesis/DEN-Splatting/5_align_scale.py -r /usr/stud/kaa/data/root/kitti360_0_mini --intrinsics 552.55 552.55 682.05 238.77 --dataset kitti360 --alignment_type sparse --sparse_cloud_path /usr/stud/kaa/data/root/kitti360_0_mini/eval/gt_map.ply --debug --exp_name sparse_scale_eval_redo --kitti360_cam_id 0 --kitti360_seq_id 0 --start_id 500 --end_id 1500 --depth_type gt

# sparse validation, smaller max_d
python /usr/stud/kaa/thesis/DEN-Splatting/5_align_scale.py -r /usr/stud/kaa/data/root/kitti360_0_mini --intrinsics 552.55 552.55 682.05 238.77 --dataset kitti360 --alignment_type sparse --sparse_cloud_path /usr/stud/kaa/data/root/kitti360_0_mini/eval/gt_map.ply --debug --exp_name sparse_scale_eval_smaller_max_d --kitti360_cam_id 0 --kitti360_seq_id 0 --start_id 500 --end_id 1500 --max_d 5 --depth_type gt

# dense validation (dense)
python /usr/stud/kaa/thesis/DEN-Splatting/5_align_scale.py -r /usr/stud/kaa/data/root/kitti360_0_mini --intrinsics 552.55 552.55 682.05 238.77 --dataset kitti360 --alignment_type dense --exp_name dense_scale_eval --kitti360_cam_id 0 --kitti360_seq_id 0 --start_id 500 --end_id 1500 --depth_type precomputed --seg_model_type precomputed --flow_steps 1 2 4 6

# dense validation (sparse)
python /usr/stud/kaa/thesis/DEN-Splatting/5_align_scale.py -r /usr/stud/kaa/data/root/kitti360_0_mini --intrinsics 552.55 552.55 682.05 238.77 --dataset kitti360 --alignment_type sparse --sparse_cloud_path /usr/stud/kaa/data/root/kitti360_0_mini/eval/gt_map.ply --exp_name dense_scale_eval_sparse_ref --kitti360_cam_id 0 --kitti360_seq_id 0 --start_id 500 --end_id 1500 --depth_type precomputed

###################
# kitti360_3
###################
# sparse validation
python /usr/stud/kaa/thesis/DEN-Splatting/5_align_scale.py -r /usr/stud/kaa/data/root/kitti360_3 --intrinsics 552.55 552.55 682.05 238.77 --dataset kitti360 --alignment_type sparse --sparse_cloud_path /usr/stud/kaa/data/root/kitti360_3/eval/gt_map.ply --debug --exp_name sparse_scale_eval --kitti360_cam_id 0 --kitti360_seq_id 3 --start_id 300 --end_id 1030  --depth_type gt

# dense val (dense)
python /usr/stud/kaa/thesis/DEN-Splatting/5_align_scale.py -r /usr/stud/kaa/data/root/kitti360_3 --intrinsics 552.55 552.55 682.05 238.77 --dataset kitti360 --alignment_type dense --sparse_cloud_path /usr/stud/kaa/data/root/kitti360_3/eval/gt_map.ply --debug --exp_name dense_scale_eval --kitti360_cam_id 0 --kitti360_seq_id 3 --start_id 300 --end_id 1030 --flow_steps 1 2 4 6 --seg_model_type precomputed --depth_type precomputed

# dense val (sparse)
python /usr/stud/kaa/thesis/DEN-Splatting/5_align_scale.py -r /usr/stud/kaa/data/root/kitti360_3 --intrinsics 552.55 552.55 682.05 238.77 --dataset kitti360 --alignment_type sparse --sparse_cloud_path /usr/stud/kaa/data/root/kitti360_3/eval/gt_map.ply --debug --exp_name dense_scale_eval_sparse_ref --kitti360_cam_id 0 --kitti360_seq_id 3 --start_id 300 --end_id 1030  --seg_model_type precomputed --depth_type precomputed

###################
# kitti360_4_mini
###################
# sparse validation
python /usr/stud/kaa/thesis/DEN-Splatting/5_align_scale.py -r /usr/stud/kaa/data/root/kitti360_4_mini --intrinsics 552.55 552.55 682.05 238.77 --dataset kitti360 --alignment_type sparse --sparse_cloud_path /usr/stud/kaa/data/root/kitti360_4_mini/eval/gt_map.ply --debug --exp_name sparse_scale_eval --kitti360_cam_id 0 --kitti360_seq_id 4 --start_id 300 --end_id 1000  --depth_type gt

# dense val (dense)
python /usr/stud/kaa/thesis/DEN-Splatting/5_align_scale.py -r /usr/stud/kaa/data/root/kitti360_4_mini --intrinsics 552.55 552.55 682.05 238.77 --dataset kitti360 --alignment_type dense --sparse_cloud_path /usr/stud/kaa/data/root/kitti360_4_mini/eval/gt_map.ply --debug --exp_name dense_scale_eval --kitti360_cam_id 0 --kitti360_seq_id 4 --start_id 300 --end_id 1000 --flow_steps 1 2 4 6 --seg_model_type precomputed --depth_type precomputed

# dense val (sparse)
python /usr/stud/kaa/thesis/DEN-Splatting/5_align_scale.py -r /usr/stud/kaa/data/root/kitti360_4_mini --intrinsics 552.55 552.55 682.05 238.77 --dataset kitti360 --alignment_type sparse --sparse_cloud_path /usr/stud/kaa/data/root/kitti360_4_mini/eval/gt_map.ply --debug --exp_name dense_scale_eval_sparse_ref --kitti360_cam_id 0 --kitti360_seq_id 4 --start_id 300 --end_id 1000  --seg_model_type precomputed --depth_type precomputed


###################
# kitti360_6_mini
###################
# sparse validation
python /usr/stud/kaa/thesis/DEN-Splatting/5_align_scale.py -r /usr/stud/kaa/data/root/kitti360_6_mini --intrinsics 552.55 552.55 682.05 238.77 --dataset kitti360 --alignment_type sparse --sparse_cloud_path /usr/stud/kaa/data/root/kitti360_6_mini/eval/gt_map.ply --debug --exp_name sparse_scale_eval --kitti360_cam_id 0 --kitti360_seq_id 6 --start_id 300 --end_id 1000  --depth_type gt

python /usr/stud/kaa/thesis/DEN-Splatting/5_align_scale.py -r /usr/stud/kaa/data/root/kitti360_6_mini --intrinsics 552.55 552.55 682.05 238.77 --dataset kitti360 --alignment_type sparse --sparse_cloud_path /usr/stud/kaa/data/root/kitti360_6_mini/eval/gt_map.ply --debug --exp_name sparse_scale_eval_smaller_max_d --kitti360_cam_id 0 --kitti360_seq_id 6 --start_id 300 --end_id 1000  --depth_type gt --max_d 5


# dense val (dense)
python /usr/stud/kaa/thesis/DEN-Splatting/5_align_scale.py -r /usr/stud/kaa/data/root/kitti360_6_mini --intrinsics 552.55 552.55 682.05 238.77 --dataset kitti360 --alignment_type dense --sparse_cloud_path /usr/stud/kaa/data/root/kitti360_6_mini/eval/gt_map.ply --debug --exp_name dense_scale_eval --kitti360_cam_id 0 --kitti360_seq_id 6 --start_id 300 --end_id 1000 --flow_steps 1 2 4 6 --seg_model_type precomputed --depth_type precomputed

# dense val (sparse)
python /usr/stud/kaa/thesis/DEN-Splatting/5_align_scale.py -r /usr/stud/kaa/data/root/kitti360_6_mini --intrinsics 552.55 552.55 682.05 238.77 --dataset kitti360 --alignment_type sparse --sparse_cloud_path /usr/stud/kaa/data/root/kitti360_6_mini/eval/gt_map.ply --debug --exp_name dense_scale_eval_sparse_ref --kitti360_cam_id 0 --kitti360_seq_id 6 --start_id 300 --end_id 1000  --seg_model_type precomputed --depth_type precomputed
