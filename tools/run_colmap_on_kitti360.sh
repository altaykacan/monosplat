#!/usr/bin/env bash

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python 2_run_colmap.py -s /usr/stud/kaa/data/root/kitti360_0/poses/colmap --camera SIMPLE_PINHOLE --init_intrinsics 552.55 552.55 682.05 238.77 --fix_intrinsics --use_sequential_matcher --vocab_tree_path /usr/stud/kaa/data/colmap/vocab_tree_flickr100K_words256K.bin --no_gpu

python 2_run_colmap.py -s /usr/stud/kaa/data/root/kitti360_4/poses/colmap --camera SIMPLE_PINHOLE --init_intrinsics 552.55 552.55 682.05 238.77 --fix_intrinsics --use_sequential_matcher --vocab_tree_path /usr/stud/kaa/data/colmap/vocab_tree_flickr100K_words256K.bin --no_gpu

python 2_run_colmap.py -s /usr/stud/kaa/data/root/kitti360_3/poses/colmap --camera SIMPLE_PINHOLE --init_intrinsics 552.55 552.55 682.05 238.77 --fix_intrinsics --use_sequential_matcher --vocab_tree_path /usr/stud/kaa/data/colmap/vocab_tree_flickr100K_words256K.bin --no_gpu

python 2_run_colmap.py -s /usr/stud/kaa/data/root/kitti360_6/poses/colmap --camera SIMPLE_PINHOLE --init_intrinsics 552.55 552.55 682.05 238.77 --fix_intrinsics --use_sequential_matcher --vocab_tree_path /usr/stud/kaa/data/colmap/vocab_tree_flickr100K_words256K.bin --no_gpu

python 2_run_colmap.py -s /usr/stud/kaa/data/root/kitti360_0_mini/poses/colmap --camera SIMPLE_PINHOLE --init_intrinsics 552.55 552.55 682.05 238.77 --fix_intrinsics --no_gpu

python 2_run_colmap.py -s /usr/stud/kaa/data/root/kitti360_4_mini/poses/colmap --camera SIMPLE_PINHOLE --init_intrinsics 552.55 552.55 682.05 238.77 --fix_intrinsics --no_gpu

python 2_run_colmap.py -s /usr/stud/kaa/data/root/kitti360_6_mini/poses/colmap --camera SIMPLE_PINHOLE --init_intrinsics 552.55 552.55 682.05 238.77 --fix_intrinsics --no_gpu

python 2_run_colmap.py -s /usr/stud/kaa/data/root/kitti360_3/poses/colmap --camera SIMPLE_PINHOLE --init_intrinsics 552.55 552.55 682.05 238.77 --fix_intrinsics --no_gpu