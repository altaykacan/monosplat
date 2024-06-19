import os
import sys

import torch

from modules.io.datasets import CustomDataset, KITTIDataset, KITTI360Dataset, COLMAPDataset
from modules.depth.models import Metric3Dv2, PrecomputedDepthModel
from modules.core.backprojection import Backprojector
from modules.core.reconstructors import SimpleReconstructor


# KITTI sequence 7 test
# image_dir = "/usr/stud/kaa/thesis/Toolbox-Draft/data/kitti/images"
# pose_file = "/usr/stud/kaa/thesis/Toolbox-Draft/data/kitti/poses_07.txt"
# pose_scale = 1.0
# orig_intrinsics = [707.0912, 707.0912, 601.8873, 183.1104]
# orig_size = [370, 1226]
# target_size = [368, 1224]
# dataset = KITTIDataset(
#     image_dir,
#     pose_file,
#     pose_scale,
#     orig_intrinsics,
#     orig_size,
#     target_size,
#     start=0,
#     end=200,
#     )

# DS01 test
colmap_dir = "/usr/stud/kaa/data/root/ds01/poses/colmap"
pose_scale = 19.96
orig_intrinsics = [534.045, 534.045, 512, 288]
target_size = [576, 1024]
depth_dir = "/usr/stud/kaa/data/root/ds01/data/depths/arrays"
dataset = COLMAPDataset(colmap_dir, 1.0, orig_intrinsics, target_size=target_size, depth_dir=depth_dir,  depth_scale=1 / pose_scale)

# depth_model = Metric3Dv2(dataset.intrinsics, backbone="vit_giant")
depth_model = PrecomputedDepthModel(dataset)
backprojector = Backprojector(cfg={"dropout":0.99}, intrinsics=dataset.intrinsics)
recon = SimpleReconstructor(dataset, backprojector, depth_model, cfg={"batch_size": 4, "output_dir": "tests/test_results/debug"})
recon.run()

print("done!")


