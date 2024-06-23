import os
import sys

import torch

from modules.io.datasets import CustomDataset, KITTIDataset, KITTI360Dataset, ColmapDataset
from modules.depth.models import Metric3Dv2, PrecomputedDepthModel
from modules.segmentation.models import SegFormer, MaskRCNN
from modules.core.models import RAFT
from modules.core.backprojection import Backprojector
from modules.core.reconstructors import SimpleReconstructor, MovingObjectRemoverReconstructor


# KITTI sequence 7 test
image_dir = "/usr/stud/kaa/thesis/Toolbox-Draft/data/kitti/images"
pose_file = "/usr/stud/kaa/thesis/Toolbox-Draft/data/kitti/poses_07.txt"
pose_scale = 1.0
orig_intrinsics = [707.0912, 707.0912, 601.8873, 183.1104]
orig_size = [370, 1226]
target_size = [368, 1224]
dataset = KITTIDataset(
    image_dir,
    pose_file,
    pose_scale,
    orig_intrinsics,
    orig_size,
    target_size,
    depth_scale=1.0,
    start=500,
    end=-1,
    )

# DS01 test
# colmap_dir = "/usr/stud/kaa/data/root/ds01/poses/colmap"
# pose_scale = 1.0
# orig_intrinsics = [2143.78, 2143.78, 512, 288]
# target_size = [576, 1024]
# depth_dir = "/usr/stud/kaa/data/root/ds01/data/depths/arrays"
# dataset = ColmapDataset(colmap_dir, pose_scale, orig_intrinsics, target_size=target_size, depth_dir=depth_dir)

depth_model = Metric3Dv2(dataset.intrinsics, backbone="vit_giant")
# depth_model = PrecomputedDepthModel(dataset, depth_scale=1/30)
raft = RAFT()
segformer = SegFormer()
mask_rcnn = MaskRCNN()
backprojector = Backprojector(cfg={"dropout":0.99}, intrinsics=dataset.intrinsics)
cfg = {"batch_size": 5, "output_dir": "tests/test_results/debug", "log_every_nth_batch": 5}
recon = MovingObjectRemoverReconstructor(dataset, backprojector, depth_model, raft, segformer, mask_rcnn, cfg)
recon.run()

print("done!")


