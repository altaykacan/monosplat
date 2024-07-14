import os
import sys

import torch

from modules.io.datasets import CustomDataset, KITTIDataset, KITTI360Dataset
from modules.depth.models import Metric3Dv2, KITTI360DepthModel
from modules.core.backprojection import Backprojector
from modules.core.reconstructors import SimpleReconstructor


pose_scale = 1.0

dataset = KITTI360Dataset(seq=3, cam_id=0, end=300)
depth_model = Metric3Dv2(dataset.intrinsics, backbone="vit_giant")
# depth_model = KITTI360DepthModel(dataset)
backprojector = Backprojector(cfg={"dropout":0.00, "max_d": 50}, intrinsics=dataset.intrinsics)
recon = SimpleReconstructor(dataset, backprojector, depth_model, cfg={"batch_size": 2, "output_dir": "tests/test_results/debug"})
recon.run()

print("done!")


