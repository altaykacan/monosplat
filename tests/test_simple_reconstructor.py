import torch

from modules.io.datasets import CustomDataset, KITTIDataset, KITTI360Dataset
from modules.depth.models import Metric3Dv2
from modules.core.models import Backprojector
from modules.core.reconstructors import SimpleReconstructor


image_dir = "/usr/stud/kaa/thesis/Toolbox-Draft/data/kitti/images"
pose_file = "/usr/stud/kaa/thesis/Toolbox-Draft/data/kitti/poses_07.txt"
pose_scale = 1.0
orig_intrinsics = [707.0912, 707.0912, 601.8873, 183.1104]
orig_size = [370, 1226]
target_size = [368, 1224]

dataset = KITTIDataset(image_dir, pose_file, pose_scale, orig_intrinsics, orig_size, target_size, start=0, end=200)
depth_model = Metric3Dv2(dataset.intrinsics, backbone="vit_giant")
backprojector = Backprojector(cfg={"dropout":0.99}, intrinsics=dataset.intrinsics)
recon = SimpleReconstructor(dataset, backprojector, depth_model, cfg={"batch_size": 2, "output_dir": "debug"})
recon.run()

print("done!")


