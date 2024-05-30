import torch

from modules.io.datasets import CustomDataset, KITTIDataset
from modules.depth.models import Metric3Dv2
from modules.core.models import Backprojector
from modules.core.reconstructors import SimpleReconstructor


image_dir = "/usr/stud/kaa/thesis/Toolbox-Draft/data/kitti/images"
pose_file = "/usr/stud/kaa/thesis/Toolbox-Draft/data/kitti/poses_07.txt"
pose_scale = 1.0
orig_intrinsics = [707.0912, 707.0912, 601.8873, 183.1104]
orig_size = [370, 1226]
target_size = [368, 1224]

dataset = KITTIDataset(image_dir, pose_file, pose_scale, orig_intrinsics, orig_size, target_size, start=10, end=30)
depth_model = Metric3Dv2(dataset.intrinsics, backbone="vit_small")
backprojector = Backprojector(cfg={}, intrinsics=dataset.intrinsics)

recon = SimpleReconstructor(dataset, backprojector, depth_model, cfg={"batch_size": 2})
recon.run()

print("done!")


