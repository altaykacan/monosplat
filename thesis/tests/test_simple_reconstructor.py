import os
import sys

import torch

from modules.io.datasets import CustomDataset, KITTIDataset, KITTI360Dataset, ColmapDataset, CombinedColmapDataset
from modules.depth.models import Metric3Dv2, PrecomputedDepthModel
from modules.core.backprojection import Backprojector
from modules.core.reconstructors import SimpleReconstructor
from modules.core.models import PrecomputedNormalModel
from modules.segmentation.models import PrecomputedSegModel, SegFormer

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
# colmap_dir = "/usr/stud/kaa/data/root/ds01/poses/colmap"
# pose_scale = 19.96
# orig_intrinsics = [534.045, 534.045, 512, 288]
# target_size = [576, 1024]
# init_cloud_path = "/usr/stud/kaa/data/root/ds01/poses/colmap/sparse/0/points3D.ply"
# depth_dir = "/usr/stud/kaa/data/root/ds01/data/depths/arrays"
# dataset = ColmapDataset(colmap_dir, 1.0, orig_intrinsics, target_size=target_size, depth_dir=depth_dir,  depth_scale=1/pose_scale, end=-1)

# ottendichler
colmap_dir = "/usr/stud/kaa/data/root/ottendichler/poses/colmap"
pose_scale = 19.047
orig_intrinsics = [525.75, 525.75, 512, 288]
target_size = [576, 1024]
init_cloud_path = "/usr/stud/kaa/data/root/ottendichler/poses/colmap/sparse/0/points3D.ply"
depth_dir = "/usr/stud/kaa/data/root/ottendichler/data/depths/arrays"
dataset = ColmapDataset(colmap_dir, 1.0, orig_intrinsics, target_size=target_size, depth_dir=depth_dir,  depth_scale=1/pose_scale, end=-1)

# DS combined
# colmap_dir =  None
# dataset = CombinedColmapDataset()

# depth_model = Metric3Dv2(dataset.intrinsics, backbone="vit_giant")
depth_model = PrecomputedDepthModel(dataset)
normal_model = PrecomputedNormalModel(dataset)
seg_model = SegFormer()
backprojector = Backprojector(cfg={"dropout":0.99, "max_d": 30}, intrinsics=dataset.intrinsics)
recon = SimpleReconstructor(
    dataset,
    backprojector,
    depth_model,
    normal_model,
    seg_model,
    cfg={"batch_size": 4,
         "output_dir": "tests/test_results/dna",
         "use_every_nth":2,
         "add_skydome": True,
         "add_init_cloud":True,
         "init_cloud_path": init_cloud_path,
         "downsample_pointcloud_voxel_size": 0.1,
         "clean_pointcloud": True,
         })
recon.run()

print("done!")


