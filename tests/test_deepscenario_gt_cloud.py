from pathlib import Path

import torch
import numpy as np
import open3d as o3d

from modules.core.maps import PointCloud
from modules.io.utils import read_deepscenario_sparse_cloud

filepath = Path("/usr/stud/kaa/data/deepscenario/reconstruction.json")

xyz, rgb = read_deepscenario_sparse_cloud(filepath)
pcd = PointCloud()
pcd.increment(xyz, rgb)
pcd.postprocess()
pcd.save(filename="map.ply", output_dir="./tests/test_results")
pcd.clean()
pcd.save(filename="cleaned_map.ply", output_dir="./tests/test_results")

print("xyz shape: ", xyz.shape)
print("rgb shape: ", rgb.shape)