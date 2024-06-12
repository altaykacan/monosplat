from pathlib import Path

from modules.io.datasets import COLMAPDataset

colmap_dir = Path("/usr/stud/kaa/data/splats/custom/munich_ottendichler_colmap/")
orig_intrinsics = (530.855, 530.855, 512, 288)
orig_size = (1024, 576)
pose_scale = 1.0

dataset = COLMAPDataset(colmap_dir, pose_scale, orig_intrinsics, orig_size)
print("Number of poses: ", len(dataset.poses))
print("Number of frames: ", len(dataset.frame_ids))