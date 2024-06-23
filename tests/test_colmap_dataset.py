from pathlib import Path

from modules.io.datasets import ColmapDataset

colmap_dir = Path("/usr/stud/kaa/data/root/ds01/poses/colmap")
orig_intrinsics = (534.045, 534.045, 512, 288)
orig_size = (1024, 576)
pose_scale = 1.0

dataset = ColmapDataset(colmap_dir, pose_scale, orig_intrinsics, orig_size)
print("Number of poses: ", len(dataset.poses))
print("Number of frames: ", len(dataset.frame_ids))