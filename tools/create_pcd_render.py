from pathlib import Path

import torch
import open3d as o3d

from modules.core.utils import format_intrinsics
from modules.io.utils import read_ply, read_ply_o3d, save_image_torch
from modules.io.datasets import ColmapDataset, KITTI360Dataset, CustomDataset
from modules.scale_alignment.sparse import project_pcd_o3d


def main(args):
    pass


if __name__ == "__main__":
    # TODO add proper argument parsing here

    root_dir = Path("/usr/stud/kaa/data/root/kitti360_0_mini")
    ply_path = Path("/usr/stud/kaa/data/root/kitti360_0_mini/reconstructions/4_colmap_sparse_scale/cloud.ply")
    pose_scale = 41.6
    colmap_dir = root_dir / "poses" / "colmap"
    image_dir = root_dir / "data" / "rgb"
    output_dir = Path("./pcd_projections")
    pose_ids = [500, 800]
    seq_id = 0
    cam_id = 0
    # pose_path = Path("")
    pose_path = None
    depth_max = 30
    target_size = ()
    padded_img_name_length = 10

    # pose rotation
    angle = 15
    radian = angle * torch.pi / 180
    R = torch.tensor([[torch.cos(radian), -torch.sin(radian), 0.0],
                      [torch.sin(radian),  torch.cos(radian), 0.0],
                      [0.0,                0.0,               1.0]])
    #TODO figure out what we have to add here to disturb the view

    dataset_type = "colmap" # kitti360, colmap, or custom
    intrinsics = [552.55, 552.55, 682.05, 238.77]

    if dataset_type == "kitti360":
        dataset = KITTI360Dataset(seq_id, cam_id, pose_scale, target_size)
        pose_path = dataset.pose_path # GT poses
    elif dataset_type == "colmap":
        dataset = ColmapDataset(
        colmap_dir,
        pose_scale=pose_scale,
        target_size=target_size,
        orig_intrinsics=intrinsics,
        padded_img_name_length=padded_img_name_length,
        )
        if pose_path is not None:
            dataset.pose_path = pose_path
    elif dataset_type == "custom":
        dataset = CustomDataset(
            image_dir,
            pose_path,
            pose_scale=pose_scale,
            target_size=target_size,
            orig_intrinsics=intrinsics,
            padded_img_name_length=padded_img_name_length,
            )

    H, W = dataset.H, dataset.W

    if ply_path.name == "points3D.txt":
        pcd = ColmapDataset.read_colmap_pcd_o3d(ply_path, convert_to_float32=True)
    else:
        pcd = read_ply_o3d(ply_path, convert_to_float32=True)

    _, _, poses = dataset.get_by_frame_ids(pose_ids)
    K = format_intrinsics(intrinsics) # [3, 3]

    images = []
    for pose in poses:
        pose_inv = torch.linalg.inv(pose)

        rgb = project_pcd_o3d(
            pcd,
            W,
            H,
            K,
            pose_inv,
            depth_max,
            get_rgb=True,
            )

        images.append(rgb)

    for i, image in enumerate(images):
        save_image_torch(image, f"projection_{i}", output_dir=output_dir)

    print("Done!")








