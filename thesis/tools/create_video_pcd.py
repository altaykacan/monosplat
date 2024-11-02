"""
A quick script to create point cloud fly throughs by perturbing the original camera trajectory with sinusoidal displacements/rotations
"""
import argparse
from pathlib import Path

import torch
import open3d as o3d
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image

from configs.data import PADDED_IMG_NAME_LENGTH
from modules.core.utils import format_intrinsics
from modules.io.utils import read_ply, read_ply_o3d, save_image_torch
from modules.io.datasets import ColmapDataset, KITTI360Dataset, CustomDataset
from modules.scale_alignment.sparse import project_pcd_o3d
from modules.pose.utils import get_transformed_pose, get_new_trajectory
from modules.core.visualization import make_vid


# TODO implement proper argument parsing
def main(args):
    pass


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Process some paths and parameters.")
    # parser.add_argument('--root_dir', type=str, required=True, help='Root directory path')
    # parser.add_argument('--recon_path', type=str, required=True, help='Path to the reconstruction (.ply file) to be used')
    # parser.add_argument('--pose_path', type=str, required=True, help='Pose path')
    # parser.add_argument('--pose_scale', type=float, required=True, help='Pose scale, inverse of the depth scale')
    # parser.add_argument('--output_dir', type=str, required=True, help='Output directory path')
    # parser.add_argument('--seq_id', type=int, default=0, help='Sequence ID for KITTI360')
    # parser.add_argument('--cam_id', type=int, default=0, help='Camera ID for KITTI360')
    # parser.add_argument('--depth_max', default=30, type=float, required=True, help='Maximum depth')
    # parser.add_argument('--target_size', type=tuple, required=True, help='Target size')
    # parser.add_argument('--padded_img_name_length', default=PADDED_IMG_NAME_LENGTH, type=int, required=True, help='Padded image name length')
    # parser.add_argument('--scale_poses', action="store_true", help='Flag to scale poses instead of the depths')

    # args = parser.parse_args()
    # main(args)

    ###
    # Input
    ###
    dataset_name = "ottendichler"
    output_name = "ottendichler_mono_dense"

    if "ottendichler" in dataset_name:
        root_dir = Path("/usr/stud/kaa/data/root/ottendichler")
        # ply_path = Path("/usr/stud/kaa/data/root/ottendichler/reconstructions/9_colmap_dense_cloud_with_init_and_skydome_denser/cloud.ply")
        # ply_path = Path("/usr/stud/kaa/data/root/ottendichler/poses/colmap/sparse/0/points3D.ply")
        ply_path = Path("/usr/stud/kaa/data/root/ottendichler/reconstructions/8_mono_dense_skydome_denser/cloud.ply")
        colmap_dir = root_dir / "poses" / "colmap"
        image_dir = root_dir / "data" / "rgb"
        output_dir = Path("./pcd_projections") / root_dir.stem
        pose_path = Path("/usr/stud/kaa/data/root/ottendichler/poses/slam/1_mono_CameraTrajectory.txt")
        # pose_path = colmap_dir / "sparse" / "0" / "images.txt"
        pose_scale = 18.83 # only important to get max depth, not relevant otherwise
        depth_scale = 1 / pose_scale
        depth_max = 70 * depth_scale
        target_size = ()
        padded_img_name_length = 5

        dataset_type = "custom" # kitti360, colmap, or custom
        intrinsics = [525.75, 525.75, 512, 288]

    if "ds01" in dataset_name:
        root_dir = Path("/usr/stud/kaa/data/root/ds01")
        # ply_path = root_dir / "/usr/stud/kaa/data/root/ds01/reconstructions/31_colmap_dense_cloud_with_skydome_vid/cloud.ply"
        # ply_path = Path("/usr/stud/kaa/data/root/ds01/poses/colmap/sparse/0/points3D.ply")
        ply_path = Path("/usr/stud/kaa/data/root/ds01/reconstructions/30_rgbd_dense_skydome_vid/cloud.ply")
        colmap_dir = root_dir / "poses" / "colmap"
        image_dir = root_dir / "data" / "rgb"
        output_dir = Path("./pcd_projections") / root_dir.stem
        pose_path = Path("/usr/stud/kaa/data/root/ds01/reconstructions/30_rgbd_dense_skydome_vid/slam_poses.txt")
        # pose_path = colmap_dir / "sparse" / "0" / "images.txt"
        pose_scale = 1.0 # only important to get max depth, not relevant otherwise
        depth_scale = 1 / pose_scale
        depth_max = depth_scale * 70
        target_size = ()
        padded_img_name_length = 5

        dataset_type = "custom" # kitti360, colmap, or custom
        intrinsics =  [534.045,  534.045, 512, 288]

    if "kitti" in dataset_name:
        # TODO implement support for KITTI too
        seq_id = 0
        cam_id = 0
        pass

    start = 0
    end = -1

    # Rotation angles around axes
    angle_x_range = 5
    angle_y_range = 15
    angle_x_period = 400 # in terms of frames
    angle_y_period = 200

    # Displacement ranges as percentages of mean displacement in each direction between consecutive images
    disp_x_range = 0.5
    disp_y_range = -10
    disp_y_max = 0.2 # to prevent going below the road
    disp_x_period = 100 # in terms of frames
    disp_y_period = 200

    disp_delay=0

    fps=45

    ###
    # Loading and Rendering
    ###
    if dataset_type == "kitti360":
        dataset = KITTI360Dataset(seq_id, cam_id, pose_scale, target_size)
        pose_path = dataset.pose_path # GT poses
    elif dataset_type == "colmap":
        dataset = ColmapDataset(
        colmap_dir,
        pose_scale=1.0,
        target_size=target_size,
        orig_intrinsics=intrinsics,
        padded_img_name_length=padded_img_name_length,
        depth_scale=depth_scale,
        start=start,
        end=end,
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
            start=start,
            end=end,
            )

    H, W = dataset.H, dataset.W

    if ply_path.name == "points3D.txt":
        pcd = ColmapDataset.read_colmap_pcd_o3d(ply_path, convert_to_float32=True)
    else:
        pcd = read_ply_o3d(ply_path, convert_to_float32=True)

    poses = dataset.poses # is a list but we want a tensor
    poses = torch.stack(poses, dim=0)
    K = format_intrinsics(intrinsics) # [3, 3]

    # Mean distance between consecutive poses for each coordinate direction in world frame
    mean_delta = (poses[1:,:3, 3] - poses[0:-1, :3, 3]).mean(dim=0, keepdim=True)

    new_poses = get_new_trajectory(
        poses,
        angle_x_range=angle_x_range,
        angle_y_range=angle_y_range,
        angle_x_period=angle_x_period,
        angle_y_period=angle_y_period,
        disp_x_range=disp_x_range,
        disp_y_range=disp_y_range,
        disp_x_period=disp_x_period,
        disp_y_period=disp_y_period,
        disp_y_max=disp_y_max,
        mean_delta=mean_delta,
        disp_delay=disp_delay,
        )

    images = []
    for p in tqdm(new_poses):
        p_inv = torch.linalg.inv(p)

        rgb = project_pcd_o3d(
            pcd,
            W,
            H,
            K,
            p_inv,
            depth_max,
            get_rgb=True,
            )

        images.append(to_pil_image(rgb))
        save_image_torch(rgb)

    print("Creating a video...")
    if output_name != "":
        output_name += "_"
    make_vid(images, fps=fps, output_path=f"{output_name}pcd_video.mp4")

    print("Done!")








