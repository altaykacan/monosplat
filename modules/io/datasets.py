"""Contains dataset implementations that store data related information and can be iterated over"""
import logging
from pathlib import Path
from typing import Union, Tuple, List

import torch
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial.transform import Rotation
import torchvision.transforms.functional as tv_F

from configs.data import KITTI360_DIR, PADDED_IMG_NAME_LENGTH
from modules.eval.kitti360 import projectVeloToImage
from modules.core.utils import compute_target_intrinsics
from modules.core.interfaces import BaseDataset

log = logging.getLogger(__name__)


class CustomDataset(BaseDataset):
    """
    Default CustomDataset implementation. Expects poses to be saved in TUM RGB-D
    dataset format unless `parse_pose()` is overriden.
    """
    def __init__(
            self,
            image_dir: Union[Path, str],
            pose_path: Union[Path, str],
            pose_scale: float,
            orig_intrinsics: Tuple,
            orig_size: Tuple,
            target_size: Tuple = (),
            gt_depth_dir: Union[Path, str] = None,
            depth_dir: Union[Path, str] = None,
            start: int = 0,
            end: int = -1,
            device: str = None
            ) -> None:
        self.image_dir = image_dir if isinstance(image_dir, Path) else Path(image_dir)
        self.pose_path = pose_path if isinstance(pose_path, Path) else Path(pose_path)

        if gt_depth_dir is None:
            self.gt_depth_dir = None
            self.has_gt_depth = False
        else:
            self.gt_depth_dir = gt_depth_dir if isinstance(gt_depth_dir, Path) else Path(gt_depth_dir)
            self.has_gt_depth = True

        if depth_dir is None:
            self.depth_dir = None
            self.has_depth = False
        else:
            self.depth_dir = gt_depth_dir if isinstance(gt_depth_dir, Path) else Path(gt_depth_dir)
            self.has_depth = True

        self.pose_scale = pose_scale
        self.orig_size = orig_size
        self.orig_intrinsics = orig_intrinsics
        self.size = target_size if target_size != () else orig_size
        self.intrinsics = ()
        self.crop_box = ()

        self.frame_ids = []
        self.image_paths = []
        self.gt_depth_paths = [] # for ground truth depth
        self.depth_paths = [] # for precomputed depth
        self.poses = []

        self.start = start
        self.end = end
        self.start_idx = None
        self.end_idx = None

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"The image directory can't be found in {str(self.image_dir)}, please check your data!")

        if not self.pose_path.is_file():
            raise FileNotFoundError(f"The pose file can't be found in {str(self.pose_path)}, please check your data!")

        self.set_target_intrinsics() # populates self.crop_box and self.intrinsics
        self.load_image_paths_and_poses() # populates self.frame_ids, self.image_paths and self.poses
        self.truncate_paths_and_poses() # truncates the lists, populates self.start_idx and self.end_idx

        if self.has_gt_depth:
            self.load_gt_depth_paths() # load in ground truth depth paths if needed

        if self.has_depth:
            self.load_depth_paths() # load in precomputed depth paths if needed

    def parse_image_path_and_frame_id(self, cols: List[str]) -> Tuple[Path, int]:
        frame_id = self.parse_frame_id(cols)
        image_name = f"{frame_id:0{PADDED_IMG_NAME_LENGTH}}.png"
        image_path = Path(self.image_dir, image_name)

        return image_path, frame_id

    def load_gt_depth_paths(self):
        # frame_ids are already truncated, no need to do it again for depths
        for frame_id in self.frame_ids:
            depth_path = Path(self.gt_depth_dir, f"{frame_id:0{PADDED_IMG_NAME_LENGTH}}.npy")

            if not depth_path.is_file():
                raise FileNotFoundError(f"The ground truth depth value at '{depth_path}' can't be found. Please check your data at '{self.gt_depth_dir}' or trying running 'projectVeloToImage' at 'modules.eval.kitti360' again.")

            self.gt_depth_paths.append(depth_path)

        return None

    def load_depth_paths(self):
        # frame_ids are already truncated, no need to do it again for depths
        for frame_id in self.frame_ids:
            depth_path = Path(self.depth_dir, f"{frame_id:0{PADDED_IMG_NAME_LENGTH}}.npy")

            if not depth_path.is_file():
                raise FileNotFoundError(f"The precomputed depth value at '{depth_path}' can't be found. Please check your data at '{self.depth_dir}' or trying running '3_precompute_depth_and_normals.py' again.")

            self.depth_paths.append(depth_path)

        return None

    @classmethod
    def parse_frame_id(cls, cols: List[str]) -> int:
        """
        Returns the frame id (or timestamp) as an integer, return None to
        enumerate frames starting from 0
        """
        # First convert to float to deal with scientific notation if it exists
        frame_id = int(float(cols[0]))
        return frame_id

    @classmethod
    def parse_pose(cls, cols: List[str]) -> torch.Tensor:
        """
        Parses a row of the pose file. The input is expected to be already
        split up using the whitespaces in between the columns.

        Assumes that poses are formatted in TUM RGB-D format, i.e.
        ```
        # timestamp tx(0)       ty(1)       tz(2)        qx(0)       qy(1)       qz(2)       qw(3)
        00000000000 0.002242719 0.006834473 0.035801470 -0.000058279 0.000172516 0.000063198 1.000000000
        ...
        ```

        where the `t` terms are the components of `t_wc` (translation vector)
        and the `q` terms are `q_wc` (quaternion) used for the homogenous
        transform `T_WC` that takes in coordinates in camera frame (C)
        and maps to the world frame (W), i.e. the cam2world transform
        """
        translation = cols[1:4]
        translation = [ float(val) for val in translation ]

        # float in python is equivalent to torch.double
        translation = torch.tensor(translation, requires_grad=False).double().reshape(3, 1)

        quaternion = cols[4:8]
        quaternion = [ float(val) for val in quaternion]
        quaternion = np.array(quaternion) # to use Rotation from scipy

        rot = Rotation.from_quat(quaternion).as_matrix()
        rot = torch.tensor(rot, requires_grad=False).double()

        # Combine rotation and translation as homogeneous transform matrix
        pose = torch.concat([rot, translation], dim=1)
        pose = torch.concat([pose, torch.tensor([[0, 0, 0, 1]])], dim=0)

        return pose

    def load_image_paths_and_poses(self) -> None:
        """
        Fetches poses from `self.pose_path` and the corresponding image paths
        from `self.image_dir`. Uses `self.parse_pose()` to convert the specific
        pose format to a unified notation.
        """
        with open(self.pose_path, "r") as pose_file:
            all_pose_rows = pose_file.readlines()

        for pose_row in all_pose_rows:
            pose_row = pose_row.strip()
            if pose_row[0] == "#":
                continue
            cols = pose_row.split(" ")

            image_path, frame_id = self.parse_image_path_and_frame_id(cols)

            if not image_path.exists():
                log.warning(f"Image {image_path} specified in pose file can't be found, skipping this frame")
                continue

            pose = self.parse_pose(cols)

            if frame_id in self.frame_ids:
                log.warning(f"Frame id {frame_id} has already been read. Skipping the associated images and poses.")
            else:
                self.frame_ids.append(frame_id)
                self.image_paths.append(image_path)
                self.poses.append(pose)

        if len(self.poses) == 0:
            raise RuntimeError(f"No poses have been read, please check your pose file at '{self.pose_path}'. Tip: Make sure the number of zeros padded for image file names (in configs/data.py) matches your dataset")

        if len(self.poses) != len(self.image_paths):
            raise RuntimeError(f"Different number of poses and images has been read, please check your pose file at '{self.pose_path}' and your images at '{self.image_dir}'")

        return None

    def set_target_intrinsics(self) -> None:
        """
        Sets attributes for the `CustomDataset`:
            `intrinsics`: The computed intrinsics for the resize `(fx, fy, cx, cy)`
            `crop_box`: The crop rectangle to preserve ratio of height/width
                    of the target image size as a `(left, upper, right, lower)`-tuple.
        """
        self.intrinsics, self.crop_box = compute_target_intrinsics(
                                            self.orig_intrinsics,
                                            self.orig_size,
                                            self.size)
        return None


    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        """
        Resize and crop the given image using torch interpolation and torchvision.

        Converts the `crop_box` notation to match the torchvision format.
        Coordinate system origin is at the top left corner, x is horizontal
        (pointing right), y is vertical (pointing down)

        Args:
            `images`: Unbatched tensors to resize and crop with shape `[C, H, W]`
        """
        C, H, W = image.shape

        crop_box = self.crop_box

        # Need to convert (left, upper, right, lower) to the torchvision format
        if crop_box != ():
            # Coordinates of the left top corner of the crop box
            left_top_corner_y = crop_box[1]
            left_top_corner_x = crop_box[0]

            box_height = crop_box[3] - crop_box[1]  # lower - upper
            box_width = crop_box[2] - crop_box[0]  # right - left
            image = tv_F.crop(
                image, left_top_corner_y, left_top_corner_x, box_height, box_width
            )

        # Using interpolate to have finer control, also RAFT doesn't work well with antialiased preprocessing
        resized_image = torch.nn.functional.interpolate(
            image.unsqueeze(0),
            self.size,
            mode="bilinear",
            antialias=False,
            align_corners=True,
        ).squeeze(0)

        return resized_image

    def truncate_paths_and_poses(self):
        start_idx_found = False
        end_idx_found = False
        # self.start and self.end are frame indices but we need the index in the lists
        while not start_idx_found:
            try:
                self.start_idx = self.frame_ids.index(self.start)
                start_idx_found = True
            except ValueError:
                log.warning(f"The specified dataset start frame {self.start} can't be found in frame_ids, looking for next possible frame id")
                self.start += 1

        while not end_idx_found:
            try:
                self.end_idx = self.frame_ids.index(self.end) if self.end != -1 else -1
                end_idx_found = True
            except ValueError:
                log.warning(f"The specified dataset end frame {self.end} can't be found in frame_ids, looking for next possible frame id")
                self.end -= 1

        if self.end == -1:
            self.frame_ids = self.frame_ids[self.start_idx:]
            self.image_paths = self.image_paths[self.start_idx:]
            self.poses = self.poses[self.start_idx:]
        else:
            self.frame_ids = self.frame_ids[self.start_idx:self.end_idx]
            self.image_paths = self.image_paths[self.start_idx:self.end_idx]
            self.poses = self.poses[self.start_idx:self.end_idx]

    def __len__(self) -> int:
        return len(self.poses)

    def __getitem__(self, idx: int) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Returns the frame id, the preprocessed image, and the scaled pose with index `idx` from the dataset"""
        frame_id = self.frame_ids[idx]
        image_path = self.image_paths[idx]
        pose = self.poses[idx]

        image = Image.open(image_path).convert("RGB")
        image = tv_F.pil_to_tensor(image) # [C, H, W]
        image = self.preprocess(image)
        scaled_pose = pose.clone()
        scaled_pose[:3, 3] = self.pose_scale * pose[:3, 3]
        image = image.to(self.device)
        scaled_pose = scaled_pose.to(self.device)

        return frame_id, image, scaled_pose


class KITTIDataset(CustomDataset):
    def __init__(self, image_dir: Union[Path, str], pose_path: Union[Path, str], pose_scale: float, orig_intrinsics: Tuple, orig_size: Tuple, target_size: Tuple, start: int = 0, end: int = -1):
        # KITTI data has poses for every frame and no frame id in poses.txt, so we need to count it
        self.frame_counter = 0
        super().__init__(image_dir, pose_path, pose_scale, orig_intrinsics, orig_size, target_size, start, end)

    def parse_image_path_and_frame_id(self, cols: List[str]) -> Tuple[Path, int]:
        frame_id = self.frame_counter
        image_name = f"{frame_id:0{PADDED_IMG_NAME_LENGTH}}.png"
        image_path = Path(self.image_dir, image_name)

        self.frame_counter += 1

        return image_path, frame_id

    @classmethod
    def parse_frame_id(cls, cols: List[str]) -> None:
        """Dummy method to keep the class interfaces consistent."""
        return None

    @classmethod
    def parse_pose(cls, cols: List[str]) -> torch.Tensor:
        """
        Assumes the pose file has rows saved in the KITTI format, 12 numbers
        representing the 3x4 transformation matrix from camera to world coordinates
        """
        values = [float(val) for val in cols] # no frame id for KITTI format

        pose = torch.tensor(values, requires_grad=False).double().reshape(3,4)
        pose = torch.concat([pose, torch.tensor([[0, 0, 0, 1]])] , dim=0)

        return pose


class KITTI360Dataset(CustomDataset):
    """
    Dataset that is used to load in and work with the ground truth data (pose
    and depth) of KITTI360. If you want to use predicted poses but KITTI360
    ground truth depth, make sure you extract the velodyne data properly and
    use a `CustomDataset` instance with the correct `gt_depth_dir` attribute.
    """
    def __init__(self, seq: int, cam_id: int, target_size: Tuple = (), pose_scale: float = 1.0, start: int = 0, end: int = -1):
        # KITTI360 specific preparation before calling the constructor of the super class
        self.seq = seq
        self.cam_id = cam_id

        sequence = '2013_05_28_drive_%04d_sync'%seq
        image_dir = Path(KITTI360_DIR, "data_2d_raw", sequence, f"image_0{cam_id}", "data_rect")
        gt_depth_dir = Path(KITTI360_DIR, "data_3d_raw", sequence, f"image_0{cam_id}", "depths")
        pose_path = Path(KITTI360_DIR, "data_poses", sequence, "cam0_to_world.txt")

        if cam_id in [0, 1]:
            orig_intrinsics = (552.554261, 552.554261, 682.049453, 238.769549)
            orig_size = (376, 1408)
        else:
            raise NotImplementedError("cam_id values other than 0 or 1 are not implemented for KITTI360. Please either choose 0 or 1.")

        if target_size == ():
            target_size = orig_size

        # Populate gt depth directory if it's empty using KITTI360 depth extraction
        gt_depth_dir_exists = gt_depth_dir.is_dir()

        if gt_depth_dir_exists:
            gt_depth_dir_is_empty = not any(gt_depth_dir.iterdir())

        if not gt_depth_dir_exists or gt_depth_dir_is_empty:
            log.info(f"The ground truth depth files for KITTI360 at {gt_depth_dir} do not exist. Reading and saving all values for {sequence} in that directory. This might take a while...")
            projectVeloToImage(cam_id, seq, KITTI360_DIR, max_d=200, image_name_padding=PADDED_IMG_NAME_LENGTH)

        # Call to constructor of CustomDataset, sets most attributes
        super().__init__(image_dir, pose_path, pose_scale, orig_intrinsics, orig_size, target_size, gt_depth_dir, start, end)


    @classmethod
    def parse_pose(cls, cols: List[str]) -> torch.Tensor:
        """
        Parses a row of the pose file. The input is expected to be already
        split up using the whitespaces in between the columns.

        Assumes that poses are formatted in KITTI360 format. As explained in
        the KITTI360 documentation:
        'Each line has 17 numbers, the first number is an integer denoting
        the frame index. The rest is a 4x4 matrix denoting the rigid body
        transform from the rectified perspective camera coordinates to the
        world coordinate system.' (i.e. T_WC, or cam2world)
        """
        transform = cols[1:] # 16 elements, 4x4 matrix
        transform = [float(val) for val in transform]
        pose = torch.tensor(transform, requires_grad=False).double().reshape(4, 4)

        return pose

class TUMRGBDDataset(CustomDataset):
    @classmethod
    def parse_frame_id(cls, cols: List[str]) -> float:
        """
        Returns the frame id (or timestamp) as a float, return None to
        enumerate frames starting from 0
        """
        # For the tum rgb-d dataset the timestamps (filenames) are floats
        frame_id = float(cols[0])
        return frame_id


class COLMAPDataset(CustomDataset):
    def __init__(self, colmap_dir: Union[Path, str], pose_scale: float, orig_intrinsics: Tuple, orig_size: Tuple, target_size: Tuple = (), gt_depth_dir: Union[Path, str] = None,  start: int = 0, end: int = -1):
        image_dir = Path(colmap_dir) / Path("images")
        data_dir = Path(colmap_dir) / Path("sparse/0")
        pose_txt = data_dir / Path("images.txt")
        points_txt = data_dir / Path("points3D.txt")
        self.pcd = None

        if not pose_txt.is_file():
            raise FileNotFoundError(f"Can't find the pose files in {str(pose_txt)}. If you only have .bin files, please run 'colmap model_converter --input_path 0 --output_path 0 --output_type TXT' in the right directory to convert binary files to txt files.")

        if not points_txt.is_file():
            raise FileNotFoundError(f"Can't find the points3D files in {str(points_txt)}. If you only have .bin files, please run 'colmap model_converter --input_path 0 --output_path 0 --output_type TXT' in the right directory to convert binary files to txt files.")

        self.load_colmap_map(points_txt) # populates self.pcd
        super().__init__(image_dir, pose_txt, pose_scale, orig_intrinsics, orig_size, target_size, gt_depth_dir, start, end)

    def load_colmap_map(self, points_txt: Path):
        """
        Saves the colmap point cloud from 'points3D.txt'
        The COLMAP outputs are formatted as follows:

        ```
        # 3D point list with one line of data per point:
        #   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
        ```
        """
        self.pcd = o3d.t.geometry.PointCloud()
        xyz = []
        rgb = []

        with open(points_txt, "r") as file:
            for line in file:
                if "#" in line:
                    continue  # means it's a comment
                else:
                    line_split = line.split(" ")

                    id, x, y, z, r, g, b = line_split[0:7]

                    # Open3D needs normalized integer values for RGB
                    r = int(r) / 255
                    g = int(g) / 255
                    b = int(b) / 255

                    xyz.append([float(x), float(y), float(z)])
                    rgb.append([r, g, b])

        self.pcd.point.positions = np.array(xyz)
        self.pcd.point.colors = np.array(rgb)

    @classmethod
    def parse_frame_id(cls, cols: List[str]) -> int:
        """
        Returns the frame id (the name of the image file) which is the last
        column in COLMAP format. The first column is the COLMAP internal image id.
        Expects to receive the space-split rows of `images.txt` from COLMAP
        describing the image poses, not the 2D points for each image (i.e.
        expects every other line of `images.txt`)
        """
        image_name = cols[-1] # has file extension, we don't want that
        image_stem = image_name.split(".")[0]
        frame_id = int(float(image_stem))
        return frame_id

    @classmethod
    def parse_pose(cls, cols: List[str]) -> torch.Tensor:
        """
        Parses a row of the pose file. Expects only the line containing the
        pose information.

        Assumes that the poses are in COLMAP format. COLMAP stores *unordered*
        trajectories and projected point coordinates as:
        ```
        # Image list with two lines of data per image:
        #   IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        #   POINTS2D[] as (X, Y, POINT3D_ID)
        ...
        ...
        ```

        Where the 4x4 transformation matrix describes T_CW (world to cam) and
        the `qw` comes first in the quaternion. These are in contrast to the
        TUM RGB-D dataset format that we follow.
        """
        # Get rotation from quaternion and translation as tensors
        quaternion = cols[1:5]  # still list of strings
        qw, qx, qy, qz = quaternion # qw first
        quaternion = [qx, qy, qz, qw] # now qw last for scipy Rotation
        quaternion = [float(val) for val in quaternion]
        quaternion = np.array(quaternion)  # scipy wants numpy arrays
        rotation = Rotation.from_quat(quaternion).as_matrix()
        rotation = torch.tensor(rotation, dtype=torch.double, requires_grad=False)

        translation = cols[5:8]
        translation = [float(val) for val in translation]
        translation = torch.tensor(
            translation, dtype=torch.double, requires_grad=False
        ).reshape(3, 1)

        # Combine into 4x4 matrix
        pose = torch.concat((rotation, translation), dim=1)  # [3, 4]
        pose = torch.concat((pose, torch.tensor([[0, 0, 0, 1]])), dim=0)  # [4, 4]

        # COLMAP has T_CW instead of T_WC like ORB-SLAM3, we work with ORB-SLAM3 notation
        pose = torch.linalg.inv(pose)

        return pose

    def load_image_paths_and_poses(self):
        """
        Fetches poses and corresponding image paths. Specific for COLMAP datasets
        because we need to skip every other line in `images.txt`.

        COLMAP stores *unordered* trajectories and projected point coordinates as:
        ```
        # Image list with two lines of data per image:
        #   IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        #   POINTS2D[] as (X, Y, POINT3D_ID)
        ...
        ...
        ```
        """
        skip_line = False # flag to skip every other line

        with open(self.pose_path, "r") as file:
            # Use while loop instead of reading all lines, `images.txt` can be really big
            while True:
                line = file.readline()
                if not line:
                    break
                line = line.strip()
                if len(line) >= 0 and line[0] != "#":
                    if skip_line:
                        skip_line = False
                        continue
                    else:
                        skip_line = True # to skip the next line

                    cols = line.split() # list of strings

                    image_path, frame_id = self.parse_image_path_and_frame_id(cols)

                    if not image_path.exists():
                        log.warning(f"Image {image_path} specified in pose file can't be found, skipping this frame")
                        continue

                    pose = self.parse_pose(cols)

                    if frame_id in self.frame_ids:
                        log.warning(f"Frame id {frame_id} has already been read. Skipping the associated images and poses.")
                    else:
                        self.frame_ids.append(frame_id)
                        self.image_paths.append(image_path)
                        self.poses.append(pose)

            if len(self.poses) == 0:
                raise RuntimeError(f"No poses have been read, please check your pose file at '{self.pose_path}'. Tip: Make sure the number of zeros padded for image file names (in configs/data.py) matches your dataset")

            if len(self.poses) != len(self.image_paths):
                raise RuntimeError(f"Different number of poses and images has been read, please check your pose file at '{self.pose_path}' and your images at '{self.image_dir}'")

            return None
