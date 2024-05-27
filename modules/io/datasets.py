import logging
from pathlib import Path
from typing import Union, Tuple, List

import torch
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import torchvision.transforms.functional as tv_F

from configs.data import KITTI360_DIR, PADDED_IMG_NAME_LENGTH
from modules.eval.kitti360 import projectVeloToImage
from modules.core.interfaces import BaseDataset

log = logging.getLogger(__name__)

class CustomDataset(BaseDataset):
    """
    Default CustomDataset implementation. Expects poses to be saved in TUM RGB-D
    dataset format unless `parse_pose()` is overriden.
    """
    def __init__(self, image_dir: Union[Path, str], pose_path: Union[Path, str], pose_scale: float, orig_intrinsics: Tuple, orig_size: Tuple, target_size: Tuple, start: int = 0, end: int = -1):
        self.image_dir = image_dir if isinstance(image_dir, Path) else Path(image_dir)
        self.pose_path = pose_path if isinstance(pose_path, Path) else Path(pose_path)

        self.pose_scale = pose_scale
        self.orig_size = orig_size
        self.orig_intrinsics = orig_intrinsics
        self.target_size = target_size
        self.new_intrinsics = ()
        self.crop_box = ()

        self.frame_ids = []
        self.image_paths = []
        self.poses = []

        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"The image directory can't be found in {str(self.image_dir)}, please check your data!")

        if not self.pose_path.is_file():
            raise FileNotFoundError(f"The pose file can't be found in {str(self.pose_path)}, please check your data!")

        self.compute_target_intrinsics() # populates self.crop_box and self.new_intrinsics
        self.load_image_paths_and_poses() # populates self.frame_ids, self.image_paths and self.poses

        # Truncate the image paths and the poses depending on config
        if end == -1:
            self.frame_ids = self.frame_ids[start:]
            self.image_paths = self.image_paths[start:]
            self.poses = self.poses[start:]
        else:
            self.frame_ids = self.frame_ids[start:end]
            self.image_paths = self.image_paths[start:end]
            self.poses = self.poses[start:end]

    def parse_pose(self, cols: List[str]) -> torch.Tensor:
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


    def load_image_paths_and_poses(self):
        """
        Fetches poses from `self.pose_path` and the corresponding image paths
        from `self.image_dir`. Uses `self.parse_pose()` to convert the specific
        pose format to a unified notation.
        """
        with open(self.pose_path, "r") as pose_file:
            pose_rows = pose_file.readlines()

        for pose_row in pose_rows:
            pose_row = pose_row.strip()
            if pose_row[0] == "#":
                continue
            cols = pose_row.split(" ")

            timestamp = int(cols[0]) # TODO will this cause problems?
            image_name = f"{timestamp:0{PADDED_IMG_NAME_LENGTH}}.png"
            image_path = Path(self.image_dir, image_name)

            if not image_path.exists():
                log.warning(f"Image {image_path} specified in pose file can't be found, skipping this frame")
                continue

            pose = self.parse_pose(cols)

            self.frame_ids.append(timestamp)
            self.image_paths.append(image_path)
            self.poses.append(pose)

        if len(self.poses) == 0:
            raise RuntimeError(f"No poses have been read, please check your pose file at '{self.pose_path}'")

        if len(self.poses) != len(self.image_paths):
            raise RuntimeError(f"Different number of poses and images has been read, please check your pose file at '{self.pose_path}' and your images at '{self.image_dir}'")

        return None


    def compute_target_intrinsics(self):
        """
        Computes new intrinsics after resizing an image to a given target size.
        The function computes how the original image should be cropped
        such that the aspect ratio (`H/W`) of the cropped image matches the ratio of
        `target_size` specified by the user. Returns new intrinsics and the crop box
        as two separate tuples.

        Code inspired from `compute_target_intrinsics()` from MonoRec: https://github.com/Brummi/MonoRec/blob/81b8dd98b86e590069bb85e0c980a03e7ad4655a/data_loader/kitti_odometry_dataset.py#L318

        Sets attributes for the `CustomDataset`:
            `new_intrinsics`: The computed intrinsics for the resize `(fx, fy, cx, cy)`
            `crop_box`: The crop rectangle to preserve ratio of height/width
                    of the target image size as a `(left, upper, right, lower)`-tuple.
        """
        height, width = self.orig_size
        height_target, width_target = self.target_size

        # Avoid extra computation if the original and target sizes are the same
        if self.orig_size == self.target_size:
            self.new_intrinsics = self.orig_intrinsics
            self.crop_box = ()
        else:
            r = height / width
            r_target = height_target / width_target

            fx, fy, cx, cy = self.orig_intrinsics

            if r >= r_target:
                # Width stays the same, we compute new height to keep target ratio
                new_height = r_target * width
                crop_box = (
                    0,
                    (height - new_height) // 2,
                    width,
                    height - (height - new_height) // 2,
                )

                rescale = width / width_target  # equal to new_height / height_target

                # Need to shift the camera center coordinate in the direction we crop
                cx = cx / rescale
                cy = (cy - (height - new_height) / 2) / rescale

            else:
                # Height stays the same, we compute new width to keep target ratio
                new_width = height / r_target
                crop_box = (
                    (width - new_width) // 2,
                    0,
                    width - (width - new_width) // 2,
                    height,
                )

                rescale = height / height_target  # equal to new_width / width_target

                cx = (cx - (width - new_width) / 2) / rescale
                cy = cy / rescale

            # Rescale the focal lenghts
            fx = fx / rescale
            fy = fy / rescale

            # Set the attributes of the dataset class
            self.new_intrinsics = (fx, fy, cx, cy)
            self.crop_box = tuple([int(coord) for coord in crop_box]) # should only have ints

        return None


    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        """
        Resize and crop the given image using torchvision.

        Converts the `crop_box` notation to match the torchvision format.
        Coordinate system origin is at the top left corner, x is horizontal
        (pointing right), y is vertical (pointing down)

        Args:
            `images`: Unbatched tensors to resize and crop of shape `[N, C, H, W]`
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
            self.target_size,
            mode="bilinear",
            antialias=False,
            align_corners=True,
        ).squeeze(0)

        return resized_image

    def __len__(self) -> int:
        return len(self.poses)


    def __get_item__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the preprocessed image and scaled pose with index `idx` from the dataset"""
        image_path = self.image_paths[idx]
        pose = self.poses[idx]

        image = Image.open(image_path).convert("RGB")
        image = tv_F.pil_to_tensor(image) # [C, H, W]

        scaled_pose = pose.clone()
        scaled_pose[:3, 3] = pose[:3, 3]

        return image, scaled_pose


# TODO implement so we can evaluate depth predictions
# with ground truth results
class KITTI360Dataset(CustomDataset):
    """
    Dataset that is used to load in and work with the ground truth data (pose
    and depth) of KITTI360
    """
    def __init__(self, seq: int, cam_id: int, target_size: Tuple = (), start: int = 0, end: int = -1):
        # KITTI360 specific preparation before calling the constructor of the super class
        self.seq = seq
        self.cam_id = cam_id

        sequence = '2013_05_28_drive_%04d_sync'%seq
        image_dir = Path(KITTI360_DIR, "data_2d_raw", sequence, f"image_0{cam_id}", "data_rect")
        pose_path = Path(KITTI360_DIR, "data_poses", sequence, "cam0_to_world.txt")
        self.gt_depth_dir = Path(KITTI360_DIR, "data_3d_raw", sequence, f"image_0{cam_id}", "depths")
        pose_scale = 1.0 # ground truth poses and depths so choosse scale is 1

        # P_rect_00, S_rect_00 and P_rect_01, S_rect_01 from perspective.txt in KITTI360 calibration
        if cam_id in [0, 1]:
            orig_intrinsics = (552.554261, 552.554261, 682.049453, 238.769549)
            orig_size = (376, 1408)
        else:
            raise NotImplementedError("cam_id values other than 0 or 1 are not implemented for KITTI360. Please either choose 0 or 1.")

        if target_size == ():
            target_size = orig_size

        # Call to the constructor of CustomDataset takes care of most things
        super().__init__(image_dir, pose_path, pose_scale, orig_intrinsics, orig_size, target_size, start, end)

        # Save and load paths for the ground truth depths
        self.gt_depth_paths = []

        gt_depth_dir_exists = self.gt_depth_dir.is_dir()
        gt_depth_dir_is_empty = not any(self.gt_depth_dir.iterdir())

        if not gt_depth_dir_exists or gt_depth_dir_is_empty:
            log.info(f"The ground truth depth files for KITTI360 at {self.gt_depth_path} do not exist. Reading and saving all values for {sequence} in that directory. This might take a while...")
            projectVeloToImage(cam_id, seq, KITTI360_DIR, max_d=200, image_name_padding=PADDED_IMG_NAME_LENGTH)

        self.load_gt_depth_paths()

    def load_gt_depth_paths(self):
        for frame_id in self.frame_ids:
            depth_path = Path(self.gt_depth_dir, f"{frame_id:0{PADDED_IMG_NAME_LENGTH}}.npy")

            if not depth_path.is_file():
                raise FileNotFoundError(f"The ground truth depth value at '{depth_path}' can't be found. Please check your data at '{self.gt_depth_dir}' or trying running 'projectVeloToImage' at 'modules.eval.kitti360' again.")

            self.gt_depth_paths.append(depth_path)

        return None


    def parse_pose(self, cols: List[str]) -> torch.Tensor:
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


