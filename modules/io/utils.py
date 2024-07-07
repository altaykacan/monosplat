"""Utility functions for data and input-output related operations"""
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Union, List, Callable, Tuple, Dict

import cv2
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement

from configs.data import SUPPORTED_DATASETS
from modules.core.maps import PointCloud
from modules.core.interfaces import BaseLogger, BaseReconstructor, BaseDataset
from modules.io.datasets import CustomDataset, KITTI360Dataset, KITTIDataset, TUMRGBDDataset, ColmapDataset, CombinedColmapDataset

log = logging.getLogger(__name__)

def save_image_torch(tensor: torch.Tensor, name: str = "debug", output_dir: Union[str, Path] = ".") -> None:
    """
    Saves a torch tensor representing an image into disk, useful for debugging.
    Only saves the first sample in batched input
    """
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    if not output_dir.is_dir():
        logging.warning(f"Output directory for saving tensors '{str(output_dir)}' does not exist. Creating it...")
        output_dir.mkdir(exist_ok=True, parents=True)

    if (".png" not in str(name).lower()) or (".jpg" not in str(name).lower()):
        output_path = output_dir / Path(f"{name}.png")
    else:
        output_path = output_dir / Path(name)

    if len(tensor.shape) == 4: # ignore rest of the batch
        tensor = tensor[0, :, :, :]
    tensor = tensor.squeeze()

    if len(tensor.shape) == 3: # rgb image
        plt.imsave(output_path, tensor.detach().cpu().permute(1, 2,0).numpy())
    if len(tensor.shape) == 2: # binary mask or 1-channel image
        plt.imsave(output_path, tensor.detach().cpu().numpy())


def clean_batch(batch: torch.Tensor, remove_list: List[int] = []) -> torch.Tensor:
    """
    Removes elements from a batched `[N, C, H, W]` torch tensor with the batch
    indices that have `1` elements in `remove_list`. The batch elements that will not
    be removed should have `0` in the corresponding places in `remove_list`.
    Returns None if the cleaned batch is empty
    """
    if len(batch.shape) == 4:
        N, C, H, W = batch.shape
    else:
        N = len(batch) # 1D tensor with a scalar value per batch element

    if len(remove_list) == 0:
        return batch

    if N != len(remove_list):
        raise ValueError(f"The size of your batch ({N}) does not match the size of remove_list ({remove_list})")

    cleaned_batch = []
    for i, batch_el in enumerate(batch):
        if remove_list[i] == 1:
            continue
        else:
            cleaned_batch.append(batch_el)

    if len(cleaned_batch) == 0:
        return None

    if isinstance(cleaned_batch[0], torch.Tensor):
        cleaned_batch = torch.stack(cleaned_batch, dim=0)
    else:
        cleaned_batch = torch.tensor(cleaned_batch)

    return cleaned_batch


class Logger(BaseLogger):
    def __init__(self, reconstructor: BaseReconstructor):
        self.reconstructor = reconstructor
        self.output_dir = self.reconstructor.output_dir / Path("logs")

    def log_step(self, state: Dict):
        ids = [val.item() for val in state["ids"]]

        # i iterates over the index within the batch
        for i, frame_id in enumerate(ids):
            for key, value in state.items():
                if key == "ids":
                    continue
                if key == "depths":
                    value[i, : , :, :] = 1 / (value[i, : ,: ,:] + 0.0001)
                save_image_torch(value[i, :, : ,:], name=f"{frame_id}_{key}", output_dir=self.output_dir)


def ask_to_clear_dir(dir_path: Union[str, Path]) -> bool:
    """
    Asks the user to delete existing files in a directory if there are existing
    files. If the user types in 'y' the files are deleted. Useful for scripts
    that compute some values and save them for later usage. Returns a boolean
    representing whether the operation should be continued or not.
    """
    if isinstance(dir_path, str):
        dir_path = Path(dir_path)

    # If directory doesn't exist don't ask to delete
    if not dir_path.is_dir():
        return True

    not_empty = any(dir_path.iterdir())
    should_continue = True
    if not_empty:
        while True:
            answer = input(f"The directory you specified {str(dir_path)} is not empty, do you want to delete existing files before continuing? [yes/no]: ")

            if answer.lower() == "yes":
                print(f"Deleting existing files at {str(dir_path)}...")
                for file in dir_path.iterdir():
                    if file.is_file():
                        file.unlink()

                should_continue = True
                still_not_empty = any(dir_path.iterdir())
                if still_not_empty:
                    print("The directory is still not empty. There must be subdirectories, please delete them manually and run this script again!")
                    should_continue = False
                break

            elif answer.lower() == "no":
                print(f"Not deleting existing files at {str(dir_path)} and aborting...")
                should_continue = False
                break

            print("Please type 'yes' or 'no'")

    return should_continue


def find_latest_number_in_dir(dir_path: Union[str, Path]) -> int:
    """
    Finds and returns the largest number in a directory that contains
    enumerated subdirectories with the naming convention `xx_some_directory_name`
    where `xx` represents the numbering of the run. Useful for keeping track
    of reconstruction or gaussian splat training runs.
    """
    largest_idx = 0

    if isinstance(dir_path, str):
        dir_path = Path(dir_path)

    for path in dir_path.iterdir():
        path_str = str(path.name)

        # Don't count hidden files
        if path_str[0] == ".":
            continue

        try:
            # Expecting xx_some_directory_name as a naming convention where xx is a number
            idx = int(path_str.split("_")[0])
            if idx > largest_idx:
                largest_idx = idx
        except ValueError:
            log.warning(f"Couldn't parse '{path_str}', not counting it when enumarating the files in '{str(dir_path)}'")

    return largest_idx


def find_all_files_in_dir(directory: Union[Path, str], extension: str = "") -> List:
    """
    Find all files in a directory (optionally with an extension `.png`, `.txt`
    etc.) and returns a list of the paths to each of the files.
    """
    if isinstance(directory, str):
        directory = Path(directory)

    # Extension needs to start with "."
    if extension != "" and extension[0] != ".":
        extension = "." + extension

    paths = []
    for f in directory.iterdir():
        if f.is_file():
            if extension == "":
                paths.append(f.absolute())
            if extension != "" and extension == f.suffix:
                paths.append(f.absolute())

    return paths


def get_parse_and_stamp_fn(dataset: str) -> Tuple[Callable, Callable]:
    """
    Returns the dataset-specific functions for parsing the 4x4 T_WC (cam to
    world) pose matrix and the frame id (or the timestamp) from the cleaned
    rows of the pose file.
    """
    if dataset == "custom":
        dataset_class = CustomDataset
        log.info(f"Using a CustomDataset for extracting frame ids and poses...")
    elif dataset == "kitti360":
        dataset_class = KITTI360Dataset
        log.info(f"Using a KITTI360Dataset for extracting frame ids and poses...")
    elif dataset == "kitti":
        dataset_class = KITTIDataset
        log.info(f"Using a KITTIDataset for extracting frame ids and poses...")
    elif dataset == "tum_rgbd":
        dataset_class = TUMRGBDDataset
        log.info(f"Using a TUMRGBDDataset for extracting frame ids and poses...")
    elif dataset  == "colmap":
        dataset_class = ColmapDataset
        log.info(f"Using a ColmapDataset for extracting frame ids and poses...")
    elif dataset not in SUPPORTED_DATASETS:
        raise NotImplementedError(f"The dataset you specified ({dataset}) is not supported yet, supported datasets are: {str(SUPPORTED_DATASETS)}")
    else:
        raise NotImplementedError("Please add the dataset you implemented to the if statement here")

    parse_fn = dataset_class.parse_pose
    stamp_fn = dataset_class.parse_frame_id

    return parse_fn, stamp_fn


def read_all_poses_and_stamps(pose_path: Union[Path, str], dataset: str = "custom") -> Tuple[torch.Tensor, List[int]]:
    """
    Reads all poses from the rows of `pose_path` depending on the dataset.
    Returns a `torch.Tensor` with shape `[N, 4, 4]` where `N` is the number of
    camera centers specified in the pose file and a list of all timestamps (or
    frame ids) with the associated poses.
    """
    if isinstance(pose_path, str):
        pose_path = Path(pose_path)

    parse_fn, stamp_fn = get_parse_and_stamp_fn(dataset)

    poses = []
    stamps = []
    with open(pose_path, "r") as p_file:
        all_pose_rows = p_file.readlines()

    for pose_row in all_pose_rows:
        pose_row = pose_row.strip()
        if pose_row[0] == "#":
            continue
        cols = pose_row.split(" ")

        pose = parse_fn(cols) # returns None if row should be skipped (relevant for COLMAP)
        if pose is None:
            continue
        else:
            stamp = stamp_fn(cols) # returns None if the dataset doesn't support timestamps in pose file
            poses.append(pose)
            stamps.append(stamp)

    # Means that all are None and dataset doesn't have timestamps in the pose file
    if stamps[0] is None:
        stamps = [s for s in range(len(stamps))]

    # Sort the poses and stamps in increasing order
    zipped = sorted(zip(stamps, poses))
    poses = [pair[1] for pair in zipped]
    stamps = [pair[0] for pair in zipped]

    poses = torch.stack(poses, dim=0)

    return poses, stamps


def read_poses_from_stamps(pose_path: Union[Path, str], matching_stamps: List[int], dataset: str = "custom") -> torch.Tensor:
    """
    Reads poses from `pose_path` if and only if the timestamps of the poses
    match the ones in `matching_stamps`. This is relevant for the cases
    when the pose estimation system does not give you poses for every ground
    truth pose entry you have.
    """
    if isinstance(pose_path, str):
        pose_path = Path(pose_path)

    parse_fn, stamp_fn = get_parse_and_stamp_fn(dataset)

    poses = []
    stamp_count = 0 # counter for datasets that don't have frame ids in pose files
    with open(pose_path, "r") as p_file:
        all_pose_rows = p_file.readlines()

    for pose_row in all_pose_rows:
        pose_row = pose_row.strip()

        # Skip comments
        if pose_row[0] == "#":
            continue

        cols = pose_row.split(" ")
        pose = parse_fn(cols)
        stamp = stamp_fn(cols)

        if stamp is not None and stamp not in matching_stamps:
            continue
        poses.append(pose)

    poses = torch.stack(poses, dim=0)
    return poses


def create_rgb_txt(path: Path, image_dir: Path):
    """
    Creates `rgb.txt` needed for TUM RGB-D dataset format.
    Args:
        `path`: The path where the `rgb.txt` file will be saved
        `image_dir`: The path to the directory where images are saved
    """
    path = Path(path)
    image_dir = Path(image_dir)

    image_paths = []
    log_time = datetime.now().strftime('%Y-%m-%d:%H-%M-%S')

    for p in image_dir.iterdir():
        if p.is_file() and p.suffix == ".png":
            image_paths.append(p)

    image_paths = sorted(image_paths)
    out_path = path / Path("rgb.txt")
    with open(out_path, "w") as out_file:
        out_file.write("# color images\n")
        out_file.write(f"# in the TUM RGB-D dataset format, created at (year-month-day:hour-minute-second): {log_time}\n")
        out_file.write("# timestamp path\n")
        for image_path in image_paths:
            timestamp = float(image_path.stem) # no extension
            out_file.write(f"{timestamp} {image_path.relative_to(path)}\n")

    log.info(f"Created 'rgb.txt' at {str(path)}, wrote {len(image_paths) + 3} lines...")


def create_depth_txt(path: Union[Path, str], depth_dir: Union[Path, str]):
    """
    Creates `depth.txt` needed for TUM RGB-D dataset format.
    Args:
        `path`: The path where the `depth.txt` file will be saved
        `depth_dir`: The path to the directory where depth images are saved
    """
    path = Path(path)
    depth_dir = Path(depth_dir)

    image_paths = []
    log_time = datetime.now().strftime('%Y-%m-%d:%H-%M-%S')

    for p in depth_dir.iterdir():
        if p.is_file() and p.suffix == ".png":
            image_paths.append(p)

    image_paths = sorted(image_paths)
    out_path = path / Path("depth.txt")
    with open(out_path, "w") as out_file:
        out_file.write("# depth images\n")
        out_file.write(f"# in the TUM RGB-D dataset format, created at (year-month-day:hour-minute-second): {log_time}\n")
        out_file.write("# timestamp path\n")
        for image_path in image_paths:
            timestamp = float(image_path.stem) # no extension
            out_file.write(f"{timestamp} {image_path.relative_to(path)}\n")


def create_associations_txt(path: Path, image_dir: Path, depth_dir: Path):
    """
    Creates the `associations.txt` needed for running RGB-D SLAM with the
    TUM RGB-D dataset format in ORB-SLAM3.

    The format of the file is, where we do not expect any comments:
    ```
    timestamp rgb_relative_path timestamp depth_relative_path
    ```

    Args:
        `path`: The path where the `associations.txt` file will be saved
        `image_dir`: The path to the directory where images are saved
        `depth_dir`: The path to the directory where depth images are saved
    """
    image_paths = []
    depth_paths = []
    log_time = datetime.now().strftime('%Y-%m-%d:%H-%M-%S')

    for p in image_dir.iterdir():
        if p.is_file() and p.suffix == ".png":
            image_paths.append(p)

    for d in depth_dir.iterdir():
        if d.is_file() and d.suffix == ".png":
            depth_paths.append(d)

    image_paths = sorted(image_paths)
    depth_paths = sorted(depth_paths)
    if len(image_paths) != len(depth_paths):
        raise ValueError(f"The length of your image paths ({len(image_paths)}) and depth paths ({len(depth_paths)} do not match! Check your image and depth directories.")

    out_path = path / Path("associations.txt")
    with open(out_path, "w") as out_file:
        for image_path, depth_path in zip(image_paths, depth_paths):
            timestamp = float(image_path.stem) # no extension
            out_file.write(f"{timestamp} {image_path.relative_to(path)} {timestamp} {depth_path.relative_to(path)}\n")

    print("wrote ", str(out_path))


def create_scales_and_shifts_txt(
        depth_paths: Union[Path, str],
        scales: Union[torch.Tensor, float],
        shifts: Union[torch.Tensor, float],
        pose_path: Union[Path, str],
        stamps: List[int],
        ) -> None:
    """
    Expects the scales and shifts to apply to each depth prediction individually
    as `[num_frames]` tensors. Stamps is a list of integers that give the timestamps
    of frames with poses so we can only save the depths with poses. This is important
    because only those depths have calculated scale and shift values.

    Saves the output at the same directory as `pose_path` in a txt file with the format:

    ```
    # Scale factors and shifts to apply to each depth prediction for 'pose_path'
    # scale and shift are floats
    precomputed_depth_path timestamp scale shift
    ```
    """
    file_path = pose_path.parent / Path("scales_and_shifts.txt")
    log_time = datetime.now().strftime('%Y-%m-%d:%H-%M-%S')

    if isinstance(scales, float):
        scales = torch.ones_like(torch.tensor(stamps)) * scales
    if isinstance(shifts, float):
        shifts = torch.ones_like(torch.tensor(stamps)) * shifts

    # Write to file, depth paths and stamps are alrady matching
    log.info(f"Writing scales and shifts as a txt file at {str(file_path)}...")
    with open(file_path, "w") as file:
        file.write(f"# Scale factors and shifts to apply to each depth prediction for {pose_path.name}\n")
        file.write(f"# scale and shift values are floats, created at (year-month-day:hour-minute-second): {log_time}\n")
        file.write("# precomputed_depth_path timestamp scale shift\n")
        for i in range(len(stamps)):
            file.write(f"{depth_paths[i]} {stamps[i]} {scales[i].item()} {shifts[i].item()}\n")
    log.info(f"Wrote scales_and_shifts.txt")


def create_intrinsics_txt(out_dir: Path, dataset: BaseDataset) -> None:
    """
    Writes `intrinsics.txt` to `out_dir`, for use with 3d gaussian splatting
    """
    out_path = out_dir / Path("intrinsics.txt")

    H, W = dataset.size
    fx, fy, cx, cy = dataset.intrinsics
    first_frame_id = dataset.frame_ids[0]

    # Save pose scale if it's not the default value
    if dataset.pose_scale != 1.0:
        scale = dataset.pose_scale
    elif dataset.depth_scale is None and dataset.scales_and_shifts_path is None:
        scale = dataset.pose_scale
    # If all depth scales are the same, use it's reciprocal to save as pose scale
    elif (torch.tensor(list(dataset.scales.values())) == dataset.scales[first_frame_id]).all():
        scale = 1 / dataset.scales[first_frame_id]
    else:
        # TODO add support for scale and shift factor reading in 3dgs (probably can save the scaled and shifted arrays on disk) (low prio)
        scale = -1 # else we must use per-depth scale and shift factors from scales_and_shifts.txt in the 3dgs code

    with open(out_path, "w") as file:
        file.write("# Camera list with one line of data per camera, -1 values mean they are ignored: \n")
        file.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, Fx, Fy, Cx, Cy, ROI_LEFT, ROI_TOP, ROI_RIGHT, ROI_LOWER, POSE_SCALE \n")
        file.write("# Number of cameras: 1 \n")
        file.write(f"1 PINHOLE {W} {H} {fx} {fy} {cx} {cy} -1 -1 -1 -1 {scale}")

    log.info(f"Wrote intrinsics.txt at '{str(out_path)}'")


# TODO make this adapt to the start and end of our datasets, currently we just copy the original pose files
def create_poses_for_3dgs(out_dir: Path, dataset: BaseDataset) -> None:
    """
    Creates either `colmap_poses.txt` or `poses.txt` for running 3D gaussian
    splatting depending on the dataset you provide. Saves unscaled poses.
    """

    if isinstance(dataset, (ColmapDataset, CombinedColmapDataset)):
        log.info(f"Found COLMAP-based dataset, saving unscaled poses 'colmap_poses.txt' at {str(out_dir)}")
        colmap_pose_path = dataset.pose_path
        out_path = out_dir / Path("colmap_poses.txt")
        shutil.copy(colmap_pose_path, out_path)

    else:
        log.info(f"Found a custom dataset, saving unscaled poses 'poses.txt' at {str(out_dir)}")
        slam_pose_path = dataset.pose_path
        out_path = out_dir / Path("slam_poses.txt")
        shutil.copy(slam_pose_path, out_path)

    log.info(f"Wrote poses at '{str(out_path)}'")


def read_deepscenario_sparse_cloud(filename: Union[str, Path]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reads in the ground truth reconstructions for the custom data from DeepScenario.
    Returns the `xyz` and `rgb` values as torch tensors with shape
    `[3, num_points]` to then be converted to any point cloud format.

    The input is expected to be a path to a file formatted as a json file
    as:
    ```
    [
      {
        "shots": {...},
        "cameras": {...},
        "points":{
            "point_id": {
                "color": [R, G, B],
                "coordinates": [x, y, z],
            },
            ...
        },
        "biases": {...},
        "rig_cameras": {...},
        "rig_instances": {...},
        "reference_lla": {...},
      }
    ]
    ```
    """
    if isinstance(filename, str):
        filename = Path(filename)

    with open(filename, "rb") as file:
        data = json.load(file) # might take some time, if the point cloud is too large use ijson

    points = data[0]["points"] # the whole json is saved as a list so we need to access the first element
    xyz = []
    rgb = []
    for point_id in points.keys():
        coords = points[point_id]["coordinates"] # both are lists with three elements
        colors = points[point_id]["color"]
        xyz.append(coords)
        rgb.append(colors)
    xyz = torch.tensor(xyz, dtype=torch.double).T # [3, num_points]
    rgb = torch.tensor(rgb, dtype=torch.uint8).T

    return xyz, rgb


def save_poses_as_ply(poses: List[torch.Tensor], scale: float = 1.0, filename: Union[str,Path]="poses.ply", output_dir: Union[str, Path]="."):
    """
    Saves a list of `(4,4)` trajectories as a `.ply` file. Optionally scales the poses.
    """
    filename = Path(filename)
    output_dir = Path(output_dir)
    filepath = output_dir / filename
    x = []
    y = []
    z = []

    for pose in poses:
        x.append(scale * pose[0, 3].item())
        y.append(scale * pose[1, 3].item())
        z.append(scale * pose[2, 3].item())

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    # RGB values for each point on the trajectory, set to be light green
    r = np.ones_like(x) * 144
    g = np.ones_like(x) * 238
    b = np.ones_like(x) * 144

    ply_header = (
        "ply\n"
        "format ascii 1.0\n"
        "element vertex %d\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header" % x.shape
    )

    np.savetxt(
        filepath,
        np.column_stack((x, y, z, r, g, b)),
        fmt="%f %f %f %d %d %d",
        header=ply_header,
        comments="",
    )


def read_ply(path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reads a ply file and reads in positions, colors, and normals.

    Slightly modified from the original 3D Gaussian Splatting repo:
    https://github.com/graphdeco-inria/gaussian-splatting/blob/main/scene/dataset_readers.py
    """
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0

    try:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    except Exception as E:
        log.warning(f"Encountered exception '{E}' when trying to read in normals. Setting them all as zeros")
        normals = np.zeros_like(positions)

    return torch.from_numpy(positions), torch.from_numpy(colors), torch.from_numpy(normals)


def read_ply_o3d(path: Union[str, Path], convert_to_float32: bool = False) -> o3d.geometry.PointCloud:
    """
    Reads in a .ply file describing a point cloud and returns PointCloud
    object from open3d
    """
    if isinstance(path, Path):
        path = str(path)

    pcd = o3d.t.io.read_point_cloud(path)

    # Open3D's project to depth function needs float32
    if convert_to_float32:
        pcd.point.positions = pcd.point.positions.to(o3d.core.Dtype.Float32)
    return pcd


def save_pcd_o3d(
        xyz: torch.Tensor,
        rgb: torch.Tensor = None,
        normals: torch.Tensor = None,
        paint_color: str = None,
        logdir: Union[str, Path] = ".",
        filename: Union[str, Path] = "debug_cloud",
        ) -> None:
    """
    Expects a `[3, num_points]` torch tensor describing the 3D coordinates of a
    point cloud and optional `rgb` and `normals` tensors with the same shape
    and saves it as a `.ply` in `filename`. Give filenames without the extensions.
    Uses open3D and our custom `PointCloud` class. You can also paint the
    whole point cloud to either `['white', 'red', 'green', 'pink', 'blue']`
    """
    if isinstance(logdir, str):
        logdir = Path(logdir)
    if isinstance(filename, str):
        filename = Path(filename)

    # RGB colors in range 0..1
    valid_colors = {
        'white': [1.0, 1.0, 1.0],
        'red': [1.0, 0.059, 0.059],
        'green': [0.376, 0.961, 0.259],
        'pink': [0.929, 0.49, 0.961],
        'blue': [0.49, 0.961, 0.953],
        }

    if (paint_color is not None) and not (paint_color in valid_colors.keys()):
        raise ValueError(f"Provided '{paint_color}' is not in {list(valid_colors.keys())}. Please provide a valid color name.")

    cloud = PointCloud()
    cloud.increment(xyz, rgb, normals)
    cloud.postprocess()
    if paint_color is not None:
        color_rgb = valid_colors[paint_color]
        cloud.pcd = cloud.pcd.paint_uniform_color(color_rgb)

    cloud.save(str(filename) + ".ply", logdir)
