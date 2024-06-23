import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

from modules.core.reconstructors import ReconstructorFactory
from modules.core.backprojection import Backprojector
from modules.core.models import RAFT
from modules.depth.models import Metric3Dv2, KITTI360DepthModel, PrecomputedDepthModel
from modules.segmentation.models import SegFormer, MaskRCNN
from modules.io.datasets import CustomDataset, ColmapDataset, CombinedColmapDataset, KITTI360Dataset
from modules.io.utils import find_latest_number_in_dir

def main(args):
    root_dir = Path(args.root_dir)
    recon_name = args.recon_name
    reconstructor_type = args.reconstructor
    backprojector_type = args.backprojector
    dataset_type = args.dataset
    pose_path = args.pose_path
    pose_scale = args.pose_scale
    depth_scale = args.depth_scale
    scales_and_shifts_path = args.scales_and_shifts_path
    intrinsics = args.intrinsics
    target_size = args.target_size
    dropout = args.dropout
    max_d = args.max_d
    batch_size = args.batch_size
    clean_pointcloud = args.clean_pointcloud
    depth_model_type = args.depth_model
    normal_model_type = args.normal_model
    seg_model_type = "segformer" if (args.seg_model is None and args.reconstructor == "moving_obj") else args.seg_model
    ins_seg_model_type = "mask_rcnn" if (args.ins_seg_model is None and args.reconstructor == "moving_obj") else args.ins_seg_model
    flow_model_type = "raft" if (args.flow_model is None and args.reconstructor == "moving_obj") else args.flow_model
    classes_to_remove = args.classes_to_remove
    cam_id = 0 if (args.cam_id is None and args.dataset == "kitti360") else args.cam_id
    seq_id = 0 if (args.seq_id is None and args.dataset == "kitti360") else args.seq_id
    debug = args.debug

    recon_root_dir = root_dir / Path("reconstructions")
    recon_root_dir.mkdir(exist_ok=True, parents=True)

    colmap_dir = root_dir / Path("poses/colmap")
    depth_dir = root_dir / Path("data/depths/arrays") # as .npy arrays, optional
    image_dir = root_dir / Path("data/rgb") # as png files

    # Expecting all folders to have 'xx_some_folder' where 'xx' is a number
    latest_recon_idx = find_latest_number_in_dir(recon_root_dir)
    if recon_name is None:
        recon_name = f"{latest_recon_idx + 1}_reconstruction"
    else:
        recon_name = f"{latest_recon_idx + 1}_{recon_name}"

    recon_dir = recon_root_dir / Path(recon_name)
    recon_dir.mkdir(exist_ok=True, parents=True)

    # Setup logging
    log_time = datetime.now().strftime('%Y-%m-%d:%H-%M-%S')
    log_path = recon_dir.absolute() / Path("log_recon.txt")
    with open(log_path, 'w'): # to clear existing log files
        pass
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(levelname)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(f"Log file for '6_create_pointcloud.py', created at (year-month-day:hour-minute-second): {log_time}")
    logging.info(f"Arguments: \n{args}") # TODO remove after debug
    # logging.info(f"Arguments: \n{json.dumps(vars(args), indent=4)}")
    logging.info(f"Using the image directory: {str(image_dir.absolute())}")

    # TODO put this in a factory
    # Get dataset
    if dataset_type == "colmap":
        dataset = ColmapDataset(
            colmap_dir,
            pose_scale=pose_scale,
            target_size=target_size,
            orig_intrinsics=intrinsics,
            depth_dir=depth_dir,
            depth_scale=depth_scale,
            scales_and_shifts_path=scales_and_shifts_path
            )
        if pose_path is not None:
            dataset.pose_path = pose_path
            logging.info(f"Overwriting COLMAP pose path with user specified pose path {str(pose_path)}")
        else:
            logging.info(f"No pose path provided to script. Using the default COLMAP pose {str(dataset.pose_path)}")

    if dataset_type == "combined_colmap":
        dataset = CombinedColmapDataset(
            colmap_dir,
            pose_scale=pose_scale,
            target_size=target_size,
            orig_intrinsics=intrinsics,
            depth_dir=depth_dir,
            depth_scale=depth_scale,
            scales_and_shifts_path=scales_and_shifts_path
            )
        if pose_path is not None:
            dataset.pose_path = pose_path
            logging.info(f"Overwriting COLMAP pose path with user specified pose path {str(pose_path)}")
        else:
            logging.info(f"No pose path provided to script. Using the default COLMAP pose {str(dataset.pose_path)}")

    elif dataset_type == "custom":
        dataset = CustomDataset(
            image_dir,
            pose_path,
            pose_scale=pose_scale,
            target_size=target_size,
            orig_intrinsics=intrinsics,
            depth_dir=depth_dir,
            depth_scale=depth_scale,
            scales_and_shifts_path=scales_and_shifts_path
            )
    elif dataset_type == "kitti360":
        dataset = KITTI360Dataset(seq_id, cam_id, pose_scale, target_size)
        pose_path = dataset.pose_path # GT poses

    # Get backprojector
    backprojector_cfg = {"dropout": dropout, "max_d": max_d}
    if backprojector_type == "simple":
        backprojector = Backprojector(backprojector_cfg, dataset.intrinsics)
    else:
        raise NotImplementedError

    # TODO put this in a factory
    # Get depth model
    if depth_model_type == "metric3d_vit":
        depth_model = Metric3Dv2(dataset.intrinsics, backbone="vit_giant")
    elif depth_model_type == "kitti360":
        if not isinstance(dataset, KITTI360Dataset):
            raise ValueError("To use a KITTI360 ground truth depth model you must choose a KITTI360 dataset!")
        depth_model = KITTI360DepthModel(dataset)
    elif depth_model_type == "precomputed":
        depth_model = PrecomputedDepthModel(dataset)

    # Create reconstructor and run it
    factory = ReconstructorFactory(dataset, backprojector, depth_model)
    recon_config = {
        "batch_size": batch_size,
        "output_dir": recon_dir,
        "clean_pointcloud": clean_pointcloud,
        "flow_model_type": flow_model_type,
        "ins_seg_model_type": ins_seg_model_type,
        "seg_model_type": seg_model_type,
        "classes_to_remove": classes_to_remove,
        "normal_model_type": normal_model_type
        }
    reconstructor = factory.get_reconstructor(reconstructor_type, recon_config)
    reconstructor.run()


if __name__=="__main__":
    # class DebugArgs(NamedTuple):
    #     root_dir: str = "/usr/stud/kaa/data/root/ds01"
    #     recon_name: str = None
    #     reconstructor: str = "simple" # options: "simple" or "moving_obj"
    #     backprojector: str = "simple" # options: "simple" or "semantic"
    #     dataset: str = "colmap" # either "colmap", "combined_colmap", "kitti360", or "custom"
    #     pose_path: str = None # default is None, only needed for custom dataset, can overwrite it for colmap datasets
    #     pose_scale: float = 19.96 # default is 1.0
    #     depth_scale: float = None # default is None, scales all depths by constant value
    #     scales_and_shifts_path: str = None # default is None, scales and shifts depths individually based on the txt file pointed to by this path
    #     intrinsics: list = [534.045,  534.045, 512, 288] # input as '--intrinsics fx fy cx cy'
    #     target_size: tuple = () # by default is empty tuple, it means the original size will be used
    #     dropout: float = 0.99 # random dropout probability of points
    #     max_d: float = 50.0 # maximum depth threshold for backprojection
    #     batch_size: int = 4 # batch size used for reconstruction, larger values are faster but needs more GPU memory
    #     clean_pointcloud: bool = False # flag to whether use statistical outlier removal on the final pointcloud
    #     depth_model: str = "metric3d_vit" # available options "metric3d_vit", "precomputed", and "kitti360" for ground truth depths (only works when "--dataset kitti360")
    #     normal_model: str = "metric3d_vit" # available options "metric3d_vit", "precomputed", and "kitti360" for ground truth depths (only works when "--dataset kitti360")
    #     seg_model: str = "segformer" # default None, segmentation model needed for '--backprojector semantic' and '--reconstructor moving_obj'
    #     ins_seg_model: str = "mask_rcnn" # default None, instance segmentation model needed for '--reconstructor moving_obj'
    #     flow_model: str = "raft" # default None, flow model needed for '--reconstructor moving_obj'
    #     classes_to_remove: list = ["car", "person"] # list of semantic classes to ignore completely, needed for '--backprojector semantic'
    #     cam_id: int = None  # default None, camera id for KITTI360 datasets, only relevant if '--dataset kitti360' is given
    #     seq_id: int = None #  default None, sequence id for KITTI360 datasets, only relevant if '--dataset kitti360' is given
    #     debug: bool = False # flag for debug
    # args = DebugArgs()

    class DebugArgs(NamedTuple):
        root_dir: str = "/usr/stud/kaa/data/root/ds_combined"
        recon_name: str = None
        reconstructor: str = "simple" # options: "simple" or "moving_obj"
        backprojector: str = "simple" # options: "simple"
        dataset: str = "combined_colmap" # either "colmap", "combined_colmap", "kitti360", or "custom"
        pose_path: str = None # default is None, only needed for custom dataset, can overwrite it for colmap datasets
        pose_scale: float = 6.10 # default is 1.0
        depth_scale: float = None # default is None, scales all depths by constant value
        scales_and_shifts_path: str = None # default is None, scales and shifts depths individually based on the txt file pointed to by this path
        intrinsics: list = [535.045,  535.045, 512, 288] # input as '--intrinsics fx fy cx cy'
        target_size: tuple = () # by default is empty tuple, it means the original size will be used
        dropout: float = 0.99 # random dropout probability of points
        max_d: float = 50.0 # maximum depth threshold for backprojection
        batch_size: int = 2 # batch size used for reconstruction, larger values are faster but needs more GPU memory
        clean_pointcloud: bool = True # flag to whether use statistical outlier removal
        depth_model: str = "precomputed" # available options "metric3d_vit", "precomputed", and "kitti360" for ground truth depths (only works when "--dataset kitti360")
        normal_model: str = "precomputed" # available options "metric3d_vit", "precomputed"
        seg_model: str = "segformer" # default None, segmentation model needed for '--backprojector semantic' and '--reconstructor moving_obj'
        ins_seg_model: str = None # default None, instance segmentati n modefd for '--reconstructor moving_obj'
        flow_model: str = None # default None, flow model needed for '--reconstructor moving_obj'
        classes_to_remove: list = ["car", "person", "truck"] # list of semantic classes to ignore completely, needed for '--backprojector semantic'
        cam_id: int = None  # default None, camera id for KITTI360 datasets, only relevant if '--dataset kitti360' is given
        seq_id: int = None #  default None, sequence id for KITTI360 datasets, only relevant if '--dataset kitti360' is given
        debug: bool = False # flag for debug
    args = DebugArgs()

    # parser = argparse.ArgumentParser(description="Main script to create dense pointclouds by reprojecting depth predictions using aligned poses.")

    # parser.add_argument("--root_dir", type=str, default="/usr/stud/kaa/data/root/ds_combined", help="Root directory")
    # parser.add_argument("--recon_name", type=str, default=None, help="Name of the reconstruction")
    # parser.add_argument("--reconstructor", type=str, default="simple", choices=["simple", "moving_obj"], help="Reconstructor type")
    # parser.add_argument("--backprojector", type=str, default="simple", choices=["simple"], help="Backprojector type")
    # parser.add_argument("--dataset", type=str, default="combined_colmap", choices=["colmap", "combined_colmap", "kitti360", "custom"], help="Dataset type")
    # parser.add_argument("--pose_path", type=str, default=None, help="Path to pose file")
    # parser.add_argument("--pose_scale", type=float, default=6.10, help="Pose scale to multiply the poses")
    # parser.add_argument("--depth_scale", type=float, default=None, help="Depth scale to multiply the depths. Do not give at the same time as '--scales_and_shifts_path'")
    # parser.add_argument("--scales_and_shifts_path", type=str, default=None, help="Path to scales and shifts file. Do not give at the same time as '--depth_scale'")
    # parser.add_argument("--intrinsics", type=float, nargs=4, default=[535.045, 535.045, 512, 288], help="Intrinsics [fx, fy, cx, cy]")
    # parser.add_argument("--target_size", type=tuple, default=(), help="Target size to resize the images")
    # parser.add_argument("--dropout", type=float, default=0.95, help="Random dropout probability of points in each backprojection iteration.")
    # parser.add_argument("--max_d", type=float, default=30.0, help="Maximum depth threshold for backprojection")
    # parser.add_argument("--batch_size", type=int, default=4, help="Batch size for reconstruction.")
    # parser.add_argument("--clean_pointcloud", action="store_true", help="Flag to use statistical outlier removal on the final pointcloud. Might cause issues with RAM if your point clouds are too large")
    # parser.add_argument("--depth_model", type=str, choices=["metric3d_vit", "precomputed", "kitti360"], help="Depth model type")
    # parser.add_argument("--normal_model", type=str, choices=["metric3d_vit", "precomputed"], help="Normal model type")
    # parser.add_argument("--seg_model", type=str, default=None, choices=["segformer"], help="Segmentation model type, needed if you want to remove all moveable objects, inject semantic information to the point cloud, and '--reconstructor moving_obj'")
    # parser.add_argument("--ins_seg_model", type=str, default=None, choices=["mask_rcnn], help="Instance segmentation model type, needed for '--reconstructor moving_obj'")
    # parser.add_argument("--flow_model", type=str, default=None, choices=["raft"] help="Flow model type, needed for '--reconstructor moving_obj'")
    # parser.add_argument("--classes_to_remove", type=list, default=["car", "person", "truck"], help="List of semantic classes to remove completely from the reconstruction")
    # parser.add_argument("--cam_id", type=int, default=None, help="Camera id, only relevant for KITTI360 datasets")
    # parser.add_argument("--seq_id", type=int, default=None, help="Sequence ID, only relevant for KITTI360 datasets")
    # parser.add_argument("--debug", action="store_true", help="Debug flag")

    main(args)
