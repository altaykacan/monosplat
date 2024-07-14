import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

from configs.data import PADDED_IMG_NAME_LENGTH
from modules.core.reconstructors import ReconstructorFactory
from modules.core.backprojection import Backprojector, DepthBasedDropoutBackprojector
from modules.core.models import RAFT
from modules.depth.models import Metric3Dv2, KITTI360DepthModel, PrecomputedDepthModel
from modules.segmentation.models import SegFormer, MaskRCNN
from modules.io.datasets import CustomDataset, ColmapDataset, CombinedColmapDataset, KITTI360Dataset
from modules.io.utils import find_latest_number_in_dir, create_intrinsics_txt, create_poses_for_3dgs

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
    use_every_nth = args.use_every_nth
    max_d = args.max_d
    batch_size = args.batch_size
    clean_pointcloud = args.clean_pointcloud
    downsample_pointcloud_voxel_size = args.downsample_pointcloud_voxel_size
    add_skydome = args.add_skydome
    init_cloud_path = args.init_cloud_path
    depth_model_type = args.depth_model
    normal_model_type = args.normal_model
    seg_model_type = "segformer" if (args.seg_model is None and args.reconstructor == "moving_obj") else args.seg_model
    ins_seg_model_type = "mask_rcnn" if (args.ins_seg_model is None and args.reconstructor == "moving_obj") else args.ins_seg_model
    flow_model_type = "raft" if (args.flow_model is None and args.reconstructor == "moving_obj") else args.flow_model
    classes_to_remove = args.classes_to_remove
    cam_id = 0 if (args.cam_id is None and args.dataset == "kitti360") else args.cam_id
    seq_id = 0 if (args.seq_id is None and args.dataset == "kitti360") else args.seq_id
    padded_img_name_length = args.padded_img_name_length
    start = args.start
    end = args.end
    dropout_prob_min = args.dropout_prob_min
    dropout_coeff = args.dropout_coeff
    debug = args.debug

    # TODO can probably abstract away this script preparation (logging and root_dir checking)
    if not root_dir.exists():
        raise RuntimeError(f"Your root_dir at '{str(root_dir)}' does not exist! Please make sure you give the right path.")

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
    logging.info(f"Arguments: \n{json.dumps(vars(args), indent=4)}")
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
            scales_and_shifts_path=scales_and_shifts_path,
            padded_img_name_length=padded_img_name_length,
            start=start,
            end=end,
            )
        if pose_path is not None:
            dataset.pose_path = pose_path
            logging.info(f"Overwriting COLMAP pose path with user specified pose path {str(pose_path)}")
        else:
            logging.info(f"No pose path provided to script. Using the default COLMAP pose {str(dataset.pose_path)}")

    if dataset_type == "combined_colmap":
        # Due to frame id naming convention specifying start and end doesn't work too well here
        dataset = CombinedColmapDataset(
            colmap_dir,
            pose_scale=pose_scale,
            target_size=target_size,
            orig_intrinsics=intrinsics,
            depth_dir=depth_dir,
            depth_scale=depth_scale,
            scales_and_shifts_path=scales_and_shifts_path,
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
            scales_and_shifts_path=scales_and_shifts_path,
            padded_img_name_length=padded_img_name_length,
            start=start,
            end=end,
            )
    elif dataset_type == "kitti360":
        dataset = KITTI360Dataset(seq_id, cam_id, pose_scale, target_size, start=start, end=end)
        pose_path = dataset.pose_path # GT poses

    # Get backprojector
    if backprojector_type == "simple":
        backprojector_cfg = {"dropout": dropout, "max_d": max_d}
        backprojector = Backprojector(backprojector_cfg, dataset.intrinsics)
    elif backprojector_type == "depth_dropout":
        backprojector_cfg = {"max_d": max_d,"dropout_prob_min": dropout_prob_min, "dropout_coeff": dropout_coeff}
        backprojector = DepthBasedDropoutBackprojector(backprojector_cfg, dataset.intrinsics)
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

    # Create reconstructor and run it, essential components are used to initalize the factory
    factory = ReconstructorFactory(dataset, backprojector, depth_model)
    recon_config = {
        "batch_size": batch_size,
        "use_every_nth": use_every_nth,
        "output_dir": recon_dir,
        "clean_pointcloud": clean_pointcloud,
        "flow_model_type": flow_model_type,
        "ins_seg_model_type": ins_seg_model_type,
        "seg_model_type": seg_model_type,
        "classes_to_remove": classes_to_remove,
        "normal_model_type": normal_model_type,
        "downsample_pointcloud_voxel_size": downsample_pointcloud_voxel_size,
        "add_skydome": add_skydome,
        "init_cloud_path": init_cloud_path,
        }
    reconstructor = factory.get_reconstructor(reconstructor_type, recon_config)
    reconstructor.run()

    # Write poses and intrinsics to use 3DGS later on
    create_intrinsics_txt(recon_dir, dataset)
    create_poses_for_3dgs(recon_dir, dataset)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Main script to create dense pointclouds by reprojecting depth predictions using aligned poses.")

    parser.add_argument("--root_dir", "-r", type=str, required=True, help="Root directory of your dataset. Outputs will be saved at 'root_dir/reconstructions/...'")
    parser.add_argument("--recon_name", "-n", type=str, default=None, help="Name of the reconstruction. The results will be saved at 'root_dir/reconsturctions/xx_recon_name' where 'xx' is the numbering of the reconstruction.")
    parser.add_argument("--reconstructor", type=str, default="simple", choices=["simple", "moving_obj"], help="Reconstructor type")
    parser.add_argument("--backprojector", type=str, default="simple", choices=["simple", "depth_dropout"], help="Backprojector type")
    parser.add_argument("--dataset", type=str, default="colmap", choices=["colmap", "combined_colmap", "kitti360", "custom"], help="Dataset type")
    parser.add_argument("--pose_path", "-p", type=str, default=None, help="Absolute path to pose file")
    parser.add_argument("--pose_scale", type=float, default=1.0, help="Pose scale to multiply the poses")
    parser.add_argument("--depth_scale", type=float, default=None, help="Depth scale to multiply the depths. Do not give at the same time as '--scales_and_shifts_path'")
    parser.add_argument("--scales_and_shifts_path", type=str, default=None, help="Absolute path to scales and shifts file. Do not give at the same time as '--depth_scale'")
    parser.add_argument("--intrinsics", type=float, nargs=4, required=True, help="Intrinsics [fx, fy, cx, cy]")
    parser.add_argument("--target_size", type=tuple, default=(), help="Target size to resize the images")
    parser.add_argument("--dropout", type=float, default=0.98, help="Random dropout probability of points in each backprojection iteration.")
    parser.add_argument("--max_d", type=float, default=30.0, help="Maximum depth threshold for backprojection")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for reconstruction.")
    parser.add_argument("--use_every_nth", type=int, default=3, help="Every nth image will be used for reconstruction.")
    parser.add_argument("--clean_pointcloud", action="store_true", help="Flag to use statistical outlier removal on the final pointcloud. Might cause issues with RAM if your point clouds are too large")
    parser.add_argument("--downsample_pointcloud_voxel_size", type=float, default=None, help="Voxel size in metric scale to use if you want to downsample the point cloud. Useful for sparsity experiments with 3D gaussian splatting. Will be ignored if None")
    parser.add_argument("--add_skydome", action="store_true", help="Flag to whether add a skydome around the dense cloud")
    parser.add_argument("--init_cloud_path", type=str, default=None, help="Path to optional initial cloud to add. WARNING: Make sure the initial cloud and the dense cloud has the same scale. They will simply be added together without any scaling (e.g. if you are adding the COLMAP cloud, make sure to give '--depth_scale' instead of '--pose_scale' because the latter scales up the poses which have the same scale as the SfM cloud you are adding).")
    parser.add_argument("--depth_model", type=str, choices=["metric3d_vit", "precomputed", "kitti360"], help="Depth model type")
    parser.add_argument("--normal_model", type=str, choices=["metric3d_vit", "precomputed"], help="Normal model type")
    parser.add_argument("--seg_model", type=str, default=None, choices=["segformer", "precomputed"], help="Segmentation model type, needed if you want to remove all moveable objects, inject semantic information to the point cloud, and '--reconstructor moving_obj'")
    parser.add_argument("--ins_seg_model", type=str, default=None, choices=["mask_rcnn"], help="Instance segmentation model type, needed for '--reconstructor moving_obj'")
    parser.add_argument("--flow_model", type=str, default=None, choices=["raft"], help="Flow model type, needed for '--reconstructor moving_obj'")
    parser.add_argument("--classes_to_remove", type=list, default=["car", "person", "truck", "sky"], help="List of semantic classes to remove completely from the reconstruction. The classes are removed if and only if you specify a segmentation or instance segmentation model with '--seg_model' or '--ins_seg_model'")
    parser.add_argument("--cam_id", type=int, default=None, help="Camera id, only relevant for KITTI360 datasets")
    parser.add_argument("--seq_id", type=int, default=None, help="Sequence id, only relevant for KITTI360 datasets")
    parser.add_argument("--debug", action="store_true", help="Debug flag")
    parser.add_argument("--padded_img_name_length", type=int, default=PADDED_IMG_NAME_LENGTH, help="Total image name length. The integer frame id will be prepended with zeros until it reaches this length. For colmap datasets use 5, for KITTI use 6, for KITTI360 use 10")
    parser.add_argument("--start", type=int, default=0, help="Frame id to start the sequence")
    parser.add_argument("--end", type=int, default=-1, help="Frame id of the last frame in the sequence. Set as -1 to include all frames")
    parser.add_argument("--dropout_prob_min", type=float, default=0.7, help="Minimum dropout probability applied to the smalles depths when doing depth-based dropout")
    parser.add_argument("--dropout_coeff", type=float, default=0.4, help="Dropout coefficient to multiply the tanh function by when doing depth-based dropout")

    args = parser.parse_args()
    main(args)
