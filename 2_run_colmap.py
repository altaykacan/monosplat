#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

# Copied from the original 3D Gaussian Splatting repository, the convert.py file
# Used to run COLMAP and get camera calibration parameters - Altay
import json
import os
import logging
import shutil
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser

from modules.io.utils import ask_to_clear_dir

# TODO test this and run it!
def main(args) -> None:
    source_p = Path(args.source_path)
    image_path = source_p / Path("../../data/rgb")
    if not source_p.is_dir():
        raise FileNotFoundError(f"Your source path '{args.source_path}' does not exist!")

    if not args.continue_last_run:
        should_continue = ask_to_clear_dir(source_p)
        if not should_continue:
            return -1

    # # Setup logging
    # log_time = datetime.now().strftime('%Y-%m-%d:%H-%M-%S')
    # log_path = source_p.absolute() / Path("log_colmap.txt")
    # with open(log_path, 'w'): # to clear existing log files
    #     pass
    # logging.basicConfig(filename=log_path, level=logging.INFO, format='%(levelname)s - %(message)s')
    # logging.info(f"Log file for '2_run_colmap.py', created at (year-month-day:hour-minute-second): {log_time}")
    # logging.info(f"Arguments: \n{json.dumps(vars(args), indent=4)}")
    # logging.info(f"Using the image path: {str(image_path.absolute())}")

    colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
    magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
    use_gpu = 1 if not args.no_gpu else 0
    # if not args.skip_matching:
    #     os.makedirs(args.source_path + "/distorted/sparse", exist_ok=True)

    #     ## Feature extraction, image path is changed to follow our directory structure - Altay
    #     if args.init_intrinsics == []:
    #         feat_extracton_cmd = colmap_command + " feature_extractor "\
    #             "--database_path " + args.source_path + "/distorted/database.db \
    #             --image_path " + str(image_path.absolute()) + " \
    #             --ImageReader.single_camera 1 \
    #             --ImageReader.camera_model " + args.camera + " \
    #             --SiftExtraction.use_gpu " + str(use_gpu)
    #     else:
    #         f, cx, cy = args.init_intrinsics
    #         logging.info(f"Using initial guesses for the camera intrinsics (f, cx, cy): ({f}, {cx}, {cy})")
    #         feat_extracton_cmd = colmap_command + " feature_extractor "\
    #             "--database_path " + args.source_path + "/distorted/database.db \
    #             --image_path " + str(image_path.absolute()) + " \
    #             --ImageReader.single_camera 1 \
    #             --ImageReader.camera_model " + args.camera + " \
    #             --ImageReader.camera_params " + f"'{f}, {cx}, {cy}'" + " \
    #             --SiftExtraction.use_gpu " + str(use_gpu)

    #     exit_code = os.system(feat_extracton_cmd)
    #     if exit_code != 0:
    #         logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
    #         exit(exit_code)

    #     ## Feature matching
    #     feat_matching_cmd = colmap_command + " exhaustive_matcher \
    #         --database_path " + args.source_path + "/distorted/database.db \
    #         --SiftMatching.use_gpu " + str(use_gpu)
    #     exit_code = os.system(feat_matching_cmd)
    #     if exit_code != 0:
    #         logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
    #         exit(exit_code)

    #     ### Bundle adjustment
    #     # The default Mapper tolerance is unnecessarily large,
    #     # decreasing it speeds up bundle adjustment steps.
    #     mapper_cmd = (colmap_command + " mapper \
    #         --database_path " + args.source_path + "/distorted/database.db \
    #         --image_path "  + str(image_path.absolute()) + " \
    #         --output_path "  + args.source_path + "/distorted/sparse \
    #         --Mapper.ba_global_function_tolerance=0.000001")
    #     exit_code = os.system(mapper_cmd)
    #     if exit_code != 0:
    #         logging.error(f"Mapper failed with code {exit_code}. Exiting.")
    #         exit(exit_code)

    # ### Image undistortion
    # ## We need to undistort our images into ideal pinhole intrinsics.
    # img_undist_cmd = (colmap_command + " image_undistorter \
    #     --image_path " + str(image_path.absolute()) + " \
    #     --input_path " + args.source_path + "/distorted/sparse/0 \
    #     --output_path " + args.source_path + "\
    #     --output_type COLMAP")
    # exit_code = os.system(img_undist_cmd)
    # if exit_code != 0:
    #     logging.error(f"Mapper failed with code {exit_code}. Exiting.")
    #     exit(exit_code)

    # files = os.listdir(args.source_path + "/sparse")
    # os.makedirs(args.source_path + "/sparse/0", exist_ok=True)
    # # Copy each file from the source directory to the destination directory
    # for file in files:
    #     if file == '0':
    #         continue
    #     source_file = os.path.join(args.source_path, "sparse", file)
    #     destination_file = os.path.join(args.source_path, "sparse", "0", file)
    #     shutil.move(source_file, destination_file)

    # if(args.resize):
    #     print("Copying and resizing...")

    #     # Resize images.
    #     os.makedirs(args.source_path + "/images_2", exist_ok=True)
    #     os.makedirs(args.source_path + "/images_4", exist_ok=True)
    #     os.makedirs(args.source_path + "/images_8", exist_ok=True)
    #     # Get the list of files in the source directory
    #     files = os.listdir(str(image_path.absolute()))
    #     # Copy each file from the source directory to the destination directory
    #     for file in files:
    #         source_file = os.path.join(str(image_path.absolute()), file)

    #         destination_file = os.path.join(args.source_path, "images_2", file)
    #         shutil.copy2(source_file, destination_file)
    #         exit_code = os.system(magick_command + " mogrify -resize 50% " + destination_file)
    #         if exit_code != 0:
    #             logging.error(f"50% resize failed with code {exit_code}. Exiting.")
    #             exit(exit_code)

    #         destination_file = os.path.join(args.source_path, "images_4", file)
    #         shutil.copy2(source_file, destination_file)
    #         exit_code = os.system(magick_command + " mogrify -resize 25% " + destination_file)
    #         if exit_code != 0:
    #             logging.error(f"25% resize failed with code {exit_code}. Exiting.")
    #             exit(exit_code)

    #         destination_file = os.path.join(args.source_path, "images_8", file)
    #         shutil.copy2(source_file, destination_file)
    #         exit_code = os.system(magick_command + " mogrify -resize 12.5% " + destination_file)
    #         if exit_code != 0:
    #             logging.error(f"12.5% resize failed with code {exit_code}. Exiting.")
    #             exit(exit_code)

    converter_cmd = f"{colmap_command} model_converter --input_path {args.source_path}/sparse/0 --output_path {args.source_path}/sparse/0 --output_type TXT"
    exit_code = os.system(converter_cmd)
    if exit_code != 0:
        logging.error(f"Converting .bin files to .txt files failed with code {exit_code}. Exiting.")
        exit(exit_code)
    print("Done.")

if __name__=="__main__":
    # This Python script is based on the shell converter script provided in the MipNerF 360 repository.
    parser = ArgumentParser("Colmap converter")
    parser.add_argument("--no_gpu", action='store_true')
    parser.add_argument("--skip_matching", action='store_true')
    parser.add_argument("--source_path", "-s", required=True, type=str, help="The path to where your COLMAP dataset")
    parser.add_argument("--camera", default="SIMPLE_PINHOLE", type=str, help="Camera model to use with COLMAP, suitable options are 'OPENCV', 'PINHOLE', and 'SIMPLE_PINHOLE'. Check COLMAP docs for more info.")
    parser.add_argument("--colmap_executable", default="", type=str)
    parser.add_argument("--resize", action="store_true")
    parser.add_argument("--magick_executable", default="", type=str)
    parser.add_argument("--init_intrinsics", default=[], nargs=3, type=float, help="Initial guess of the camera intrinsics as (f, cx, cy) if they exist. Assuming a simple pinhole camera model (fx and fy are the same)")
    parser.add_argument("--continue_last_run", action="store_true", help="Flag to specify whether the last COLMAP run should be continued. If it is not provided, you will be prompted to delete existing files or abort the script.")
    args = parser.parse_args()
    main(args)

