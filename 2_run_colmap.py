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
"""
Copied and adapted from the original 3D Gaussian Splatting repository for
running COLMAP (https://github.com/graphdeco-inria/gaussian-splatting).
Used to run COLMAP for pose prediction and to get an estimate of camera
calibration parameters if they do not exist.
"""
import json
import os
import logging
import shutil
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser

from modules.io.utils import ask_to_clear_dir, create_rgb_txt

def main(args) -> None:
    # Do some initial parsing of the arguments
    source_p = Path(args.source_path)
    image_dir = source_p / Path("../../data/rgb") # according to assumed directory structure
    image_dir = image_dir.resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Can't find '{str(image_dir)}'. Your source path '{args.source_path}' probably isn't in the right place or you need to extract frames from your video!")

    source_p.mkdir(exist_ok=True, parents=True)
    if not args.continue_last_run:
        should_continue = ask_to_clear_dir(source_p)
        if not should_continue:
            return -1

    if args.fix_intrinsics and args.init_intrinsics is None:
        raise ValueError("You specified '--fix_intrinsics' but did not provide any initial intrinsics with '--init_intrinsics'. You need initial intrinsics to fix them!")

    # Setup logging
    log_time = datetime.now().strftime('%Y-%m-%d:%H-%M-%S')
    log_path = source_p.absolute() / Path("log_colmap.txt")
    with open(log_path, 'w'): # to clear existing log files
        pass
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(levelname)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(f"Log file for '2_run_colmap.py', created at (year-month-day:hour-minute-second): {log_time}")
    logging.info(f"Arguments: \n{json.dumps(vars(args), indent=4)}")
    logging.info(f"Using the image path: {str(image_dir.absolute())}")

    colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
    magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
    use_gpu = 1 if not args.no_gpu else 0
    if not args.skip_matching:
        os.makedirs(args.source_path + "/distorted/sparse", exist_ok=True)

        ## Feature extraction, image path is changed to follow our directory structure - Altay
        feat_extraction_cmd = colmap_command + " feature_extractor "\
            "--database_path " + args.source_path + "/distorted/database.db \
            --image_path " + str(image_dir.absolute()) + " \
            --ImageReader.single_camera 1 \
            --ImageReader.camera_model " + args.camera + " \
            --SiftExtraction.use_gpu " + str(use_gpu)
        if args.init_intrinsics is not None:
            fx, fy, cx, cy = args.init_intrinsics
            logging.info(f"Using initial guesses for the camera intrinsics (fx, fy, cx, cy): ({fx}, {fy}, {cx}, {cy})")
            if "simple" in args.camera.lower():
                intrinsics_arg = f"'{fx}, {cx}, {cy}'"
            else:
                intrinsics_arg =f"'{fx}, {fy}, {cx}, {cy}'"

            feat_extraction_cmd += " " + "--ImageReader.camera_params " + intrinsics_arg

        logging.info(f"Command used to run feature extraction: \n {feat_extraction_cmd}")
        exit_code = os.system(feat_extraction_cmd)
        if exit_code != 0:
            logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
            exit(exit_code)

        ## Feature matching either sequential (faster) or exhaustive matching (better performance)
        if args.use_sequential_matcher:
            logging.info("Using sequential matcher...")
            feat_matching_cmd = colmap_command + " sequential_matcher \
            --database_path " + args.source_path + "/distorted/database.db \
            --SiftMatching.use_gpu " + str(use_gpu)

            if args.vocab_tree_path is not None:
                logging.info(f"Using vocabulary tree for sequential matching and loop detection (path: '{args.vocab_tree_path}')...")
                vocab_tree_path = Path(args.vocab_tree_path)
                if not vocab_tree_path.exists():
                    print(f"The vocabulary tree path you provided ({str(vocab_tree_path)}) does not exist. Please check that directory!")

                feat_matching_cmd = feat_matching_cmd \
                                + " " + "--SequentialMatching.loop_detection 1" \
                                + " " + "--SequentialMatching.vocab_tree_path " + str(vocab_tree_path)
        else:
            feat_matching_cmd = colmap_command + " exhaustive_matcher \
                --database_path " + args.source_path + "/distorted/database.db \
                --SiftMatching.use_gpu " + str(use_gpu)

        logging.info(f"Command used to run feature matching: \n {feat_matching_cmd}")
        exit_code = os.system(feat_matching_cmd)
        if exit_code != 0:
            logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
            exit(exit_code)

        ### Bundle adjustment
        # The default Mapper tolerance is unnecessarily large,
        # decreasing it speeds up bundle adjustment steps.
        mapper_cmd = (colmap_command + " mapper \
            --database_path " + args.source_path + "/distorted/database.db \
            --image_path "  + str(image_dir.absolute()) + " \
            --output_path "  + args.source_path + "/distorted/sparse \
            --Mapper.ba_global_function_tolerance=0.000001")

        if args.fix_intrinsics:
            logging.info("The flag '--fix_intrinsics' is given. Not optimizing given intrinsics...")
            mapper_cmd += " " + "--Mapper.ba_refine_focal_length 0"

        logging.info(f"Command used to run mapping: \n {mapper_cmd}")
        exit_code = os.system(mapper_cmd)
        if exit_code != 0:
            logging.error(f"Mapper failed with code {exit_code}. Exiting.")
            exit(exit_code)

    ### Image undistortion
    ## We need to undistort our images into ideal pinhole intrinsics.
    img_undist_cmd = (colmap_command + " image_undistorter \
        --image_path " + str(image_dir.absolute()) + " \
        --input_path " + args.source_path + "/distorted/sparse/0 \
        --output_path " + args.source_path + "\
        --output_type COLMAP")
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

    files = os.listdir(args.source_path + "/sparse")
    os.makedirs(args.source_path + "/sparse/0", exist_ok=True)
    # Copy each file from the source directory to the destination directory
    for file in files:
        if file == '0':
            continue
        source_file = os.path.join(args.source_path, "sparse", file)
        destination_file = os.path.join(args.source_path, "sparse", "0", file)
        shutil.move(source_file, destination_file)

    if(args.resize):
        print("Copying and resizing...")

        # Resize images.
        os.makedirs(args.source_path + "/images_2", exist_ok=True)
        os.makedirs(args.source_path + "/images_4", exist_ok=True)
        os.makedirs(args.source_path + "/images_8", exist_ok=True)
        # Get the list of files in the source directory
        files = os.listdir(str(image_dir.absolute()))
        # Copy each file from the source directory to the destination directory
        for file in files:
            source_file = os.path.join(str(image_dir.absolute()), file)

            destination_file = os.path.join(args.source_path, "images_2", file)
            shutil.copy2(source_file, destination_file)
            exit_code = os.system(magick_command + " mogrify -resize 50% " + destination_file)
            if exit_code != 0:
                logging.error(f"50% resize failed with code {exit_code}. Exiting.")
                exit(exit_code)

            destination_file = os.path.join(args.source_path, "images_4", file)
            shutil.copy2(source_file, destination_file)
            exit_code = os.system(magick_command + " mogrify -resize 25% " + destination_file)
            if exit_code != 0:
                logging.error(f"25% resize failed with code {exit_code}. Exiting.")
                exit(exit_code)

            destination_file = os.path.join(args.source_path, "images_8", file)
            shutil.copy2(source_file, destination_file)
            exit_code = os.system(magick_command + " mogrify -resize 12.5% " + destination_file)
            if exit_code != 0:
                logging.error(f"12.5% resize failed with code {exit_code}. Exiting.")
                exit(exit_code)

    converter_cmd = f"{colmap_command} model_converter --input_path {args.source_path}/sparse/0 --output_path {args.source_path}/sparse/0 --output_type TXT"
    exit_code = os.system(converter_cmd)
    if exit_code != 0:
        logging.error(f"Converting .bin files to .txt files failed with code {exit_code}. Exiting.")
        exit(exit_code)

    # Create rgb.txt for running SLAM
    data_dir = image_dir.parent
    if not (data_dir / Path("rgb.txt")).exists():
        logging.info(f"Creating rgb.txt for running SLAM at {str(data_dir)}...")
        create_rgb_txt(data_dir, image_dir)
    else:
        logging.info(f"Found existing rgb.txt for running SLAM at {str(data_dir)}. Not overwriting it.")


    print("Done.")

if __name__=="__main__":
    # This Python script is based on the shell converter script provided in the MipNerF 360 repository.
    parser = ArgumentParser("Colmap converter")
    parser.add_argument("--no_gpu", action='store_true')
    parser.add_argument("--skip_matching", action='store_true')
    parser.add_argument("--source_path", "-s", required=True, type=str, help="The path to where your COLMAP dataset is")
    parser.add_argument("--camera", default="SIMPLE_PINHOLE", type=str, help="Camera model to use with COLMAP, suitable options are 'OPENCV', 'PINHOLE', and 'SIMPLE_PINHOLE'. Check COLMAP docs for more info.")
    parser.add_argument("--colmap_executable", default="", type=str)
    parser.add_argument("--resize", action="store_true")
    parser.add_argument("--magick_executable", default="", type=str)
    parser.add_argument("--init_intrinsics", default=None, nargs=4, type=float, help="Initial guess of the camera intrinsics as (fx, fy, cx, cy) if they exist. If you choose 'SIMPLE_PINHOLE' as your camera only the first focal length will be chosen (fx and fy are the same). Provide input as '--init_intrinsics fx fy cx cy'")
    parser.add_argument("--continue_last_run", action="store_true", help="Flag to specify whether the last COLMAP run should be continued. If it is not provided, you will be prompted to delete existing files or abort the script.")
    parser.add_argument("--use_sequential_matcher", action="store_true", help="Flag to specify whether the sequential matcher instead of the exhaustive matcher is used for COLMAP. This is much faster for frames extracted from a video (single video) and has loop detection support. You need to provide '--vocab_tree_path' to use loop closure.")
    parser.add_argument("--vocab_tree_path", type=str, default=None, help="The path to the used vocabulary tree. Only relevant if you provide '--use_sequential_matcher' flag.")
    parser.add_argument("--fix_intrinsics", action="store_true", help="Flag to whether fix the intrinsics to a constant value or not. If you provide this flag, COLMAP will not try to optimize the intrinsics you provide with '--init_intrinsics'. This is useful for getting results with benchmark datasets.")

    args = parser.parse_args()
    main(args)

