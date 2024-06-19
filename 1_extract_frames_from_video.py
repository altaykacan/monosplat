import json
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import cv2

from modules.core.utils import compute_target_intrinsics, resize_image_torch
from modules.io.utils import ask_to_clear_dir
from configs.data import PADDED_IMG_NAME_LENGTH

def main(args):
    video_path = Path(args.video_path) # expected to be in '/.../root_dir/data_dir/video.mp4'
    data_dir = video_path.parent
    output_dir =  data_dir / Path("rgb")
    file_format = args.file_format
    target_size = tuple(args.target_size) if args.target_size is not None else ()
    intrinsics = tuple(args.intrinsics) if args.intrinsics is not None else ()

    output_dir.mkdir(parents=True, exist_ok=True)
    should_continue = ask_to_clear_dir(output_dir) # check if output_dir empty

    # Setup logging
    log_path = data_dir.absolute() / Path("log_video.txt")
    log_time = datetime.now().strftime('%Y-%m-%d:%H-%M-%S')
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'w'): # to clear existing logs
        pass
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(levelname)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(f"Log file for '1_extract_frames_from_video.py', created at (year-month-day:hour-minute-second): {log_time}")
    logging.info(f"Arguments: \n{json.dumps(vars(args), indent=4)}")

    if not should_continue:
        logging.warning(f"Output directory at '{str(output_dir)}' is not empty and you chose to not continue. Aborting frame extraction.")
        return -1

    if not (file_format == "jpg"  or file_format == "png"):
        raise ValueError("Image format not recognized. Please use 'png' or 'jpg' as input to extract_frames().")

    # Code heavily inspired from the stack overflow answer: https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames
    vidcap = cv2.VideoCapture(str(video_path))
    success, image = vidcap.read()
    if success and target_size != ():
        logging.info(f"Original size of your images: {image.shape}")
        logging.info(f"Resizing all your images to {target_size} (H, W)")

        size = (image.shape[0], image.shape[1]) # image is [H, W, C]
        intrinsics = (100,100,100,100) if intrinsics == () else intrinsics # dummy values if not given
        target_intrinsics, crop_box = compute_target_intrinsics(intrinsics, size, target_size)
        image = torch.tensor(np.array(image / 255, dtype=float))
        image = image.permute(2, 0, 1) # put channel dim first, now [C, H, W]
        image = resize_image_torch(image, target_size, crop_box)
        image = image.permute(1, 2, 0) # put channel dim last, now [H, W, C]
        image = (image * 255).to(torch.uint8).numpy()

        logging.info(f"Size of your resized images: {image.shape}")
    count = 0

    while success:
        sys.stdout.write("\r")
        sys.stdout.write(f"Reading frame {count}...")
        sys.stdout.flush()

        file_path = output_dir / Path(f"{count:0{PADDED_IMG_NAME_LENGTH}}.{file_format}")
        cv2.imwrite(str(file_path), image)
        success, image = vidcap.read()
        if success and target_size != ():
            size = (image.shape[0], image.shape[1]) # image is [H, W, C]
            intrinsics = (100,100,100,100) if intrinsics == () else intrinsics # dummy values if not given
            target_intrinsics, crop_box = compute_target_intrinsics(intrinsics, size, target_size)
            image = torch.tensor(np.array(image / 255, dtype=float))
            image = image.permute(2, 0, 1) # put channel dim first, now [C, H, W]
            image = resize_image_torch(image, target_size, crop_box)
            image = image.permute(1, 2, 0) # put channel dim last, now [H, W, C]
            image = (image * 255).to(torch.uint8).numpy()
            count += 1
    sys.stdout.write("\n")

    if args.target_size is not None:
        logging.info(f"Resized all your images to {target_size} (H, W)")
    if args.intrinsics is not None:
        logging.info(f"Your initial intrinsics were: {args.intrinsics}. After resizing to {target_size} from {size}, the new intrinsics would be {target_intrinsics}.")
    logging.info(f"Read {count} frames in total. Finished extracting frames from video!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from an mp4 video either as png or jpg images and save them")
    parser.add_argument("--video_path", "-v", type=str, required=True,  help="Path to the mp4 file")
    parser.add_argument("--file_format", type=str, default="png", help="Image format to save extracted frames, either 'png' or 'jpg'")
    parser.add_argument("--target_size", type=int, nargs=2, default=None, help="Optional target size to resize the extracted images to. Specify as '--target_size H W'")
    parser.add_argument("--intrinsics", type=float, nargs=4, default=None, help="The initial intrinsics (if you already have them) as a list, i.e. '--intrinsics fx fy cx cy'. The values are not needed but new intrinsic values are saved in the log if you specify target size")
    args = parser.parse_args()

    main(args)
