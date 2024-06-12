import json
import sys
import argparse
import logging
from pathlib import Path

import cv2

from modules.io.utils import ask_to_clear_dir
from configs.data import PADDED_IMG_NAME_LENGTH

def main(args):
    video_path = Path(args.video_path) # expected to be in '/.../root_dir/data_dir/video.mp4'
    data_dir = video_path.parent
    output_dir =  data_dir / Path("rgb")
    file_format = args.file_format
    resize_factor = args.resize_factor

    output_dir.mkdir(parents=True, exist_ok=True)
    should_continue = ask_to_clear_dir(output_dir) # check if output_dir empty

    # Setup logging
    log_path = data_dir.absolute() / Path("log_video.txt")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open the log file in write mode to clear it
    with open(log_path, 'w'):
        pass

    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(levelname)s - %(message)s')
    logging.info(f"Arguments: \n{json.dumps(vars(args), indent=4)}")

    if not should_continue:
        logging.warning(f"Output directory at '{str(output_dir)}' is not empty and you chose to not continue. Aborting frame extraction.")
        return -1

    if not (file_format == "jpg"  or file_format == "png"):
        raise ValueError("Image format not recognized. Please use 'png' or 'jpg' as input to extract_frames().")

    # Code heavily inspired from the stack overflow answer: https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames
    vidcap = cv2.VideoCapture(str(video_path))
    success, image = vidcap.read()
    if success:
        logging.info(f"Original size of your images: {image.shape}")
        image = cv2.resize(
            image,
            dsize = None,
            fx = 1/ resize_factor,
            fy = 1 / resize_factor,
            interpolation=cv2.INTER_AREA
            )
        logging.info(f"Size of your resized images: {image.shape}")
    count = 0

    while success:
        sys.stdout.write("\r")
        sys.stdout.write(f"Reading frame {count}...")
        sys.stdout.flush()

        file_path = output_dir / Path(f"{count:0{PADDED_IMG_NAME_LENGTH}}.{file_format}")
        cv2.imwrite(str(file_path), image)
        success, image = vidcap.read()
        if success:
            image = cv2.resize(
                image,
                dsize = None,
                fx = 1/ resize_factor,
                fy = 1 / resize_factor,
                interpolation=cv2.INTER_AREA
                )
        count += 1
    sys.stdout.write("\n")


    logging.info(f"Read {count} frames in total. Finished extracting frames from video!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from an mp4 video either as png or jpg images and save them")
    parser.add_argument("--video_path", type=str, required=True,  help="Path to the mp4 file")
    parser.add_argument("--file_format", type=str, default="png", help="Image format to save extracted frames, either 'png' or 'jpg'")
    parser.add_argument("--resize_factor", type=float, default=1.0, help="The factor to by which downscale the images. Useful if the image resolution is too high")
    args = parser.parse_args()

    main(args)