import sys
from typing import Union
from pathlib import Path

import cv2

from configs.data import PADDED_IMG_NAME_LENGTH


def extract_frames(video_path: Union[Path, str], output_dir: Union[Path, str], file_format: str = "png"):
    if not (file_format == "jpg"  or file_format == "png"):
        raise ValueError("Image format not recognized. Please use 'png' or 'jpg' as input to extract_frames().")

    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Code heavily inspired from the stack overflow answer: https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames
    vidcap = cv2.VideoCapture(str(video_path))
    success, image = vidcap.read()
    count = 0

    while success:
        sys.stdout.write("\r")
        sys.stdout.write(f"Reading frame {count}...")
        sys.stdout.flush()

        file_path = output_dir / Path(f"{count:0{PADDED_IMG_NAME_LENGTH}}.{file_format}")
        cv2.imwrite(str(file_path), image)
        success, image = vidcap.read()
        count += 1
