import argparse
from pathlib import Path

from configs.data import OUTPUT_DIR
from modules.io.video import extract_frames
from modules.io.utils import ask_to_clear_dir

def main(args):
    video_path = Path(args.video_path)
    output_dir = args.output_dir
    file_format = args.file_format

    if output_dir == "":
        output_dir = video_path.parent / Path("images")

    should_continue = ask_to_clear_dir(output_dir)

    if should_continue:
        extract_frames(video_path, output_dir, file_format)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from an mp4 video either as png or jpg images and save them")
    parser.add_argument("--video_path", type=str, required=True,  help="Path to the mp4 file")
    parser.add_argument("--output_dir", type=str, default="", help="Path to the directory where the output images and the timestamps will be saved")
    parser.add_argument("--file_format", type=str, default="png", help="Image format to save extracted frames, either 'png' or 'jpg'")

    args = parser.parse_args()

    main(args)