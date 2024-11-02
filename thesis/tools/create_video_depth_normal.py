"""
A quick script to read precomputed depth and normal predictions and create a video of the stichted results. The depths are normalized accross the whole sequence
"""
import argparse
from pathlib import Path

import moviepy
import numpy as np
from PIL import Image
from tqdm import tqdm
from matplotlib import cm

from modules.core.visualization import make_vid


# TODO implement proper argument parsing and add usage instructions
if __name__ == "__main__":
    output_name = "ds01"
    fps = 45

    # depths and normals are expected to be in "depths/arrays" and "normals/arrays"
    data_path = "/usr/stud/kaa/data/root/ds01/data"
    data_path = Path(data_path)

    rgb_dir = data_path / "rgb"
    depth_dir = data_path / "depths" / "arrays"
    normal_dir  = data_path / "normals" / "arrays"

    # To normalize the depths accross all the frames
    min_d = np.inf
    min_id = ""
    max_d = 0.0
    max_id = ""

    # tuples of (path, array)
    depth_tuples = []
    print("Loading depths...")
    for p in tqdm(depth_dir.iterdir()):
        if p.is_file() and p.suffix == ".npy":
            depth = np.load(p)

            if depth.max() > max_d:
                max_d = depth.max()
                max_id = p
            if depth.min() < min_d:
                min_d = depth.min()
                min_id = p

            depth_tuples.append((p, depth))

    # Sort the frames
    depth_tuples = sorted(depth_tuples, key=lambda x: int(x[0].stem))

    print(f"Maximum depth value found at {max_id.name} and is {max_d}")
    print(f"Minimum depth value found at {min_id.name} and is {min_d}")

    # Stitch the images
    images = []
    print("Stitching the images...")
    for depth_path, depth in tqdm(depth_tuples):
        file_name = depth_path.stem
        rgb_path = rgb_dir / (file_name + ".png")
        normal_path = normal_dir / (file_name + ".npy")

        inv_depth = 1 / (depth)
        inv_max = 1 / (min_d)
        inv_min = 1 / (max_d)
        inv_depth = (inv_depth - inv_min) / (inv_max - inv_min) # such that the inverse depths are between 0..1
        inv_depth_img = Image.fromarray(np.uint8(cm.viridis(inv_depth) * 255)).convert('RGB')

        rgb = Image.open(rgb_path)

        normal = np.load(normal_path).astype(np.float32)
        normal = (normal - normal.min()) / (normal.max() - normal.min())
        normal = normal.transpose(1,2,0)
        normal_img = Image.fromarray(np.uint8(normal * 255)).convert('RGB')

        width, height = rgb.size
        merged = Image.new("RGB", (3*width, height))
        merged.paste(im=rgb, box=(0, 0))
        merged.paste(im=inv_depth_img, box=(width, 0))
        merged.paste(im=normal_img, box=(2*width, 0))

        images.append(merged)

    # Create video
    print("Creating a video...")
    if output_name!="":
        output_name+="_"
    make_vid(images, fps=fps, output_path=f"{output_name}pred_video.mp4")

    print("Done!")










