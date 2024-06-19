import json
import argparse
import logging
from pathlib import Path
from typing import NamedTuple
from datetime import datetime

import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision.transforms.functional import pil_to_tensor

from configs.data import PADDED_IMG_NAME_LENGTH
from modules.core.utils import grow_bool_mask
from modules.io.utils import ask_to_clear_dir
from modules.depth.models import Metric3Dv2
from modules.segmentation.models import SegFormer
from modules.segmentation.utils import combine_segmentation_masks


def main(args):
    root_dir = Path(args.root_dir)
    model_name = args.model.lower()
    intrinsics = tuple(args.intrinsics) if args.intrinsics is not None else ()
    skip_depth_png = args.skip_depth_png
    skip_mask_depth_png = args.skip_mask_depth_png
    skip_normals = args.skip_normals
    max_depth_png = args.max_depth_png
    scale_factor_depth_png = args.scale_factor_depth_png
    debug = args.debug

    data_dir = root_dir / Path("data")
    image_dir = data_dir / Path("rgb")

    # Setup logging
    log_path = data_dir.absolute() / Path("log_depth_and_normal.txt")
    log_time = datetime.now().strftime('%Y-%m-%d:%H-%M-%S')
    with open(log_path, 'w'): # to clear existing log files
        pass
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(levelname)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(f"Log file for '3_precompute_depth_and_normals.py', created at (year-month-day:hour-minute-second): {log_time}")
    logging.info(f"Arguments: \n{json.dumps(vars(args), indent=4)}")

    if not skip_depth_png and (max_depth_png * scale_factor_depth_png >= 2**16):
        raise ValueError(f"The maximum depth threshold ({max_depth_png}) and the scale factor ({scale_factor_depth_png}) you chose for saving the depths as png images exceeds the largest unsigned integer that can be represented by 16-bits, please lower the scale or the maximum depth!")

    if intrinsics == ():
        raise ValueError(f"Your intrinsics are empty, please specify them as a list: '--intrinsics fx fy cx cy' in the command line.")

    # TODO can implement this with a model factory when we add more models
    # TODO separate the depth and normal model, currently we use metric3Dv2 to get both
    # Initialize pretrained models
    if model_name == "metric3d":
        model = Metric3Dv2(intrinsics, backbone="vit_giant")
    else:
        raise ValueError(f"The provided model name '{model_name}' is not recognized. Please try 'metric3d'.")

    if not skip_depth_png and not skip_mask_depth_png:
        seg_model = SegFormer()
        seg_classes = ["car", "bus", "person", "truck", "bicycle", "motorcycle", "rider"]

    # Paths to save the results
    if debug:
        depth_array_dir = data_dir / Path("depths_debug") / Path("arrays")
        normal_array_dir = data_dir / Path("normals_debug") / Path("arrays")
        depth_png_dir = data_dir / Path("depths_debug") / Path("images")
        normal_png_dir = data_dir / Path("normals_debug") / Path("images")
    else:
        depth_array_dir = data_dir / Path("depths") / Path("arrays")
        normal_array_dir = data_dir / Path("normals") / Path("arrays")
        depth_png_dir = data_dir / Path("depths") / Path("images")
        normal_png_dir = data_dir / Path("normals") / Path("images")

    depth_array_dir.mkdir(parents=True, exist_ok=True)
    normal_array_dir.mkdir(parents=True, exist_ok=True)
    depth_png_dir.mkdir(parents=True, exist_ok=True)
    normal_png_dir.mkdir(parents=True, exist_ok=True)

    # If any of these directories are not empty ask to clear or abort
    if not ask_to_clear_dir(depth_array_dir):
        logging.warning(f"Depth array directory '{str(depth_array_dir)}' is not empty and you chose not to continue. Aborting.")
        return -1
    if not skip_normals and not ask_to_clear_dir(normal_array_dir):
        logging.warning(f"Normal array directory '{str(normal_array_dir)}' is not empty and you chose not to continue. Aborting.")
        return -1
    if not skip_depth_png and not ask_to_clear_dir(depth_png_dir):
        logging.warning(f"Depth png directory '{str(depth_png_dir)}' is not empty and you chose not to continue. Aborting.")
        return -1
    if not skip_normals and not ask_to_clear_dir(normal_png_dir):
        logging.warning(f"Normal png directory '{str(normal_png_dir)}' is not empty and you chose not to continue. Aborting.")
        return -1

    params = [cv2.IMWRITE_PNG_COMPRESSION, 0]  # lossless compression for png
    logging.info(f"Iterating through every path in {str(image_dir)} and predicting depths/normals for each image. This might take a while...")
    for image_path in tqdm(image_dir.iterdir()):
        if image_path.is_file() and image_path.suffix == ".png":
            frame_id = image_path.stem # no extension, has padded zeros
            image = pil_to_tensor(Image.open(image_path))
            pred = model.predict({"images": image[None, :, : ,:], "frame_ids": torch.tensor([float(frame_id)])}) # dummy batch dimension
            depth = pred["depths"].squeeze(0) # [1, H, W]
            depth = depth.squeeze().detach().cpu().float().numpy() # cast to float to save disk memory

            depth_array_path = depth_array_dir / Path(f"{frame_id}.npy")
            normal_array_path = normal_array_dir / Path(f"{frame_id}.npy")
            depth_png_path = depth_png_dir / Path(f"{frame_id}.png")
            normal_png_path = normal_png_dir / Path(f"{frame_id}.png")

            # Save the depth array
            np.save(depth_array_path, depth)

            # Save the depth images if specified
            if not skip_depth_png:
                # Threshold the max depth, important to be able to save as 16-bit png
                depth[depth > max_depth_png] = max_depth_png

                # Use float64 just in case values multiplied by scale are too large
                depth = depth.astype(np.float64)

                # Scale to represent decimal points even after casting to integer
                depth = depth * scale_factor_depth_png
                depth = depth.astype(np.uint16)

                if not skip_mask_depth_png:
                    pred_seg = seg_model.predict({"images": image[None, : ,: ,:], "classes_to_segment": seg_classes})
                    masks = pred_seg["masks_dict"]
                    masks = combine_segmentation_masks(masks)
                    masks = grow_bool_mask(masks, iterations=2) # for robustness
                    masks = masks.squeeze().detach().cpu().numpy()

                    # Set the moveable objects to be REALLY far so RGB-D SLAM can be tricked to ignore it
                    depth[masks] = (2**16 - 1)

                # Save the depth png image as 16-bit unsigned integer monochrome png
                cv2.imwrite(str(depth_png_path), depth, params)

            # Save normal arrays and png images if specified
            if not skip_normals:
                if "normals" not in pred.keys():
                    raise RuntimeError("You specified '--save_normals' but no normal predictions are found. Please check that your GPU supports ViT models if you are using Metric3Dv2 to get normal predictions.")
                normals = pred["normals"].squeeze().detach().cpu().numpy()
                normals_vis = pred["normals_vis"].squeeze().detach().cpu().numpy()

                np.save(normal_array_path, normals)
                cv2.imwrite(str(normal_png_path), normals_vis, params)
    logging.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict and save depth and normal predictions as numpy arrays with the original sizes of the images. Optionally saves depth predictions as scaled 16-bit monochrome png images as well.")
    parser.add_argument("--root_dir", "-r", type=str, help="The root of your data directory.")
    parser.add_argument("--model", "-m", type=str, help="The model to get depth and normal predictions. Currently only the input 'metric3d' is implemented which gives both depth and normal predictions.")
    parser.add_argument("--intrinsics", type=float, nargs=4, help="Camera intrinsics as [fx, fy, cx, cy]. Run '2_run_colmap.py' to get them. If you already have intrinsics, make sure to scale them with the same factor you are resizing your images (dividing image dimensions by 2 -> dividing intrinsic parameters by 2). Provide inputs as '--intrinsics fx fy cx cy'")
    parser.add_argument("--skip_depth_png", action="store_true", help="Flag to skip saving scaled depth images as 16-bit unsigned integer monochrome png images. These images are useful for RGB-D SLAM.")
    parser.add_argument("--skip_mask_depth_png", action="store_true", help="Flag to skip using moveable object masks to set the depths of moveable objects to a very high value when saving depth png images. This is useful to ignore potentially moving objects by setting a hard depth limit on the RGB-D SLAM system")
    parser.add_argument("--skip_normals", action="store_true", help="Flag to skip predicting normals. Use this if your 'model' does not have normal prediction capabilities.")
    parser.add_argument("--scale_factor_depth_png", type=float, default=1000.0, help="Scale factor to multiply depth predictions when saving as png images. This is useful because we would otherwise lose decimal points when converting to 16-bit unsigned integers. Scale factor times maximum depth need to be less than 2**16 - 1 == 65535")
    parser.add_argument("--max_depth_png", type=float, default=50.0, help="Maximum depth threshold for saving the depth png image. Any predicted depth larger than this value will be capped to it. Scale factor times maximum depth need to be less than 2**16 - 1 == 65535")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    main(args)



