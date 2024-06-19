"""This is a script that you can use to mask out the sky or any other semantic mask and generate a new dataset or just save the masks"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from modules.segmentation.models import SegFormer
from modules.segmentation.utils import combine_segmentation_masks
import argparse

def main(args):
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir) if output_dir is not None else image_dir.parent / Path("masks")
    classes_to_mask = args.classes_to_mask
    save_masks_only= args.save_masks_only
    image_extension = args.image_extention

    image_paths = []
    if image_extension.lower() == "png":
        image_paths.extend(sorted(list(image_dir.glob("*.png"))))
    elif image_extension.lower() == "jpg":
        image_paths.extend(sorted(list(image_dir.glob("*.jpg"))))

    model = SegFormer()
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in tqdm(image_paths):
        image = Image.open(path)
        image = F.pil_to_tensor(image).unsqueeze(0) # dummy batch dimension
        masks_dict = model.predict({"images": image, "classes_to_seg": classes_to_mask})
        masks = combine_segmentation_masks(masks_dict["masks_dict"])
        masks = torch.logical_not(masks).cpu()

        if save_masks_only:
            mask = mask.int().squeeze().numpy()

            # Save masks as png to avoid jpg compression, we want a binary mask
            output_path = Path(output_dir, path.name + ".png")
            cv2.imwrite(str(output_path), mask * 255)
        else:
            masked_image = mask * image
            masked_image = F.to_pil_image(masked_image)
            masked_image.save(Path(output_dir, path.name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to mask out the sky or any other semantic mask and generate a new dataset or just save the masks")
    parser.add_argument("--image_dir", "-i", type=str, help="Path to the directory containing the input images")
    parser.add_argument("--output_dir", "-o", type=str, default=None, help="Path to the directory where the masked images or masks will be saved")
    parser.add_argument("--classes_to_mask", nargs="+", default=["car"], help="List of classes to mask, provide input as '--classes_to_mask car person bus'")
    parser.add_argument("--save_masks_only", action="store_true", help="Flag to specify whether only masks are saved or the masked images are saved")
    parser.add_argument("--image_extention", type=str, choices=["png", "jpg"], default="png", help="Extension of the input images")

    args = parser.parse_args()
    main(args)
