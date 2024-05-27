from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_pil_image, pil_to_tensor

from modules.io.utils import save_image_torch, save_mask_torch

from modules.depth.models import Metric3Dv2

image_a_path = Path("./demo/munich_1.png")
image_b_path = Path("./demo/munich_2.png")
image_a = pil_to_tensor(Image.open(image_a_path).convert("RGB"))
image_b = pil_to_tensor(Image.open(image_b_path).convert("RGB"))

images = torch.stack((image_a, image_b), dim=0)
intrinsics = [535.2, 534.9, 512, 288]
depth_pred_size = (616, 1064)

metric3d = Metric3Dv2({"intrinsics": intrinsics, "depth_pred_size": depth_pred_size})
preds = metric3d.predict({"images": images})

depths = preds["depths"]
save_image_torch(depths[0], "debug_1")
save_image_torch(depths[1], "debug_2")

metric3d.unload()