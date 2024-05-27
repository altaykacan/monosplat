from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_pil_image, pil_to_tensor

image_a_path = Path("../demo/munich_1.png")
image_b_path = Path("../demo/munich_2.png")
image_a = pil_to_tensor(Image.open(image_a_path).convert("RGB"))
image_b = pil_to_tensor(Image.open(image_b_path).convert("RGB"))

images = torch.stack((image_a, image_b), dim=0)

##############################
# Debug
##############################

