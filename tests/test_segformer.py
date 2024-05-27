from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_pil_image, pil_to_tensor

from modules.segmentation.models import SegFormer

image_a_path = Path("./demo/munich_1.png")
image_b_path = Path("./demo/munich_2.png")
image_a = pil_to_tensor(Image.open(image_a_path).convert('RGB'))
image_b = pil_to_tensor(Image.open(image_b_path).convert('RGB'))

images = torch.stack((image_a, image_b), dim=0)

segformer = SegFormer()
segformer.load()

input_dict = {"images": images, "classes_to_segment": ["car", "sky"]}
output_dict = segformer.predict(input_dict)
segformer.unload()

print(output_dict)
