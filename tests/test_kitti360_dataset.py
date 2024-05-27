from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_pil_image, pil_to_tensor

from modules.io.datasets import KITTI360Dataset
from modules.depth.models import KITTI360DummyDepthModel

dataset = KITTI360Dataset(0, 0, start=300, end=500)
model = KITTI360DummyDepthModel(dataset)

preds = model.predict({"indices": [321, 394, 499]})
print(preds["depths"].shape)



