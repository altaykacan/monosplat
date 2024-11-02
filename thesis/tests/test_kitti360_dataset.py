from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_pil_image, pil_to_tensor

from modules.io.datasets import KITTI360Dataset
from modules.depth.models import KITTI360DepthModel, Metric3Dv2
from modules.io.utils import save_image_torch

dataset = KITTI360Dataset(0, 0, start=300, end=900)
gt = KITTI360DepthModel(dataset)
model = Metric3Dv2(dataset.intrinsics, backbone="convnext")

frames = [391, 394, 399, 846]
images = []

# Frame ids don't match the indices from the dataset (not all frames have poses)
for frame in frames:
    idx = dataset.frame_ids.index(frame)
    frame_id, image, pose = dataset[idx]
    images.append(image)

images = torch.stack(images, dim=0) # batched tensors

gt = gt.predict({"frame_ids": frames})
preds = model.predict({"images": images})
print(gt["depths"].shape)
print(preds["depths"].shape)

save_image_torch(gt["depths"], "./tests/test_results/debug_gt")
save_image_torch(preds["depths"], "./tests/test_results/debug_pred")



