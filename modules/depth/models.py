from typing import Dict, List, Tuple

import torch
import cv2
import numpy as np

from modules.io.datasets import KITTI360Dataset
from modules.core.interfaces import BaseModel


class DepthModel(BaseModel):
    def _check_input(self, input_dict: Dict):
        pass

    def _check_output(self, output_dict: Dict):
        pass

class Metric3Dv2(DepthModel):
    """
    Metric3Dv2 model for monocular metric depth prediction.

    Expected inputs when initializing
    `cfg`: `{"intriniscs": Tuple[float] pinhole intrinsics as (fx, fy, cx, cy), "depth_pred_size": Tuple[int] size that the input images will be resized to before giving them to the depth model}`

    Expected inputs and outputs for prediction
    `input_dict`: `{"images": batched image tensors with shape [N, C, H, W]}`
    `output_dict`: `{"pred_depth": batched depth predictions with shape [N, C, H, W]}`
    """
    def __init__(self, intrinsics: Tuple, depth_pred_size: Tuple, device: str = None):
        self._intrinsics = intrinsics
        self._transformed_intrinsics = None

        # Suggestions from the authors for vit model: (616, 1064), for convnext model: (544, 1216)
        self._input_size = depth_pred_size
        self._pad_info = None

        self._model = None
        if device is None:
            self._device ="cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device


    def load(self):
        if self._model is None:
            self._model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_giant2', pretrain=True)
            self._model.cuda().eval()
            # TODO implement own wrapper just in case torch hub is not available


    def unload(self):
        self._model = None
        torch.cuda.empty_cache()


    def _preprocess(self, input_dict: Dict) -> Dict:
        # Code heavily inspired from original metric3dv2 authors: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py#L122
        images = input_dict["images"]

        N, C, H, W = images.shape

        preprocessed_images = []

        # Convert to numpy arrays to use cv2 image processing per item
        for batch_idx in range(N):
            image_origin = images[batch_idx] # [C, H, W]
            image_origin = image_origin.detach().cpu().permute(1,2,0).numpy() # [H, W, C]

            scale = min(self._input_size[0] / H, self._input_size[1] / W)
            image = cv2.resize(image_origin, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_LINEAR)

            # TODO what about cropping? Do we need to consider that?
            # Scaling the intrinsics for new size
            self._transformed_intrinsics = [i * scale for i in self._intrinsics]

            # Padding to input_size
            padding = [123.675, 116.28, 103.53] # color of the padded border, light brown gray-ish
            H, W, _ = image.shape
            pad_h = self._input_size[0] - H
            pad_w = self._input_size[1] - W
            pad_h_half = pad_h // 2
            pad_w_half = pad_w // 2

            image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)
            preprocessed_images.append(image)

        pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

        # Convert back to torch, get [C, H, W], and get batch dimension back
        preprocessed_images = [torch.from_numpy(img).permute(2, 0, 1).to(self._device).float() for img in preprocessed_images]
        preprocessed_images = torch.stack(preprocessed_images, dim=0) # [N, C, H, W]

        # Normalize using imagenet mean and standard deviation as floats scaled between 0-255
        mean = torch.tensor([123.675, 116.28, 103.53]).float()[None, :, None, None]
        std = torch.tensor([58.395, 57.12, 57.375]).float()[None, :, None, None]

        preprocessed_images = (preprocessed_images - mean) / std

        input_dict["preprocessed_images"] = preprocessed_images
        input_dict["pad_info"] = pad_info # for unpadding later

        return input_dict

    def _predict(self, input_dict: Dict) -> Dict:
        # Code heavily inspired from original metric3dv2 authors: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py#L122
        images_origin = input_dict["images"]
        images = input_dict["preprocessed_images"]
        pad_info = input_dict["pad_info"]

        N, _, H_origin, W_origin = images_origin.shape

        with torch.no_grad():
            pred_depth, confidence, depth_output_dict = self._model.inference({"input": images})

        # Undo padding, keeping the batch and channel dimensions
        N, C, H, W = pred_depth.shape
        pred_depth = pred_depth[:, :, pad_info[0] : H - pad_info[1], pad_info[2] : W - pad_info[3]]

        # Upsample to original size
        pred_depth = torch.nn.functional.interpolate(pred_depth, (H_origin, W_origin), mode="bilinear")

        avg_intrinsics = (self._transformed_intrinsics[0] + self._transformed_intrinsics[1])
        canonical_to_real_scale = avg_intrinsics / 1000.0 # 1000.0 is the focal length of the canonical camera

        pred_depth = pred_depth * canonical_to_real_scale # now the depth is supposed to be metric
        pred_depth = pred_depth.clamp(min=0, max=300)

        return {"depths": pred_depth}

class KITTI360DummyDepthModel(DepthModel):
    """
    Dummy Depth Model that reads in the ground truth data from KITTI360

    Expected input and output for `KITTI360DummyDepthModel.predict()`:
    `input_dict`: `{"indices":  List[int] indices of the images to 'predict' depths by loading in the daset}`
    `output_dict`: `{"depths": Batched loaded in ground truth depth values [N, 1, H, W]}`

    """
    def __init__(self, dataset: KITTI360Dataset):
        self._dataset = dataset

    def load(self):
        pass

    def unload(self):
        pass

    def _preprocess(self, input_dict: Dict) -> Dict:
        return input_dict

    def _predict(self, input_dict: Dict) -> Dict:
        indices = input_dict["indices"] # indices from dataset as a list
        depths = []

        for idx in indices:
            depth_path = self._dataset.gt_depth_paths[idx]
            depth = torch.from_numpy(np.load(depth_path)) # [H , W]
            depths.append(depth.unsqueeze(0)) # [1, H, W]

        depths = torch.cat(depths, dim=0) # [N, 1, H, W]

        return {"depths": depths}


# TODO implement, the  torch versions are different and that might cause issues with metric3dv2 and mmseg installations
class UniDepth(DepthModel):
    def __init__(self):
        pass