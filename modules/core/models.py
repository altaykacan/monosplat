from typing import Dict, Tuple
from pathlib import Path

from modules.segmentation.models import SegmentationModel
from modules.core.interfaces import BaseMap, BaseModel, BaseReconstructor, BaseBackprojector, BaseDataset

import torch
import numpy as np
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor


try:
    from torchvision.models.optical_flow import (
        raft_large,
        raft_small,
        Raft_Large_Weights,
        Raft_Small_Weights,
    )
    torchvision_found = True
except ImportError:
    torchvision_found = False


class OpticalFlowModel(BaseModel):
    """
    Optical flow model parent class. Implement `__init__()`, `_preprocess()`,
    `_predict()`, and `load()` and `unload()`

    Expected inputs and outputs for `self.predict()`:
    `input_dict`: `{}`
    `output_dict`: `{}`
    """
    def _check_input(self, input_dict: Dict):
        if not "image_a" in input_dict.keys():
            raise ValueError
        if not "image_b" in input_dict.keys():
            raise ValueError
        if len(input_dict["image_a"].shape) != 4:
            raise ValueError
        if len(input_dict["image_b"].shape) != 4:
            raise ValueError

    def _check_output(self, output_dict: Dict):
        pass


class RAFT(OpticalFlowModel):
    """
    Pretrained RAFT model using torchvision.

    Expected inputs and outputs for `RAFT.predict()`:
    `intput_dict`: `{"image_a": torch.Tensor [N, 3, H, W], "image_b": torch.Tensor [N, 3, H, W]}`
    `output_dict`: `{"flow": torch.Tensor [N, 2, H, W]}`
    """
    def __init__(self, model_size: str="LARGE", weights_config: str = "DEFAULT", device: str = None):
        if not torchvision_found:
            raise ImportError("torchvision can't be imported. Please check you have it installed before using a RAFT model!")

        self.model_size =  model_size
        self.weights_config = weights_config

        self._model = None
        self._weights = None
        if device is None:
            self._device ="cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

    def load(self):
        if self._model is None and self._weights is None:
            if self.model_size == "LARGE":
                if self.weights_config == "DEFAULT":
                    self._weights = Raft_Large_Weights.DEFAULT
                else:
                    raise NotImplementedError("Weights other than 'DEFAULT' not implemented for RAFT models!")

                self._model = raft_large(self._weights, progress=False)

            elif self.model_size == "SMALL":
                if self.weights_config == "DEFAULT":
                    self._weights = Raft_Small_Weights.DEFAULT
                else:
                    raise NotImplementedError("Weights other than 'DEFAULT' not implemented for RAFT models!")

                self._model = raft_small(self._weights, progress=False)

            else:
                raise NotImplementedError("Model sizes other than 'LARGE' and 'SMALL' are not implemented for RAFT models!")

            self._model.to(self._device)
            self._model.eval()

    def unload(self):
        self._model = None
        self._weights = None
        torch.cuda.empty_cache()

    def _preprocess(self, input_dict: Dict) -> Dict:
        image_a = input_dict["image_a"]
        image_b = input_dict["image_b"]

        image_a = image_a.to(self._device)
        image_b = image_b.to(self._device)

        preprocess_transforms = self._weights.transforms()
        image_a, image_b = preprocess_transforms(image_a, image_b)

        input_dict["image_a"] = image_a
        input_dict["image_b"] = image_b

        return input_dict

    def _predict(self, input_dict: Dict) -> Dict:
        image_a = input_dict["image_a"]
        image_b = input_dict["image_b"]

        with torch.no_grad():
            # RAFT is a recurrent model so we get a list of predictions, last element is
            # the most accurate
            list_of_flows = self._model(image_a, image_b)
            flow = list_of_flows[-1]

        return {"flow": flow}


# TODO could put these normal models in their own module
class NormalModel(BaseModel):
    """
    Model parent class for surface normal prediction. Implement `__init__()`,
    `_preprocess()`, `_predict()`, and `load()` and `unload()`

    Expected inputs and outputs for `self.predict()`:
    `input_dict`: `{}`
    `output_dict`: `{}`
    """
    def _check_input(self, input_dict: Dict):
        pass

    def _check_output(self, output_dict: Dict):
        pass

class Metric3Dv2NormalModel(NormalModel):
    """
    Surface normal prediction model that uses the same underlying Metric3Dv2 model
    to get normal predictions

    Expected inputs and outputs for `self.predict()`:
    `input_dict`: `{"metric3d_preds": Dict, output dictionary of a Metric3Dv2 model with a ViT backbone }`
    `output_dict`: `{"normals": torch.Tensor [N, 3, H, W] tensors extracted from the Metric3Dv2 model, "normals_vis": torch.Tensor RGB images for visualizing the normals}`
    """
    def __init__(self, metric3d_model: BaseModel):
        self.model = metric3d_model

        if "vit" not in metric3d_model._backbone:
            raise ValueError(f"To use normal predictions with Metric3D you need to use a ViT backbone! Current backbone is: {metric3d_model._backbone}")

    def load(self):
        pass

    def unload(self):
        pass

    def _preprocess(self, input_dict: Dict) -> Dict:
        return input_dict

    def _predict(self, input_dict: Dict) -> Dict:
        metric3d_preds = input_dict["metric3d_preds"]
        normals = metric3d_preds["normals"]
        normals_vis = metric3d_preds["normals_vis"]
        return {"normals": normals, "normals_vis": normals_vis}


# TODO implement
class PrecomputedNormalModel(NormalModel):
    """
    Dummy normal model that loads in precomputed normals from `3_precompute_depths_and_normals.py`

    Expected input and output for `PrecomputedNormalModel.predict()`:
    `input_dict`: `{"frame_ids":  List[int] frame ids of the images to 'predict' normals by loading in precomputed depths}`
    `output_dict`: `{"normals": Batched loaded in precomputed normal values [N, 3, H, W], "normals_vis": Same shape tensors but for surface normal visualization}`
    """
    def __init__(self, dataset: BaseDataset, device: str = None):
        self._dataset = dataset
        if device is None:
            self._device ="cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        if not self._dataset.has_depth:
            raise ValueError("Your dataset has no depth predictions, precomputed depth predictions are needed to use a precomputed normal model.")

    def load(self):
        pass

    def unload(self):
        pass

    def _preprocess(self, input_dict: Dict) -> Dict:
        return input_dict

    def _predict(self, input_dict: Dict) -> Dict:
        frame_ids = input_dict["frame_ids"]
        normals = []
        normals_vis = []

        for frame_id in frame_ids:
            idx = self._dataset.frame_ids.index(frame_id)
            normal_path = self._dataset.normal_paths[idx] # hacky way to get the normal paths
            normal = torch.from_numpy(np.load(normal_path)).double() # [3, H , W]

            # Resize and crop the normal according to the target_size of dataset
            if self._dataset.size != self._dataset.orig_size:
                normal = self._dataset.preprocess(normal) # [3, H_new, W_new]

                # Make sure normals have unit magnitude
                magnitude = torch.linalg.norm(normal, dim=1, keepdims=True)
                normal = normal / magnitude

            normals.append(normal)

            normal_vis_path = self._dataset.normal_vis_paths[idx]
            normal_vis = Image.open(normal_vis_path)
            normal_vis = pil_to_tensor(normal_vis)

            normals_vis.append(normal_vis)

        normals = torch.stack(normals, dim=0).to(self._device) # [N, 3, H_new, W_new]
        normals_vis = torch.stack(normals_vis, dim=0).to(self._device) # [N, 3, H_new, W_new]

        return {"normals": normals, "normals_vis": normals_vis}

