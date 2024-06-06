from typing import Dict, Tuple

from modules.segmentation.models import SegmentationModel
from modules.core.interfaces import BaseMap, BaseModel, BaseReconstructor, BaseBackprojector

import torch

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

    def _check_output(self):
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


class Backprojector(BaseBackprojector):
    def __init__(self, cfg: Dict, intrinsics: Tuple):
        self.cfg = cfg
        self.intrinsics = intrinsics

    def backproject(self, values: torch.Tensor, depths: torch.Tensor, poses: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        N, C, H, W = values.shape
        fx, fy, cx, cy = self.intrinsics
        device = depths.device

        # Pixel coordinates
        U, V = torch.meshgrid((torch.arange(0, W), torch.arange(0, H)), indexing="xy")

        # Move origin to the middle of the image plane
        X = U - cx
        Y = V - cy

        # Undo scaling
        X = X / fx
        Y = Y / fy
        Z = torch.ones_like(X)  # z coordinates will be scaled by depth later

        coords = torch.stack([X, Y, Z], axis=2).double().unsqueeze(0).to(device) # [1 ,H, W, 3]

        # Scales coordinates by depth value
        xyz = depths[:, :, :, :, None] * coords  # [N, 1, H, W, 3]
        xyz = xyz.squeeze(1).permute(0, 3, 1, 2) # [N, 3, H , W]

        # Flatten the output tensors and the masks
        xyz = xyz.reshape(N, 3, H * W)
        values = values.reshape(N, C, H * W)
        masks = masks.reshape(N, 1, H * W)

        # Transform the coordinates to world frame, the poses are T_WC (camera to world)
        R = poses[:, :3, :3] # [N, 3, 3]
        t = poses[:, :3, 3].reshape(N, 3, 1) # [N, 3, 1]
        xyz = torch.bmm(R, xyz) + t

        # Apply masks, points that are ignored have all 0's
        xyz = xyz * masks
        values = values * masks

        return xyz, values

    def compute_backprojection_masks(self, images: torch.Tensor, depths: torch.Tensor) -> torch.Tensor:
        masks = torch.ones_like(depths).bool()
        device = depths.device
        N, C, H, W = masks.shape # C is 1

        # Region of interest as [top, bottom, left, right] edge indices
        roi = self.cfg.get("roi", [])

        if roi != []:
            masks[:, :, : roi[0], :] = False
            masks[:, :, roi[1] :, :] = False
            masks[:, :, :, : roi[2]] = False
            masks[:, :, :, roi[3] :] = False

        # Maximum and minimum depths for backprojection
        max_d = self.cfg.get("max_d", 50.0)
        min_d = self.cfg.get("min_d", 0.0)

        masks = masks & (depths < max_d) & (depths > min_d)

        # Random dropout
        dropout = self.cfg.get("dropout", 0.9)
        if dropout > 0.0:
            masks = masks & (torch.rand((N, C, H, W)).to(device) > dropout)

        return masks


class SemanticBackprojector(Backprojector):
    """
    `Backprojector` that uses a semantic segmentation model to compute
    backprojection masks
    """
    def __init__(self, cfg: Dict, intrinsics: Tuple, segmentation_model: SegmentationModel):
        self.cfg = cfg
        self.intrinsics = intrinsics
        self.segmentation_model = segmentation_model
        self.classes_to_segment = None # TODO implement

    def compute_backprojection_masks(self, images: torch.Tensor, depths: torch.Tensor) -> torch.Tensor:
        geometric_masks = super().compute_backprojection_masks(images, depths)

        semantic_masks = self.segmentation_model({"images": images, "classes_to_segment": self.classes_to_segment})
        # TODO implement
