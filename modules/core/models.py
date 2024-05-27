from typing import Dict
from .interfaces import BaseModel

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

    Expected inputs and outputs for `predict()`:
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
