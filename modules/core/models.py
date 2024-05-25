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
        assert "image_a" in input_dict.keys()
        assert "image_b" in input_dict.keys()

    def _check_output(self):
        pass

class RAFT(OpticalFlowModel):
    def __init__(self, cfg: Dict = {}):
        if not torchvision_found:
            raise ImportError("torchvision can't be imported. Please check you have it installed before using a RAFT model!")

        self.model_size = cfg.get("raft_model_size", "LARGE")
        self.weights_config = cfg.get("raft_weights_config", "DEFAULT")

        self._model = None
        self._weights = None
        self._device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

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