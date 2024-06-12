"""Defines the abstract interfaces for core components of the framework"""
from pathlib import Path
from typing import Union, Dict, Tuple, List, NamedTuple
from abc import ABC, abstractmethod

import torch

class BaseModel(ABC):
    """
    Abstract base class for every model used within the framework. Implement
    `_check_input()` and `_check_output` when creating a parent class for some
    type of network. When creating subclasses of that parent class, implement
    `_preprocess()` and `_predict()`.
    """
    @abstractmethod
    def load(self):
        """Loads the model into device memory"""
        pass

    @abstractmethod
    def unload(self):
        """Unloads the model from device memory"""
        pass

    @abstractmethod
    def _preprocess(self, input_dict: Dict) -> Dict:
        """Preprocesses the input images according to model specifications"""
        pass

    @abstractmethod
    def _check_input(self, input_dict: Dict):
        """
        Checks if the `input_dict` matches the expected format of the model
        Raises assertion errors if not.
        """
        pass

    @abstractmethod
    def _check_output(self, output_dict: Dict):
        """
        Checks if the `output_dict` matches the expected format of the model.
        Raises assertion errors if not.
        """

    @abstractmethod
    def _predict(self, input_dict: Dict) -> Dict:
        """
        Returns model predictions expects preprocessed data. Inputs and outputs
        are dictionaries to maintain flexibility accross different model types.
        """
        pass

    def predict(self, input_dict: Dict) -> Dict:
        """
        Main interface to get predictions from any model. See the docstrings
        of the respective classes for info on what the input and output dictionaries
        should be.
        """
        self.load()
        self._check_input(input_dict)
        input_dict = self._preprocess(input_dict)
        output_dict = self._predict(input_dict)
        self._check_output(output_dict)

        return output_dict


class BaseMap(ABC):
    @abstractmethod
    def save(self, filename: Union[str, Path]):
        """Saves map to disk"""
        pass

    @abstractmethod
    def transform(self, T: torch.Tensor):
        """Transforms the map in-place according to a rigid body transform `T` """
        pass

    @abstractmethod
    def increment(self, xyz: torch.Tensor, rgb: torch.Tensor, normals: torch.Tensor):
        """Initializes and Increments the existing map"""
        pass

    @abstractmethod
    def postprocess(self):
        """Applies postprocessing to the map, i.e. cleaning, downsampling, normal estimation, etc."""
        pass


class BaseReconstructor(ABC):
    @abstractmethod
    def run(self):
        """Runs all the steps to create a 3D reconstruction using dense depth backprojection for posed images"""
        pass

    @abstractmethod
    def step(self, ids, images, poses):
        """One step of the reconstruction process. Specific to reconstructors"""
        pass

    @abstractmethod
    def parse_config(self, cfg: Dict):
        """Parses the configuration dictionary and sets instance variables"""
        pass

class BaseBackprojector(ABC):
    @abstractmethod
    def backproject(self, values: torch.Tensor, depths: torch.Tensor, poses: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Backprojects pixels based on depth and masks, converts the 3D points
        to world coordinates from the camera frame using the poses. Returns a
        tuple of the `xyz` coordinates and the associated value from `values`
        for each point as `[C, num_points]`
        """
        pass

    @abstractmethod
    def compute_backprojection_masks(self, images: torch.Tensor, depths: torch.Tensor) -> torch.Tensor:
        """
        Computes which pixels should be included in the backprojection, returns
        the masks as `[N, 1, H, W]` tensors with pixels to backproject as `True`
        """
        pass


class BaseLogger(ABC):
    @abstractmethod
    def __init__(self, reconstructor: BaseReconstructor):
        pass

    @abstractmethod
    def log_step(self, state: Dict):
        pass


class BaseDataset(ABC):
    @abstractmethod
    def set_target_intrinsics(self):
        pass

    @abstractmethod
    def load_image_paths_and_poses(self):
        pass

    @abstractmethod
    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[int, torch.Tensor, torch.Tensor]:
        pass