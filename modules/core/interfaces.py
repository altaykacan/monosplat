from pathlib import Path
from typing import Union, Dict, NamedTuple
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
    def clean(self):
        """Cleans outliers from the map"""
        pass

    @abstractmethod
    def transform(self, T: torch.Tensor):
        """Transforms the Map in place according to a rigid body transform `T` """
        pass

    @abstractmethod
    def increment(self, submap):
        """Adds a new submap to the existing map"""



class BaseReconstructor(ABC):
    @abstractmethod
    def run(self):
        """Runs all the steps to create a 3D reconstruction using dense depth backprojection for posed images"""
        pass

    @abstractmethod
    def still_running(self) -> bool:
        """Function to check whether the Reconstructor is still processing the images"""
        pass

    @abstractmethod
    def step(self):
        """One step of the reconstruction process. Specific to reconstructors."""
        pass

class BaseConfigParser(ABC):
    @abstractmethod
    def parse(self, Reconstructor: BaseReconstructor, cfg: Dict):
        pass

