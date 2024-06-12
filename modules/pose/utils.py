from typing import List, Dict
from pathlib import Path

import torch
import numpy as np

from modules.io.datasets import CustomDataset, KITTI360Dataset, KITTIDataset


def get_transformation_matrix(rot: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Forms a 4x4 homogenous transformation matrix which corresponds to the
    following transform when left-multiplied by a homogeneous vector:
    `rot @ vector + t`.
    """
    if rot.shape != (3, 3):
        raise ValueError(f"Expected the rotation matrix to be of shape (3, 3) but got {tuple(rot.shape)} instead.")
    if not (t.shape == (3,) or t.shape == (3,1)):
        raise ValueError(f"Expected the translation vector to be of shape (3, 1) or (3,) but got {tuple(t.shape)} instead.")

    device = rot.device
    matrix = torch.eye(4).to(device) # identity matrix
    matrix[:3, :3] = rot
    matrix[:3, 3] = t.squeeze()
    return matrix

