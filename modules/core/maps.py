"""Has implementation of standard map classes for the framework"""
from typing import Dict

import torch
import numpy as np
import open3d as o3d

from .interfaces import BaseMap


class PointCloud(BaseMap):
    def __init__(self, cfg: Dict =  {}):
        self._xyz = None
        self._rgb = None
        self._normals = None


    def increment(self, depths):
        pass