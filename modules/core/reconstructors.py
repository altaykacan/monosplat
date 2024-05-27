"""Has implementations of standard models used in the framework"""
from typing import Dict

import torch

from .interfaces import BaseReconstructor

class SimpleReconstructor(BaseReconstructor):
    def __init__(self, cfg: Dict = {}):
        pass

    def run(self):
        self.config_parser.parse()
        self.logger.log_input()
        self.dataset.load_image_paths_and_poses()

        while self.still_running():
            self.step()

        if self.clean_cloud:
            self.map.clean()

        if self.estimate_normals:
            self.map.estimate_normals()

        if self.add_skydome:
            self.map.add_skydome()

        self.map.save(self.output_path)

    def step(self):
        images, poses = self.dataset[self.image_idx]

        depth_preds = self.depth_model.predict({"images": images})
        depths = depth_preds["depths"]





