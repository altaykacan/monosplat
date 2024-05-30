"""Has implementations of standard models used in the framework"""
from typing import Dict, List, Union

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules.core.maps import PointCloud
from modules.core.models import Backprojector
from modules.depth.models import DepthModel
from modules.core.interfaces import BaseReconstructor, BaseModel, BaseBackprojector, BaseDataset
from modules.core.utils import Logger

class SimpleReconstructor(BaseReconstructor):
    def __init__(self, dataset: BaseDataset, backprojector: BaseBackprojector, depth_model: DepthModel, cfg: Dict = {}):
        self.parse_config(cfg)

        self.depth_model = depth_model
        self.dataset = dataset
        self.backprojector = backprojector

        self.map = PointCloud()
        self.logger = Logger(self)
        self.dataloader = DataLoader(self.dataset, self.batch_size, shuffle=False, drop_last=False)

        self.current_idx = 0

    def parse_config(self, cfg: Dict):
        self.batch_size = cfg.get("batch_size", 2)
        self.use_every_nth = cfg.get("use_every_nth", 1)

    def run(self):
        for images, poses in tqdm(self.dataloader):
            self.step(images, poses)

        self.map.postprocess()
        self.map.save("map.ply")

    def step(self, images: torch.Tensor, poses: torch.Tensor):
        depth_preds = self.depth_model.predict({"images": images})
        depths = depth_preds["depths"]

        # TODO implement
        masks_backproj = self.backprojector.compute_backprojection_masks(images, depths)
        xyz, rgb = self.backprojector.backproject(images, depths, poses, masks_backproj)

        self.map.increment(xyz, rgb)

        self.logger.log_step(state={"depths": depths, "rgb": rgb, "poses": poses})
        self.current_idx += 1


    def still_running(self):
        return self.current_idx < len(self.dataset)






