"""Has implementations of standard models used in the framework"""
from typing import Dict, List, Union
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules.core.maps import PointCloud
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

    def parse_config(self, cfg: Dict):
        self.output_dir = Path(cfg.get("output_dir", "."))
        self.batch_size = cfg.get("batch_size", 2)
        self.use_every_nth = cfg.get("use_every_nth", 1)

    def run(self):
        # TODO how can we do this better with batches too?
        # for idx in tqdm(range(0, len(self.dataset), self.batch_size)):
        #     frame_ids, images, poses  = self.form_batch(idx)
        #     self.step(frame_ids, images, poses)
        for frame_ids, images, poses in tqdm(self.dataloader):
            self.step(frame_ids, images, poses)


        self.map.postprocess()
        self.map.save(self.output_dir / Path("map.ply"))

    def step(self, frame_ids: torch.Tensor, images: torch.Tensor, poses: torch.Tensor):
        depth_preds = self.depth_model.predict({"images": images, "frame_ids": frame_ids})
        depths = depth_preds["depths"]

        masks_backproj = self.backprojector.compute_backprojection_masks(images, depths)
        xyz, rgb = self.backprojector.backproject(images, depths, poses, masks_backproj)

        self.map.increment(xyz, rgb)
        self.logger.log_step(state={"ids": frame_ids, "depths": depths, "images": images})


# TODO implement
class BufferedReconsructor(BaseReconstructor):
    pass






