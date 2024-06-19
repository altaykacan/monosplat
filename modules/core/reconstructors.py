"""Has implementations of standard models used in the framework"""
from typing import Dict, List, Union
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules.core.maps import PointCloud
from modules.depth.models import DepthModel
from modules.core.utils import compute_occlusions, grow_bool_mask
from modules.core.interfaces import BaseReconstructor, BaseModel, BaseBackprojector, BaseDataset
from modules.io.utils import Logger, clean_batch, save_image_torch
from modules.segmentation.moving_objects import compute_dists
from modules.segmentation.visualization import plot_moving_object_removal_results
from modules.segmentation.utils import combine_segmentation_masks


class SimpleReconstructor(BaseReconstructor):
    def __init__(
            self,
            dataset: BaseDataset,
            backprojector: BaseBackprojector,
            depth_model: BaseModel,
            cfg: Dict = {},
            ):
        self.parse_config(cfg) # inherited from interface

        self.dataset = dataset
        self.backprojector = backprojector
        self.depth_model = depth_model

        self.map = PointCloud()
        self.logger = Logger(self)
        self.dataloader = DataLoader(self.dataset, self.batch_size, shuffle=False, drop_last=False)
        self.device = dataset.device

    def run(self):
        for batch_idx, (frame_ids, images, poses) in enumerate(tqdm(self.dataloader)):
            self.step(frame_ids, images, poses, batch_idx)

        self.map.postprocess()
        self.map.save(self.output_dir / Path("map.ply"))

    def step(self, frame_ids: torch.Tensor, images: torch.Tensor, poses: torch.Tensor, batch_idx: int):
        depth_preds = self.depth_model.predict({"images": images, "frame_ids": frame_ids})
        depths = depth_preds["depths"]

        # Collect depth scale and shifts based on scale alignment
        depth_scales, depth_shifts = self.dataset.get_depth_scale_shift_by_frame_ids(frame_ids)
        if (depth_scales is not None) and (depth_shifts is not None):
            depths = depth_scales * depths + depth_shifts

        masks_backproj = self.backprojector.compute_backprojection_masks(
            images,
            depths,
            depth_scales,
            depth_shifts,
            )
        xyz, rgb = self.backprojector.backproject(images, depths, poses, masks_backproj)
        self.map.increment(xyz, rgb)
        if batch_idx % self.log_every_nth_batch == 0:
            self.logger.log_step(state={"ids": frame_ids, "depths": depths, "images": images})


# TODO implement
class MovingObjectRemoverReconstructor(SimpleReconstructor):
    def __init__(
            self,
            dataset: BaseDataset,
            backprojector: BaseBackprojector,
            depth_model: BaseModel,
            flow_model: BaseModel,
            seg_model: BaseModel,
            ins_seg_model: BaseModel,
            cfg: Dict = {},
            ) -> None:
        self.parse_config(cfg)

        self.dataset = dataset
        self.backprojector = backprojector
        self.depth_model = depth_model
        self.flow_model = flow_model
        self.seg_model = seg_model
        self.ins_seg_model = ins_seg_model # instance segmentation

        self.map = PointCloud()
        self.raw_map = PointCloud()
        self.logger = Logger(self)
        self.dataloader = DataLoader(self.dataset, self.batch_size, shuffle=False, drop_last=False)

    def parse_config(self, cfg: Dict) -> None:
        super().parse_config(cfg)

        self.flow_steps = cfg.get("flow_steps", [-2, -1, 1, 2])
        self.min_flow = cfg.get("min_flow", 0.5)
        self.classes_to_seg = cfg.get("classes_to_seg", ["car", "bus", "person", "truck", "bicycle", "motorcycle", "rider"])
        self.max_dists_moving_obj = cfg.get("max_dists_moving_obj", 30)
        self.max_d_moving_obj = cfg.get("max_d_moving_obj", 40.0)
        self.dists_thresh_moving_obj = cfg.get("dists_thresh_moving_obj", 1.0) # dists threshold as this value multiplied by the mean
        self.intersection_thresh_moving_obj = cfg.get("intersection_thresh_moving_obj", 0.7) # intersection of dists mask and instance mask
        self.min_hits_moving_obj = cfg.get("min_hits_moving_obj", 1)

    def run(self):
        super().run()
        self.raw_map.postprocess()
        self.raw_map.save(self.output_dir / Path("map_no_mov_obj_filtering.ply"))

    def step(
            self,
            frame_ids: torch.Tensor,
            images: torch.Tensor,
            poses: torch.Tensor,
            batch_idx: int
            ) -> None:
        # Get depths for target frames
        depth_preds = self.depth_model.predict({"images": images, "frame_ids": frame_ids})
        depths = depth_preds["depths"]
        N, _, H, W = depths.shape

        # Scale and shift target depths (no transformation if not specified in dataset)
        depth_scales, depth_shifts = self.dataset.get_depth_scale_shift_by_frame_ids(frame_ids)
        depths = depth_scales * depths + depth_shifts

        # Get segmentation and instance masks for moveable objects in target frame
        # Segmentation masks are usually more complete so it helps with robustness
        seg_preds = self.seg_model.predict({
            "images": images,
            "classes_to_segment": self.classes_to_seg
            })
        ins_preds = self.ins_seg_model.predict({
            "images": images,
            "classes_to_detect": self.classes_to_seg,
        })
        moveable_masks = combine_segmentation_masks(seg_preds["masks_dict"]) # [N, 1, H, W] tensor
        people_masks = seg_preds["masks_dict"]["person"] # [N, 1, H, W] tensor
        instance_masks = ins_preds["masks"] # list lists of [N, 1, H, W] tensors

        # Collect source frames around a window from the current target frame
        moving_object_votes = {} # each flow step votes for moving objects
        dists_for_plot = {} # keys are flow steps
        dists_mask_for_plot = {} # keys are flow steps
        for flow_step in self.flow_steps:
            src_ids = frame_ids + flow_step # 1D tensor of ids of the source frames

            # These are batched tensors with -1 entries if there is an index error
            src_frame_ids, src_images, src_poses = self.dataset.get_by_frame_ids(src_ids)

            # If one frame has an out-of-index flow step, skip current flow step
            if (src_frame_ids == -1).any():
                continue

            # Compute depths for source frames (frames in the window)
            src_depth_preds = self.depth_model.predict({"images": src_images, "frame_ids": src_frame_ids})
            src_depths = src_depth_preds["depths"]

            src_depth_scales, src_depth_shifts = self.dataset.get_depth_scale_shift_by_frame_ids(src_frame_ids)
            src_depths = src_depth_scales * src_depths + src_depth_shifts

            # Compute flows from target to source frames and vice-versa
            flow_pred = self.flow_model.predict({"image_a": images, "image_b": src_images})
            rev_flow_pred = self.flow_model.predict({"image_a": src_images, "image_b": images})
            flow = flow_pred["flow"] # [N, 2, H, W]
            rev_flow = rev_flow_pred["flow"]
            occlusion_mask, _ = compute_occlusions(flow, rev_flow)

            # Compute 3D distances between corresponding points, [N, 1, H, W]
            dists = compute_dists(depths, src_depths, poses, src_poses, flow, self.dataset.intrinsics, self.min_flow, occlusion_mask)

            # Mask dists according to scaled maximum depth and max dists
            max_d_tensor = self.max_d_moving_obj * torch.ones((N, 1, 1, 1)).to(depths.device)
            scaled_max_d = depth_scales * max_d_tensor + depth_shifts
            dists[depths > scaled_max_d] = 0 # TODO not sure if this makes sense
            dists[dists > self.max_dists_moving_obj] = 0

            # Mask according to the mean of non-zero values of dists over batch
            dists_mean = torch.mean(dists[dists > 0]) # scalar
            dists_mask = dists >= self.dists_thresh_moving_obj * dists_mean

            if batch_idx % self.log_every_nth_batch == 0:
                dists_for_plot[flow_step] = dists
                dists_mask_for_plot[flow_step] = dists_mask

            # Check relative intersection size of dists_masks with instance masks
            proposal_mask = torch.zeros_like(dists_mask)
            for i, ins_masks_for_sample in enumerate(instance_masks):
                for ins_mask in ins_masks_for_sample:
                    ins_mask_size = torch.sum(ins_mask) # scalar

                    # i is the index of our sample within the current batch
                    intersection_size = torch.sum(ins_mask & dists_mask[i])
                    relative_size = intersection_size / ins_mask_size

                    # if dists mask intersects instance mask take the whole instance
                    if relative_size > self.intersection_thresh_moving_obj:
                        grown_ins_mask = grow_bool_mask(ins_mask, iterations=3)
                        refined_ins_mask = moveable_masks[i] & grown_ins_mask
                        proposal_mask[i] = proposal_mask[i] | refined_ins_mask

            moving_object_votes[flow_step] = proposal_mask # [N, 1, H, W]

        # Skip reconstruction step if window is empty (no flow can be computed)
        if len(moving_object_votes) == 0:
            return None

        moving_object_masks = torch.zeros_like(depths).int()
        for proposal_masks in moving_object_votes.values():
            moving_object_masks += proposal_masks

        # Moving object pixels are 'True'
        moving_object_masks = moving_object_masks >= self.min_hits_moving_obj
        if batch_idx % self.log_every_nth_batch == 0:
            plot_moving_object_removal_results(
                frame_ids,
                moving_object_votes,
                dists_for_plot,
                dists_mask_for_plot,
                moving_object_masks,
                log_dir=self.output_dir / Path("moving_obj")
                ) # TODO implement

        masks_backproj = self.backprojector.compute_backprojection_masks(images, depths)
        masks_combined = masks_backproj & torch.logical_not(moving_object_masks | people_masks)
        xyz, rgb = self.backprojector.backproject(images, depths, poses, masks_combined)
        self.map.increment(xyz, rgb)

        # Results without moving object masking to compare
        xyz, rgb = self.backprojector.backproject(images, depths, poses, masks_backproj)
        self.raw_map.increment(xyz, rgb)
        if batch_idx % self.log_every_nth_batch == 0:
            self.logger.log_step(state={"ids": frame_ids, "depths": depths, "images": images})
        return None




