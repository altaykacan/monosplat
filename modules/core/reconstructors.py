"""Has implementations of standard models used in the framework"""
from typing import Dict, List, Union
from pathlib import Path
import logging

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules.core.maps import PointCloud
from modules.core.models import RAFT, Metric3Dv2NormalModel, PrecomputedNormalModel
from modules.core.utils import compute_occlusions, grow_bool_mask
from modules.core.interfaces import BaseReconstructor, BaseModel, BaseBackprojector, BaseDataset
from modules.core.visualization import visualize_flow
from modules.depth.models import DepthModel
from modules.io.utils import Logger, clean_batch, save_image_torch
from modules.segmentation.moving_objects import compute_dists
from modules.segmentation.models import MaskRCNN, SegFormer
from modules.segmentation.visualization import plot_moving_object_removal_results
from modules.segmentation.utils import combine_segmentation_masks

log = logging.getLogger(__name__)


class SimpleReconstructor(BaseReconstructor):
    def __init__(
            self,
            dataset: BaseDataset,
            backprojector: BaseBackprojector,
            depth_model: BaseModel,
            normal_model: BaseModel = None,
            seg_model: BaseModel = None,
            cfg: Dict = {},
            ):
        self.parse_config(cfg) # inherited from interface

        self.dataset = dataset
        self.backprojector = backprojector
        self.depth_model = depth_model
        self.normal_model = normal_model # optional
        self.seg_model= seg_model # optional

        self.map = PointCloud()
        self.logger = Logger(self)
        self.device = dataset.device

    def parse_config(self, cfg: Dict):
        self.output_dir = Path(cfg.get("output_dir", "."))
        self.log_every_nth_batch = cfg.get("log_every_nth_batch", 50)
        self.classes_to_remove = cfg.get("classes_to_remove", ["car"])
        self.map_name = cfg.get("map_name", "cloud.ply")

        self.use_every_nth = cfg.get("use_every_nth", 1)
        self.batch_size = cfg.get("batch_size", 4)

        self.clean_pointcloud = cfg.get("clean_pointcloud", False)
        self.add_skydome = cfg.get("add_skydome", False)
        self.init_cloud_path = cfg.get("init_cloud_path", None)
        self.downsample_pointcloud_voxel_size = cfg.get("downsample_pointcloud_voxel_size", None)

    def run(self):
        batch_idx = 0
        batch = []
        # We don't use a PyTorch DataLoader because it doesn't support skipping samples
        # We could create a dataset wrapper that has the skipped paths instead but
        # custom batching seemed like a better idea
        for i in tqdm(range(0, len(self.dataset), self.use_every_nth)):
            frame_id, image, pose = self.dataset[i]
            if len(batch) < self.batch_size:
                batch.append((frame_id, image, pose))

            # Do step if batch is full or if the next iteration would end the loop
            if (len(batch) >= self.batch_size) or ((i + self.use_every_nth) >= len(self.dataset)):
                frame_ids = torch.stack([torch.tensor(el[0]) for el in batch], dim=0)
                images = torch.stack([el[1] for el in batch], dim=0)
                poses = torch.stack([el[2] for el in batch], dim=0)

                self.step(frame_ids, images, poses, batch_idx)
                batch = []
                batch_idx +=1
            else:
                continue

        # Converts it to Open3D PointCloud
        self.map.postprocess()

        if self.clean_pointcloud:
            logging.info("Cleaning your point cloud...")
            self.map.clean()

        if self.init_cloud_path is not None:
            logging.info(f"The initial cloud at '{self.init_cloud_path}' to your point cloud...")
            self.map.add_init_cloud(self.init_cloud_path)

        if self.downsample_pointcloud_voxel_size is not None:
            logging.info("Downsampling your point cloud...")
            self.map.downsample(self.downsample_pointcloud_voxel_size, self.dataset.depth_scale)

        # Skydome doesn't get influenced by the initial cloud we add
        if self.add_skydome:
            logging.info("Adding skydome to your point cloud...")
            self.map.add_sky_dome()

        self.map.save(self.output_dir / Path(self.map_name))

    def step(self, frame_ids: torch.Tensor, images: torch.Tensor, poses: torch.Tensor, batch_idx: int):
        N, C, H, W = images.shape
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

        if self.seg_model is not None:
            seg_preds = self.seg_model.predict({"images": images, "frame_ids": frame_ids, "classes_to_segment": self.classes_to_remove})
            masks_dict = seg_preds["masks_dict"]
            semantic_masks = combine_segmentation_masks(masks_dict)
            semantic_masks = torch.logical_not(semantic_masks) # we want movables to be False
            masks_backproj = masks_backproj &  semantic_masks

        if self.normal_model is not None:
            normal_preds = self.normal_model.predict({"metric3d_preds": depth_preds, "images": images, "frame_ids": frame_ids})
            normals = normal_preds["normals"] # [N, 3, H, W]

            # Normals are in camera frame, need to rotate to world frame to save in map
            R_WC = poses[:, :3, :3] # [N, 3, 3]
            normals = torch.bmm(R_WC, normals.view(N, 3, -1)).reshape(N, 3, H, W)
            _, normals_backproj = self.backprojector.backproject(normals, depths, poses, masks_backproj)
        else:
            normals_backproj = None

        # If both models exist we save an additional boolean tensor marking the ground of the point cloud
        if (self.seg_model is not None) and (self.normal_model is not None):
            road_preds = self.seg_model.predict({"images": images, "classes_to_segment": ["road", "sidewalk"]})
            road_dict = road_preds["masks_dict"]
            road_masks = combine_segmentation_masks(road_dict)

            # is_road is a [N, 1, num_points] boolean tensor, it's true for every point that is from a road pixel
            _, is_road = self.backprojector.backproject(road_masks, depths, poses, masks_backproj)
        else:
            is_road = None

        xyz, rgb = self.backprojector.backproject(images, depths, poses, masks_backproj)
        self.map.increment(xyz, rgb=rgb, normals=normals_backproj, is_road=is_road)
        if batch_idx % self.log_every_nth_batch == 0:
            self.logger.log_step(state={"ids": frame_ids, "depths": depths, "images": images})


class MovingObjectRemoverReconstructor(SimpleReconstructor):
    def __init__(
            self,
            dataset: BaseDataset,
            backprojector: BaseBackprojector,
            depth_model: BaseModel,
            flow_model: BaseModel,
            seg_model: BaseModel,
            ins_seg_model: BaseModel,
            normal_model: BaseModel = None,
            cfg: Dict = {},
            ) -> None:
        self.parse_config(cfg)

        self.dataset = dataset
        self.backprojector = backprojector
        self.depth_model = depth_model
        self.flow_model = flow_model
        self.seg_model = seg_model # semantic segmentation
        self.ins_seg_model = ins_seg_model # instance segmentation
        self.normal_model = normal_model # surface normal prediction

        self.map = PointCloud()
        self.raw_map = PointCloud()
        self.logger = Logger(self)

        self.prev_mov_obj_mask = None # for tracking moving objects

    def parse_config(self, cfg: Dict) -> None:
        super().parse_config(cfg)

        self.flow_steps = cfg.get("flow_steps", [-4, -3, -2, -1, 1, 2, 3, 4])
        self.min_flow = cfg.get("min_flow", 0.5)
        self.classes_to_remove = cfg.get("classes_to_remove", ["car", "bus", "person", "truck", "bicycle", "motorcycle", "rider", "sky"])
        self.max_dists_moving_obj = cfg.get("max_dists_moving_obj", 30)
        self.max_d_moving_obj = cfg.get("max_d_moving_obj", 40.0)
        self.dists_thresh_moving_obj = cfg.get("dists_thresh_moving_obj", 1.0) # dists threshold is this value multiplied by the mean
        self.intersection_thresh_moving_obj = cfg.get("intersection_thresh_moving_obj", 0.7) # intersection of dists mask and instance mask
        self.min_hits_moving_obj = cfg.get("min_hits_moving_obj", 2)

        self.track_moving_obj = cfg.get("track_moving_obj", True) # flag to whether to track
        self.intersection_thresh_tracking = cfg.get("intersection_thresh_tracking", 0.4) # intersection of dists mask and instance mask

        if self.track_moving_obj and self.batch_size != 1:
            raise ValueError("Tracking moving objects only makes sense if your batch size is 1!")

    def run(self):
        super().run()
        self.raw_map.postprocess() # to compare with/without point clouds
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

        # For debug
        if frame_ids[0].item() == 641:
            print("hoold up")

        # For naive tracking
        if self.prev_mov_obj_mask is None and self.track_moving_obj:
            self.prev_mov_obj_mask = torch.zeros_like(depths).bool()

        # Scale and shift target depths (no transformation if not specified in dataset)
        depth_scales, depth_shifts = self.dataset.get_depth_scale_shift_by_frame_ids(frame_ids)
        depths = depth_scales * depths + depth_shifts

        # Get segmentation and instance masks for moveable objects in target frame
        # Segmentation masks are usually more complete so it helps with robustness
        seg_preds = self.seg_model.predict({
            "images": images,
            "classes_to_segment": self.classes_to_remove
            })
        ins_preds = self.ins_seg_model.predict({
            "images": images,
            "classes_to_detect": self.classes_to_remove,
        })
        moveable_masks = combine_segmentation_masks(seg_preds["masks_dict"]) # [N, 1, H, W] tensor
        people_masks = seg_preds["masks_dict"]["person"] # [N, 1, H, W] tensor
        instance_masks = ins_preds["masks"] # list of lists [1, 1, H, W] tensors

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
            for i, ins_masks_for_sample in enumerate(instance_masks): # i is the batch index
                for ins_mask in ins_masks_for_sample:
                    ins_mask_size = torch.sum(ins_mask) # scalar

                    # i is the index of our sample within the current batch
                    intersection_size = torch.sum(ins_mask & dists_mask[i])
                    track_intersection_size = torch.sum(ins_mask / self.prev_mov_obj_mask[i])
                    relative_size = intersection_size / ins_mask_size
                    track_relative_size = track_intersection_size / ins_mask_size

                    # if current instance mask intersects previous moving object masks
                    # if dists mask intersects instance mask take the whole instance

                    if self.track_moving_obj:
                        add_ins_mask = (relative_size >= self.intersection_thresh_moving_obj)  \
                                    or (track_relative_size >= self.intersection_thresh_moving_obj)
                    else:
                        add_ins_mask = (relative_size >= self.intersection_thresh_moving_obj)

                    if add_ins_mask:
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

        if self.track_moving_obj:
            self.prev_mov_obj_mask = moving_object_masks # remember for next iteration

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


# TODO can extend this factory implementation and do it for other classes too
class ReconstructorFactory():
    """
    Factory to create and return `Reconstructor` instances. The essential components
    for all reconstructors (dataset, backprojector, and depth model) are needed
    to initalize the factory itself. The optional bells and whistles for the
    reconstructors can be specified in the `cfg` dictionary for the `get_reconstructor()`
    method.
    """
    def __init__(self, dataset: BaseDataset, backprojector: BaseBackprojector, depth_model: BaseModel) -> None:
        self.dataset = dataset
        self.backprojector = backprojector
        self.depth_model = depth_model

    def get_reconstructor(self, reconstructor_type: str, cfg: Dict = {}) -> BaseReconstructor:
        if cfg["seg_model_type"] == "segformer":
            seg_model = SegFormer()
        else:
            seg_model = None

        if cfg["normal_model_type"] == "precomputed":
            normal_model = PrecomputedNormalModel(self.dataset)
        elif cfg["normal_model_type"] == "metric3d_vit":
            normal_model = Metric3Dv2NormalModel(self.depth_model)
        else:
            normal_model = None

        if reconstructor_type == "simple":
            recon = SimpleReconstructor(
                self.dataset,
                self.backprojector,
                self.depth_model,
                seg_model=seg_model,
                normal_model=normal_model,
                cfg=cfg
                )
        elif reconstructor_type == "moving_obj":
            if cfg["flow_model_type"] == "raft":
                flow_model = RAFT()
            else:
                raise NotImplementedError

            if cfg["ins_seg_model_type"] == "mask_rcnn":
                ins_seg_model = MaskRCNN()
            else:
                raise NotImplementedError

            recon =  MovingObjectRemoverReconstructor(
                                                self.dataset,
                                                self.backprojector,
                                                self.depth_model,
                                                flow_model,
                                                seg_model,
                                                ins_seg_model,
                                                normal_model,
                                                cfg=cfg,
                                            )
        else:
            raise NotImplementedError

        return recon

