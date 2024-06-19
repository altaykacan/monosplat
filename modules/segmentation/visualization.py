import logging
from typing import Dict, Union, List
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from modules.io.utils import save_image_torch

log = logging.getLogger(__name__)


def plot_moving_object_removal_results(
    frame_ids: torch.Tensor,
    moving_object_votes: Dict,
    dists_for_plot: Dict,
    dists_mask_for_plot: Dict,
    moving_object_mask: torch.Tensor,
    log_dir: Path = Path("debug_moving_obj"),
    ) -> None:
    """
    Saves visualizations for debugging of moving object removal. Saves visualizations
    for only the

    Args:
    `frame_ids`: 1D torch tensor that contains the frame ids for every sample in the batch
    `moving_object_votes`: Dictionary where the keys are the different flow steps,
                           each entry has a `[N, 1, H, W]` boolean tensor
                           corresponding to the predicted moving object mask for
                           the respective flow step.
    `dists_for_plot`: Dictionary where the keys are the flow steps. Each entry
                      has a `[N, 1, H, W]` float tensor representing the
                      computed 3D distances for each pixel in the pointmap.
    `dists_mask_for_plot`: Dictionary where the keys are the flow steps. Each
                           entry has a `[N, 1, H, W]` boolean tensor of the masked
                           `dists` values
    `moving_object_mask`: Boolean torch tensor of shape `[N, 1, H, W]` containing
                          the combined predictions from all flow steps.
    """

    log_dir.mkdir(exist_ok=True, parents=True)
    idx = 0 # index in the batch
    frame_id = frame_ids[idx].item()

    # Save the moving object predictions for each flow step
    for flow_step, votes in moving_object_votes.items():
        vote = votes[idx, :, :, :] # get the sample from batch
        filename = f"moving_obj_vote_frame_{frame_id}_flow_step_{flow_step}.png"
        save_image_torch(vote, filename, log_dir)

    # Save the dists for each flow step
    for flow_step, dists_batch in dists_for_plot.items():
        dists = dists_batch[idx]
        filename = f"moving_obj_dists_frame_{frame_id}_flow_step_{flow_step}.png"
        save_image_torch(dists, filename, log_dir)

    # Save the dists masks
    for flow_step, dists_mask_batch in dists_mask_for_plot.items():
        dists_mask = dists_mask_batch[idx]
        filename = f"moving_obj_dists_mask_frame_{frame_id}_flow_step_{flow_step}.png"
        save_image_torch(dists_mask, filename, log_dir)

    # Save the mask of each frame id
    for i, frame_id in enumerate(frame_ids.tolist()):
        filename = f"moving_obj_mask_frame_{frame_id}.png"
        save_image_torch(moving_object_mask[i], filename, log_dir)