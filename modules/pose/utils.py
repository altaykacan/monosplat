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


def get_transformed_pose(
        pose: torch.Tensor,
        angle_x: float = 0.0,
        angle_y: float = 0.0,
        angle_z: float = 0.0,
        tx: float = 0.0,
        ty: float = 0.0,
        tz: float = 0.0,
        mean_delta = None):

        device = pose.device

        # pose rotation
        radian_z = torch.tensor(angle_z * torch.pi / 180)
        Rz = torch.tensor([[torch.cos(radian_z), -torch.sin(radian_z), 0.0],
                        [torch.sin(radian_z),  torch.cos(radian_z), 0.0],
                        [0.0,                0.0,                   1.0]]).to(device).double()

        radian_y = torch.tensor(angle_y * torch.pi / 180)
        Ry = torch.tensor([[torch.cos(radian_y), 0.0, torch.sin(radian_y)],
                        [0.0,                  1.0,                 0.0],
                        [-torch.sin(radian_y), 0.0, torch.cos(radian_y)]]).to(device).double()

        radian_x = torch.tensor(angle_x * torch.pi / 180)
        Rx = torch.tensor([[1.0,                0.0,               0.0],
                           [0.0,    torch.cos(radian_x), -torch.sin(radian_x)],
                           [0.0, torch.sin(radian_x),  torch.cos(radian_x)]]).to(device).double()

        # displacement
        disp = torch.tensor([[tx, ty, tz]]).to(device).double()

        # Displacement as percentages of mean deltas for each direction
        if mean_delta is not None:
            disp = disp * mean_delta

        R = pose[:3, :3] # (cam_1 --> world)
        t = pose[:3, 3]

        R_new = R @ Rx @ Ry @ Rz
        t_new = t + (R @ disp.T).T # we give the displacement in C1 coordinates but represent the new translation t in world coordinates

        pose_new = torch.eye(4).to(device).double()
        pose_new[:3, :3] = R_new
        pose_new[:3, 3] = t_new

        return pose_new


def get_new_trajectory(
          poses: torch.Tensor,
          angle_x_range: float = 15,
          angle_y_range: float = 15,
          angle_x_period: float = 100,
          angle_y_period: float = 100,
          disp_x_range: float = 1.0,
          disp_y_range: float = 1.0,
          disp_x_period: float = 40,
          disp_y_period: float = 40,
          disp_y_max: float = 0.1,
          disp_delay: int = 0,
          mean_delta: torch.Tensor = torch.ones((1,3)),
          ):
            num_poses = poses.shape[0]
            new_poses = []

            angle_x_wave = angle_x_range * torch.sin(torch.arange(num_poses) * 2 * torch.pi / angle_x_period)
            angle_y_wave = angle_y_range * torch.sin(torch.arange(num_poses) * 2 * torch.pi / angle_y_period)
            disp_x_wave = disp_x_range * torch.cos(torch.arange(num_poses) * 2 * torch.pi / disp_x_period)

            # Positive y is downwards, too large values cause the camera to be below the road
            disp_y_wave = (disp_y_range * torch.cos(torch.arange(num_poses) * 2 * torch.pi / disp_y_period)).clamp(max=disp_y_max)

            # Shift the displacement wave till disp_delay, no displacement when delay is active
            if disp_delay != 0:
                temp = disp_x_wave.clone()
                disp_x_wave[disp_delay:] = temp[0:(num_poses - disp_delay)]
                disp_x_wave[:disp_delay] = 0

                temp = disp_y_wave.clone()
                disp_y_wave[disp_delay:] = temp[0:(num_poses - disp_delay)]
                disp_y_wave[:disp_delay] = 0

            for i, p in enumerate(poses):
                new_p = get_transformed_pose(
                      p,
                      angle_x=angle_x_wave[i],
                      angle_y=angle_y_wave[i],
                      tx=disp_x_wave[i],
                      ty=disp_y_wave[i],
                      mean_delta=mean_delta,
                      )
                new_poses.append(new_p)

            return new_poses






