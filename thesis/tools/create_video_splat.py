"""
A quick script to create trained gaussian splatting model fly throughs by perturbing the original camera trajectory with sinusoidal displacements/rotations
"""
import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image

from configs.data import PADDED_IMG_NAME_LENGTH
from modules.io.utils import save_image_torch
from modules.io.datasets import ColmapDataset, KITTI360Dataset, CustomDataset
from modules.pose.utils import get_new_trajectory
from modules.core.visualization import make_vid

# To be able to load modules and functions for 3DGS
sys.path.append("./submodules/gaussian-splatting/")

# This is ugly and would cause problems if we have modules with the same names
# in the root directory, but renaming the 3dgs submodule to gaussian_splatting
# isn't straightforward due to the nested submodules
from scene import Scene
from scene.cameras import Camera
from gaussian_renderer import render, GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args

def render_sets(
          dataset : ModelParams,
          iteration : int,
          pipeline : PipelineParams,
          skip_train : bool = False,
          skip_test : bool = False,
          output_name: str = "",
          ):

    with torch.no_grad():
        fps = 45

        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        # Get the poses
        original_cams = scene.getTrainCameras() # + scene.getTestCameras() # concat the lists
        original_cams = sorted(original_cams, key=lambda cam: cam.uid)
        poses = []
        for cam in original_cams:
            R_WC = cam.R # rotation matrix of T_WC (cam --> world)
            t = cam.T # translation of T_CW (world --> cam)
            W_t_C = - R_WC @ t # the inverse, translation of (cam --> world)

            # We need to save T_WC (cam --> world) for the poses
            pose  = np.eye(4)
            pose[:3, :3] = R_WC
            pose[:3, 3] = W_t_C
            poses.append(torch.from_numpy(pose))

        poses = torch.stack(poses, dim=0)

        ###
        # Displacements from the original trajectory
        ###
        # Rotation angles around axes
        angle_x_range = 5
        angle_y_range = 15
        angle_x_period = 400 # in terms of frames
        angle_y_period = 200

        # Displacement ranges as percentages of mean displacement in each direction between consecutive images
        disp_x_range = 5
        disp_y_range = -10
        disp_y_max = 5 # to prevent going below the road
        disp_x_period = 200 # in terms of frames
        disp_y_period = 300

        disp_delay=150

        ###
        # Loading and Rendering
        ###
        # Mean distance between consecutive poses for each coordinate direction in world frame
        mean_delta = (poses[1:,:3, 3] - poses[0:-1, :3, 3]).mean(dim=0, keepdim=True)

        new_poses = get_new_trajectory(
            poses,
            angle_x_range=angle_x_range,
            angle_y_range=angle_y_range,
            angle_x_period=angle_x_period,
            angle_y_period=angle_y_period,
            disp_x_range=disp_x_range,
            disp_y_range=disp_y_range,
            disp_x_period=disp_x_period,
            disp_y_period=disp_y_period,
            disp_y_max=disp_y_max,
            mean_delta=mean_delta,
            disp_delay=disp_delay,
            )

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # We need to reset the list of cameras that were read in with our new poses
        first_cam = scene.getTrainCameras()[0]
        FoVx = first_cam.FoVx
        FoVy = first_cam.FoVy
        dummy_image = torch.zeros_like(first_cam.original_image)
        images = []

        print("Rendering the images...")
        for idx, p in enumerate(tqdm(new_poses)):
            # Camera() expects R_WC (cam --> world) and t_cw (world --> cam)
            R_WC = p[:3, :3]
            t_cw = torch.linalg.inv(p)[:3,3].squeeze()

            view = Camera(
                    colmap_id=idx,
                    uid=idx,
                    R=R_WC.numpy(),
                    T=t_cw.numpy(),
                    FoVx=FoVx,
                    FoVy=FoVy,
                    image=dummy_image,
                    image_name="",
                    gt_alpha_mask=None,
                )

            img = render(view, gaussians, pipeline, background)["render"]
            images.append(to_pil_image(img.clamp(0,1).cpu()))

    print("Creating a video...")
    if output_name != "":
        output_name += "_"
    make_vid(images, fps=fps, output_path=f"{output_name}splat_video.mp4")

    print("Done!")


if __name__ == "__main__":
    # Inspired heavily from the render.py script in the original 3DGS repo
    parser = argparse.ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output_name", "-o", type=str)

    args = get_combined_args(parser)

    print("Rendering " + args.model_path)
    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        output_name=args.output_name,
        )




