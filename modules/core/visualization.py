from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image, to_tensor
import matplotlib.pyplot as plt


def visualize_flow(flow: torch.Tensor, original_image: torch.Tensor = None, flow_vis_step=15, output_path="debug_flow"):
    """
    Function to visualize optical flow vector fields on top of an optional original image.
    The resulting visualization is saved as specified in `output_path`. The argument
    `flow_vis_step` determines the pixel spacing between the visualized flow vectors,
    otherwise the flow field visualization is too dense.
    """
    N, _, H, W = flow.shape

    if N != 1:
        raise ValueError("visualize_flow only implemented for batch sizes of one, please give your optical flow predictions one by one!")

    X_f, Y_f = torch.meshgrid(
                [
                    torch.arange(0, W, flow_vis_step),
                    torch.arange(0, H, flow_vis_step),
                ],
                indexing="xy",
            )  # 2 [H,W] arrays

    flow_c = flow.clone() # clone so we can modify the visualization if needed for debugging

    # Optional code to modify the flow field, useful for debugging
    #flow_c[0, 0, :, 500:] = 0
    #flow_c[0, 1, :, 500:] = 0

    dX = flow_c[0, 0, :: flow_vis_step, :: flow_vis_step].cpu().detach()
    dY = flow_c[0, 1, :: flow_vis_step, :: flow_vis_step].cpu().detach()

    plt.figure(dpi=300)
    plt.quiver(
        X_f, Y_f, dX, dY, angles="xy", scale_units="xy", scale=1, color="red", width=0.001
    )

    if original_image is not None:
        original_image_pil = to_pil_image(original_image.squeeze()) # only for batch size of 1
        plt.imshow(original_image_pil)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(Path(f"{output_path}.png"))
    plt.close()