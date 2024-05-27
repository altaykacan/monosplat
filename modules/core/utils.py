from typing import Tuple

import torch

def compute_occlusions(flow0: torch.Tensor, flow1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Function to generate occlusion masks by checking forward and backward
    flow computations (`flow0`: image 1 -> image 2 and `flow1`: image 2 -> image 1)

    The outputs `mask0` represents a boolean mask with pixels in image 1 not
    visible in image 2 set to False. On the other hand, `mask1` is a boolean
    mask with pixels in image 2 not visible in image 1.

    Code provided by Felix Wimbauer (https://github.com/Brummi)
    """
    n, _, h, w = flow0.shape
    device = flow0.device
    x = torch.linspace(-1, 1, w, device=device).view(1, 1, w).expand(1, h, w) # [1, H, W]
    y = torch.linspace(-1, 1, h, device=device).view(1, h, 1).expand(1, h, w) # [1, H, W]
    xy = torch.cat((x, y), dim=0).view(1, 2, h, w).expand(n, 2, h, w) # does what torch.meshgrid does, pixel coordinates between -1 and 1

    flow0_r = torch.cat((flow0[:, 0:1, :, :] * 2 / w , flow0[:, 1:2, :, :] * 2 / h), dim=1) # flow in normalized pixel coordinates
    flow1_r = torch.cat((flow1[:, 0:1, :, :] * 2 / w , flow1[:, 1:2, :, :] * 2 / h), dim=1)

    xy_0 = xy + flow0_r # new normalized coordinates after applying flow
    xy_1 = xy + flow1_r

    xy_0 = xy_0.view(n, 2, -1) # [n, 2, num_pixels]
    xy_1 = xy_1.view(n, 2, -1) # [n, 2, num_pixels]

    ns = torch.arange(n, device=device, dtype=xy_0.dtype)
    nxy_0 = torch.cat((ns.view(n, 1, 1).expand(-1, 1, xy_0.shape[-1]), xy_0), dim=1)
    nxy_1 = torch.cat((ns.view(n, 1, 1).expand(-1, 1, xy_1.shape[-1]), xy_1), dim=1)

    # Set the values at transformed coordinates to 1 (zeros are where there has been no flow)
    mask0 = torch.zeros_like(flow0[:, :1, :, :])
    mask0[nxy_1[:, 0, :].long(), 0, ((nxy_1[:, 2, :] * .5 + .5) * h).round().long().clamp(0, h-1), ((nxy_1[:, 1, :] * .5 + .5) * w).round().long().clamp(0, w-1)] = 1

    mask1 = torch.zeros_like(flow1[:, :1, :, :])
    mask1[nxy_0[:, 0, :].long(), 0, ((nxy_0[:, 2, :] * .5 + .5) * h).round().long().clamp(0, h-1), ((nxy_0[:, 1, :] * .5 + .5) * w).round().long().clamp(0, w-1)] = 1

    return mask0.bool(), mask1.bool()