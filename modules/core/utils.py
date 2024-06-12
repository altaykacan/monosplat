from typing import Tuple, Dict

import torch
import torch.nn.functional as F


def compute_target_intrinsics(
        orig_intrinsics: Tuple[float, float, float, float],
        orig_size: Tuple[int, int],
        target_size:Tuple[int, int]
        ) -> Tuple[Tuple[float, float, float, float], Tuple[int, int, int, int]]:
    """
    Computes new intrinsics after resizing an image to a given target size.
    The function computes how the original image should be cropped such that the
    aspect ratio (`H/W`) of the cropped image matches the ratio of `target_size`
    specified by the user. Returns new intrinsics and the crop box as two
    separate tuples. The crop box should be used to crop the input image before
    resizing.

    Code inspired from `compute_target_intrinsics()` from MonoRec: https://github.com/Brummi/MonoRec/blob/81b8dd98b86e590069bb85e0c980a03e7ad4655a/data_loader/kitti_odometry_dataset.py#L318

    Returns:
        `intrinsics`: The computed intrinsics for the resize `(fx, fy, cx, cy)`
        `crop_box`: The crop rectangle to preserve ratio of height/width
                of the target image size as an integer `(left, upper, right, lower)`-tuple.
    """
    height, width = orig_size
    height_target, width_target = target_size

    # Avoid extra computation if the original and target sizes are the same
    if orig_size == target_size:
        target_intrinsics = orig_intrinsics
        crop_box = ()
    else:
        r = height / width
        r_target = height_target / width_target
        fx, fy, cx, cy = orig_intrinsics

        if r >= r_target:
            # Width stays the same, we compute new height to keep target ratio
            new_height = r_target * width
            crop_box = (
                0,
                (height - new_height) // 2,
                width,
                height - (height - new_height) // 2,
            )

            rescale = width / width_target  # equal to new_height / height_target
            cx = cx / rescale
            cy = (cy - (height - new_height) / 2) / rescale # need to shift center due to cropping

        else:
            # Height stays the same, we compute new width to keep target ratio
            new_width = height / r_target
            crop_box = (
                (width - new_width) // 2,
                0,
                width - (width - new_width) // 2,
                height,
            )

            rescale = height / height_target  # equal to new_width / width_target
            cx = (cx - (width - new_width) / 2) / rescale
            cy = cy / rescale

        # Rescale the focal lenghts
        fx = fx / rescale
        fy = fy / rescale

        # Set the attributes of the dataset class
        target_intrinsics = (fx, fy, cx, cy)
        crop_box = tuple([int(coord) for coord in crop_box]) # should only have ints

    return target_intrinsics, crop_box

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

def unravel_batched_pcd_tensor(batched_tensor: torch.Tensor) -> torch.Tensor:
    """
    Unravels a batched tensor representing point clouds and concatenates
    the samples into the last dimension (representing number of points).

    Expects a `[N, C, num_el]` tensor and returns a  `[C, N * num_el]` tensor.
    """
    if len(batched_tensor.shape) != 3:
        raise ValueError(f"Input is expected to have shape '[N, C, num_el]' but found shape {batched_tensor.shape}")

    N = batched_tensor.shape[0]

    individual_tensors = []
    for batch_idx in range(N):
        individual_tensors.append(batched_tensor[batch_idx, ...])

    return torch.cat(individual_tensors, dim=1)


def shrink_bool_mask(mask: torch.Tensor, iterations: int = 1, kernel_size: int = 3):
    """
    Shrinks a given mask (reduces the number of True pixels) starting
    from the borders of the mask by repeatedly applying min pooling to an image
    `iterations` many times. This is useful to make the system more robust
    to inaccuracies or missed pixels in the masks.

    Since `torch` has no explicit min pooling the mask is first inverted and
    max pooling is used.

    `kernel_size` has to be an odd number for the same padding to work
    """
    if not kernel_size % 2 == 1:
        raise ValueError("kernel_size for shrink_bool_mask() has to be an odd number!")

    mask_inv = torch.logical_not(mask)

    # Padding to keep the spatial shape the same
    padding = int((kernel_size - 1) / 2)

    # TODO figure out a better way instead of using stride 1 to keep spatial dims the same
    for i in range(iterations):
        mask_inv = F.max_pool2d(mask_inv.float(), kernel_size=kernel_size, stride=1, padding=padding)
        mask_inv = mask_inv.bool()

    # Invert again to get original mask
    mask = torch.logical_not(mask_inv)

    return mask


def grow_bool_mask(mask: torch.Tensor, iterations: int = 1, kernel_size: int = 3):
    """
    Grows the region with True values of a boolean tensor by applying max pooling
    `iterations` many times. Does the opposite `shrink_bool_mask()`
    """
    if not kernel_size % 2 == 1:
        raise ValueError("kernel_size for grow_bool_mask() has to be an odd number!")

    # Padding to keep the spatial shape the same
    padding = int((kernel_size - 1) / 2)

    # TODO figure out a better way instead of using stride 1 to keep spatial dims the same
    for i in range(iterations):
        mask = F.max_pool2d(mask.float(), kernel_size=kernel_size, stride=1, padding=padding)
        mask = mask.bool()

    return mask