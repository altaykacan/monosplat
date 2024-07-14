from typing import Tuple, Dict, List

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tv_F

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


def resize_image_torch(image: torch.Tensor, target_size: tuple, crop_box: tuple = ()) -> torch.Tensor:
        """
        Resize and crop the given image using torch interpolation and torchvision
        for cropping.

        Converts the `crop_box` notation to match the torchvision format.
        Coordinate system origin is at the top left corner, x is horizontal
        (pointing right), y is vertical (pointing down)

        Args:
            `images`: Unbatched tensors to resize and crop with shape `[C, H, W]`
            `target_size`: Size to resize to as a `(H, W)`-tuple
            `crop_box`: The box to crop the original images before resizing such that
                the aspect ratio of the to-be-transformed image becomes the same
                as the specified aspect ratio in `target_size`. Box is represented
                as `(left, upper, right, lower)` tuple
        """
        C, H, W = image.shape
        orig_size = (H, W)

        if orig_size == target_size:
            return image

        # Need to convert (left, upper, right, lower) to the torchvision format
        if crop_box != ():
            # Coordinates of the left top corner of the crop box
            left_top_corner_y = crop_box[1]
            left_top_corner_x = crop_box[0]

            box_height = crop_box[3] - crop_box[1]  # lower - upper
            box_width = crop_box[2] - crop_box[0]  # right - left
            image = tv_F.crop(
                image, left_top_corner_y, left_top_corner_x, box_height, box_width
            )

        # Using interpolate to have finer control, also RAFT doesn't work well with antialiased preprocessing
        resized_image = torch.nn.functional.interpolate(
            image.unsqueeze(0),
            target_size,
            mode="bilinear",
            antialias=False,
            align_corners=True,
        ).squeeze(0)

        return resized_image


def format_intrinsics(intrinsics: List[float]) -> torch.Tensor:
    """Formats the intrinsics to be a 3x3 projection matrix for a pinhole camera model"""
    fx, fy, cx, cy = intrinsics
    return torch.tensor(
        [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.double, requires_grad=False
    )


def compute_occlusions(flow0: torch.Tensor, flow1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Function to generate occlusion masks by checking forward and backward
    flow computations (`flow0`: image 1 -> image 2 and `flow1`: image 2 -> image 1)

    The outputs `mask0` represents a boolean mask with pixels in image 1 not
    visible in image 2 set to False (occlusion_mask).
    On the other hand, `mask1` is a boolean mask with pixels in image 2 not
    visible in image 1 (disocclusion mask).

    Code provided by Felix Wimbauer (https://github.com/Brummi)

    Indexing logic for `mask0` and `mask1`
    - The first dimension of mask0 is indexed with nxy_1[:, 0, :].long(),
    which represents the batch indices. This ensures that the operation is
    applied to the correct element in the batch.
    - The second dimension is hardcoded to 0 because mask0 is prepared to have
    a single channel, and this operation is intended to fill in this channel.
    - The third dimension (height) is indexed with
    ((nxy_1[:, 2, :] * .5 + .5) * h).round().long().clamp(0, h-1).
    This expression takes the y-coordinates from nxy_1 (after reverse flow adjustment),
    normalizes them to the range [0, h-1], rounds them to the nearest integer,
    and clamps the values to ensure they are within the valid range of
    indices for the height dimension.
    - The fourth dimension (width) is indexed similarly but uses the
    x-coordinates from nxy_1: ((nxy_1[:, 1, :] * .5 + .5) * w).round().long().clamp(0, w-1).
    This normalizes the x-coordinates to the range [0, w-1] (where w is the
    width of the image), rounds, and clamps them.
    """
    n, _, h, w = flow0.shape
    device = flow0.device
    x = torch.linspace(-1, 1, w, device=device).view(1, 1, w).expand(1, h, w) # [1, H, W]
    y = torch.linspace(-1, 1, h, device=device).view(1, h, 1).expand(1, h, w) # [1, H, W]
    xy = torch.cat((x, y), dim=0).view(1, 2, h, w).expand(n, 2, h, w) # does what torch.meshgrid does, pixel coordinates between -1 and 1

    flow0_r = torch.cat((flow0[:, 0:1, :, :] * 2 / w , flow0[:, 1:2, :, :] * 2 / h), dim=1) # flow in normalized pixel coordinates
    flow1_r = torch.cat((flow1[:, 0:1, :, :] * 2 / w , flow1[:, 1:2, :, :] * 2 / h), dim=1)

    xy_0 = xy + flow0_r # new normalized coordinates after applying forward flow
    xy_1 = xy + flow1_r # after applying reverse flow

    xy_0 = xy_0.view(n, 2, -1) # [n, 2, num_pixels]
    xy_1 = xy_1.view(n, 2, -1) # [n, 2, num_pixels]

    ns = torch.arange(n, device=device, dtype=xy_0.dtype) # batches, [0, 1, 2, ..., n-1]
    nxy_0 = torch.cat((ns.view(n, 1, 1).expand(-1, 1, xy_0.shape[-1]), xy_0), dim=1) # additional number for batch idx [n, 3, num_points]
    nxy_1 = torch.cat((ns.view(n, 1, 1).expand(-1, 1, xy_1.shape[-1]), xy_1), dim=1)

    mask0 = torch.zeros_like(flow0[:, :1, :, :])
    mask0[nxy_1[:, 0, :].long(), 0, ((nxy_1[:, 2, :] * .5 + .5) * h).round().long().clamp(0, h-1), ((nxy_1[:, 1, :] * .5 + .5) * w).round().long().clamp(0, w-1)] = 1

    mask1 = torch.zeros_like(flow1[:, :1, :, :])
    mask1[nxy_0[:, 0, :].long(), 0, ((nxy_0[:, 2, :] * .5 + .5) * h).round().long().clamp(0, h-1), ((nxy_0[:, 1, :] * .5 + .5) * w).round().long().clamp(0, w-1)] = 1

    return mask0.bool(), mask1.bool()


def get_correspondences_from_flow(flow: torch.Tensor, H: int, W: int):
    """
    Computes a correspondence tensor using the optical flow between
    two images using `torch.nn.functional.grid_sample()`. The correspondences
    have the shape `[N, 2, H, W]` and contain the new x and y coordinates that
    the flow vectors point to from image 1 to image 2. Nearest neighbor
    interpolation is used with `grid_sample()` to get integer pixel coordinates
    in the second image. Example:
    The position `(n, 1, i, j)` in the `correspondences` tensor has the `u`
    (horizontal pixel coordinate) the pixel (i, j) in image 1 gets mapped to
    in image 2 if you follow the flow vector at that position from image 1 to
    image 2. The index `n` is used to index the images within the batch.

    The flow values are added to the original coordinates and the new
    coordinates are determined by taking the nearest coordinate point from the
    second image. The points that fall out of bounds after adding the flow are
    set to have the value zero.

    PyTorch needs (N, C, H_in, W_in) input and a (N, H_out, W_out, 2) grid
    for grid_sample where N is the batch size and C is the channel amount.

    Args:
        flow (Tensor): Tensor representing the optical flow, shape [N, 2, H, W]
        H (int): Height of the image
        W (int): Width of the image
    """
    N, _, H, W = flow.shape
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Pixel coordinates, U and V are both [H, W]
    U, V = torch.meshgrid([torch.arange(0, W), torch.arange(0, H)], indexing="xy")
    UV = torch.stack((U, V), dim=2)  # [H, W, 2]
    UV_source = UV.permute(2, 0, 1).unsqueeze(0)  # [1, 2, H, W], input
    UV_source = UV_source.repeat(N, 1, 1, 1).float().to(device) # [N, 2, H, W]
    UV_target = UV.unsqueeze(0).to(device)  # [1, H, W, 2], original positions

    flow_in = flow.permute(0, 2, 3, 1).to(device)  # [N, H, W, 2], flow vector field
    grid = UV_target + flow_in  # new positions after flow

    # Grid needs to be normalized to [-1, 1] (normalized by image dimensions)
    # (x - 0) / (W - 0) == (x_n - (-1)) / (1 - (-1))
    grid[:, :, :, 0] = 2 * grid[:, :, :, 0] / (W - 1) - 1
    grid[:, :, :, 1] = 2 * grid[:, :, :, 1] / (H - 1) - 1

    # F.grid_sample already takes care of the reversed direction of the flows
    # [N, 2, H, W]
    correspondences = F.grid_sample(UV_source, grid, mode="nearest", padding_mode="zeros", align_corners=True)

    return correspondences


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