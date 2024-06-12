from typing import List, Dict

import torch

from ..core.constants import CITYSCAPES_LABELS


def mmseg_get_class_ids(class_names: List[str], dataset: str = "CITYSCAPES") -> List[int]:
    """
    Converts a list of class names (strings) into a list of corresponding
    index integers for mmseg. Unmatched classes get an id of -1.

    If `class_names` is None an empty list of class ids is returned.
    """
    class_ids = []
    if class_names is not None:
        if dataset == "CITYSCAPES":
            for name in class_names:
                match_found = False
                for id, label in enumerate(CITYSCAPES_LABELS):
                    if label == name:
                        class_ids.append(id)
                        match_found = True
                        break

                if not match_found:
                    class_ids.append(-1)

        else:
            raise NotImplementedError(
                f"Dataset '{dataset}' not recognized, try 'CITYSCAPES'"
            )

    return class_ids


def combine_segmentation_masks(masks_dict: Dict, combine_list: List[str] = []) -> torch.Tensor:
    """
    Combines the standard output of `SegmentationModel` classes into a single
    batched mask.

    Args:
        - `masks_dict`: A dictionary which has the class names as keys and the batched
            mask tensors as the values.
        - `combine_list`: The list of class names that should be combined into a
            single mask. Leave as an empty list to combine all masks.
    """
    mask_names = list(masks_dict.keys())
    if combine_list == []:
        combine_list = mask_names # combine all masks if combine_list is empty

    final_mask = torch.zeros_like(masks_dict[combine_list[0]])

    for class_name in combine_list:
        if not (class_name in masks_dict.keys()):
            raise ValueError(f"The class name {class_name} cannot be found in the provided masks. Check your input to 'combine_segmentation_masks'!")

        final_mask = torch.logical_or(masks_dict[class_name], final_mask) # aggregating the masks

    return final_mask
