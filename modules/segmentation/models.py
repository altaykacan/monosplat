from typing import Dict

import torch

from ..core.interfaces import BaseModel
from .utils import mmseg_get_class_ids

try:
    from torchvision.models.detection import  maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
    from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
    torchvision_found = True
except ImportError:
    torchvision_found = False

try:
    from mmengine.model.utils import revert_sync_batchnorm
    from mmseg.apis import init_model, inference_model, show_result_pyplot
    mmseg_found = True
except ImportError:
    mmseg_found = False


class InstanceSegmentationModel(BaseModel):
    """
    Parent class for all instance segmentation models.

    Expected inputs and outputs:
        `intput_dict`: ...
        `output_dict`: ...
    """
    def _check_input(self, input_dict: Dict):
        if "images" not in input_dict.keys():
            raise ValueError("Couldn't find the key 'images' in your input_dict, check your input to the model!")
        if "classes_to_detect" not in input_dict.keys():
            raise ValueError("Couldn't find the key 'classes_to_detect' in your input_dict, check your input to the model!")
        if len(input_dict["images"].shape) != 4:
            raise ValueError("Expected (N, C, H, W) input, the shape of your inputs do not match!")
        if not isinstance(input_dict["classes_to_detect"], list):
            raise ValueError("Expected 'classes_to_detect' to be a list, check your input to the model!")

    def _check_output(self, output_dict: Dict):
        if "instance_masks" not in output_dict.keys():
            raise ValueError("Couldn't find the key 'instance_masks' in your output_dict, check your prediction function!")
        if "boxes" not in output_dict.keys():
            raise ValueError("Couldn't find the key 'boxes' in your output_dict, check your prediction function!")
        if not isinstance(output_dict["masks"], list):
            raise ValueError("Expected 'masks' to be a list of tensors, check your prediction function!")
        if not isinstance(output_dict["boxes"], list):
            raise ValueError("Expected 'boxes' to be a list of tensors, check your prediction function!")


class SegmentationModel(BaseModel):
    """
    Parent class for all semantic segmentation models.

    Expected inputs and outputs:
        `intput_dict`: ...
        `output_dict`: ...
    """
    def _check_input(self, input_dict: Dict):
        if "images" not in input_dict.keys():
            raise ValueError("Couldn't find the key 'images' in your input_dict, check your input to the model!")
        if "classes_to_segment" not in input_dict.keys():
            raise ValueError("Couldn't find the key 'classes_to_segment' in your input_dict, check your input to the model!")
        if len(input_dict["images"].shape) != 4:
            raise ValueError("Expected (N, C, H, W) input, the shape of your inputs do not match!")
        if not isinstance(input_dict["classes_to_segment"], list):
            raise ValueError("Expected 'classes_to_segment' to be a list, check your input to the model!")

    def _check_output(self, output_dict: Dict):
        assert "masks_dict" in output_dict.keys(), "Couldn't find the key 'masks_dict' in your output_dict, check your prediction function!"


class MaskRCNN(InstanceSegmentationModel):
    """
    Mask RCNN from torchvision.

    Expected inputs and outputs for `predict()`:
    `intput_dict`: `{"images": torch.Tensor, "classes_to_detect": List[str]}`
    `output_dict`: `{"boxes": List[torch.Tensor], "scores": List[torch.Tensor], "masks": List[torch.Tensor]}`
    """
    def __init__(self, cfg: Dict = {}):
        if not torchvision_found:
            raise ImportError("Couldn't find torchvision. Please install it before using this class")

        self._weights_config = cfg.get("torchseg_weights", "DEFAULT")

        self._model = None
        self._weights = None
        self._device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        self._detection_threshold = cfg.get("torchseg_det_threshold", 0.8)
        self._prob_threshold = cfg.get("torchseg_prob_threshold", 0.2)


    def load(self):
        if self._model is None and self._weights is None:
            if self._weights_config == "DEFAULT":
                weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
            else:
                raise NotImplementedError("Weight configurations other than 'DEFAULT' are not implemented for Mask RCNN models")

            model = maskrcnn_resnet50_fpn(weights=weights, progress=False)

            model.eval()
            model.to(self._device)

            self._model = model
            self._weights = weights

    def unload(self):
        self._model = None
        self._weights = None
        torch.cuda.empty_cache()

    def _preprocess(self, input_dict: Dict) -> Dict:
        images = input_dict["images"]
        transforms = self._weights.transforms()
        images = images.to(self._device)

        input_dict["images"] = transforms(images) # to not overwrite other inputs

        return input_dict

    def _predict(self, input_dict: Dict) -> Dict:
        images = input_dict["images"]
        images = images.to(self._device)
        classes_to_detect = input_dict["classes_to_detect"]

        categories = self._weights.meta["categories"]

        predictions = self._model(images)

        labels = []
        boxes = []
        masks = []

        for pred in predictions:
            pred_labels = []
            pred_boxes = []
            pred_masks = []

            for i, curr_label in enumerate(pred["labels"]):
                if categories[curr_label] in classes_to_detect and pred["scores"][i] > self._detection_threshold:
                    pred_labels.append(categories[curr_label])
                    pred_boxes.append(pred["boxes"][i])
                    pred_masks.append((pred["masks"][i] > self._prob_threshold).unsqueeze(0))


            labels.append(pred_labels)
            boxes.append(pred_boxes)
            masks.append(pred_masks)

        return {"boxes": boxes, "labels": labels, "masks": masks}


# TODO implement
class FPN(SegmentationModel):
    def __init__(self, cfg: Dict):
        pass


class SegFormer(SegmentationModel):
    """
    SegFormer model for semantic segmentation using MMSeg library.

    Expected inputs and outputs:
    `intput_dict`: `{"images": torch.Tensor (N, 3, H, W), "classes_to_segment": List[str]}`
    `output_dict`: `{"masks_dict": dictionary of (N, 1, H, W) torch tensors where the keys are the names of the classes as strings}`
    """
    def __init__(self, cfg: Dict = {}):
        if not mmseg_found:
            raise ImportError("Couldn't find mmseg. Please check you have mmseg properly installed (pay attention to supported PyTorch versions) before using this class.")

        # TODO figure out if these paths work properly
        # These are intended to be called from a main function/script running in the base directory of the project
        default_checkpoint = "./checkpoints/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth"
        default_config =  "./configs/thirdparty/mmseg/segformer/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py"
        default_dataset = "CITYSCAPES"

        self._checkpoint = cfg.get("mmseg_checkpoint", default_checkpoint)
        self._config = cfg.get("mmseg_config", default_config)
        self._dataset = cfg.get("mmseg_dataset", default_dataset)

        self._model = None
        self._device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")


    def load(self):
        if self._model is None:
            self._model = init_model(self._config, self._checkpoint, self._device)

            if self._device == "cpu":
                self._model = revert_sync_batchnorm(self._model) # need to convert SyncBatchnorm layers to standard ones for cpu

            self._model.eval()


    def unload(self):
        self._model = None
        torch.cuda.empty_cache()

    def _preprocess(self, input_dict: Dict) -> Dict:
        return input_dict


    def _predict(self, input_dict: Dict) -> Dict:
        images = input_dict["images"]
        classes_to_segment = input_dict["classes_to_segment"]

        # inference_model from mmseg wants a list of images as numpy (H, W, C) arrays
        images_unstacked = torch.unbind(images, dim=0)
        images_list = [image.permute(1,2,0).cpu().numpy() for image in images_unstacked]
        results = inference_model(self._model, images_list)
        preds = []
        for result in results:
            preds.append(result.pred_sem_seg.data)

        preds = torch.stack(preds, dim=0) # get batch dimension again

        class_ids = mmseg_get_class_ids(classes_to_segment, self._dataset)

        masks_dict = {}
        for id, name in zip(class_ids, classes_to_segment):
            masks_dict[name] = (preds == id)


        return {"masks_dict": masks_dict}




