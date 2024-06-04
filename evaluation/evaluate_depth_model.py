"""
Script for evaluating depth model predictions by comparing
dense depth predictions with ground truth KITTI360 data
"""
import json
import argparse
import logging
import datetime
import pprint
from pathlib import Path
from typing import NamedTuple

from tqdm import tqdm
from torch.utils.data import DataLoader

from configs.data import KITTI360_DIR, EVAL_DECIMAL_POINTS
from modules.io.datasets import KITTI360Dataset
from modules.depth.models import Metric3Dv2, KITTI360DepthModel
from modules.eval.metrics import (
    AverageDepthMetric,
    compute_absrel,
    compute_sqrel,
    compute_rmse,
    compute_rmse_log,
    compute_accuracy_threshold_1,
    compute_accuracy_threshold_2,
    compute_accuracy_threshold_3
)


def main(args):
    sequences = args.sequences
    cam_id = args.cam_id
    depth_model = args.depth_model.lower()
    model_variant = args.model_variant.lower()
    num_images = args.num_images
    batch_size = args.batch_size
    output_dir = Path(args.output)
    model = None
    gt_loader = None


    # Setup logging
    now = datetime.datetime.now()
    timestamp = now.strftime("%m-%d_%H-%M-%S")
    output_dir = output_dir / Path(f"depth_{timestamp}")
    log_path = output_dir.absolute() / Path("log.txt")
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(levelname)s - %(message)s')
    logging.info(f"Arguments: \n{json.dumps(vars(args), indent=4)}")

    # Used metrics which are averaged over sequences
    metrics = {
        "absrel": AverageDepthMetric(compute_absrel),
        "sqrel": AverageDepthMetric(compute_sqrel),
        "rmse": AverageDepthMetric(compute_rmse),
        "acc_1": AverageDepthMetric(compute_accuracy_threshold_1)
    }

    # Nested dictionary to store results, keys are metrics with dictionaries for sequences
    results = {}
    for metric_name in metrics.keys():
        results[metric_name] = {}

    for count, seq in enumerate(sequences):
        print(f"Processing sequence {seq} ({count + 1}/{len(sequences)})...")
        dataset = KITTI360Dataset(seq, cam_id, start=0, end=num_images)
        dataloader = DataLoader(dataset, batch_size, shuffle=False, drop_last=False)
        gt_loader = KITTI360DepthModel(dataset)

        if model is None and depth_model == "metric3d":
            model = Metric3Dv2(dataset.intrinsics, backbone=model_variant)
            model_variant = model._backbone
        if depth_model == "gt_depth":
            model = KITTI360DepthModel(dataset) # need to instantiate for every new dataset
            model_variant = "-"

        # Iterate over the images in batches for this sequence
        for frame_ids, images, _ in tqdm(dataloader):
            preds = model.predict({"images": images, "frame_ids": frame_ids})
            depths = preds["depths"]

            frame_ids = list(frame_ids)
            gt = gt_loader.predict({"images": images, "frame_ids": frame_ids})
            gt_depths = gt["depths"]

            for metric_name, metric in metrics.items():
                err = metric.update(depths, gt_depths)

        # Record results and reset metrics
        for metric_name, metric in metrics.items():
            logging.info(f"Sequence {seq}, {metric_name}: {metric.avg}")
            results[metric_name][f"seq_{seq}"] = round(metric.avg, EVAL_DECIMAL_POINTS)
            metric.reset()

    # Compute averages over sequences for each metric
    for metric_name in results.keys():
        total = 0.0
        count = len(sequences)
        for val in results[metric_name].values():
            total += val

        results[metric_name]["avg"] = total / count


    logging.info(f"Used model and variant: {depth_model}, {model_variant}")
    logging.info(f"Results: \n{pprint.pformat(results)}")
    return results


if __name__=="__main__":
    # parser = argparse.ArgumentParser(description="Extracts ground truth depth information from KITTI360 and compares it with Depth Model predictions")
    # parser.add_argument("--sequences", type=int, default=0, nargs="+", help="Sequence ids from KITTI360")
    # parser.add_argument("--cam_id", type=int, default=0, help="Camera id from KITTI360")
    # parser.add_argument("--depth_model", type=str, default="metric3d", required=True, help="Depth model to use")
    # parser.add_argument("--model_variant", type=str, default="vit_giant", required=False, help="Depth model variant to use")
    # parser.add_argument("--num_images", type=int, default=-1, help="Number of images to compute metrics and average over, enter -1 to use all images")
    # parser.add_argument("--batch_size", type=int, default=2, help="Number of images to predict depths for in one iteration")
    # parser.add_argument("--output", type=str, default="./evaluation/eval_results/", help="Path to save evaluation results")

    # args = parser.parse_args()

    # Useful when you want to debug
    class DebugArgs(NamedTuple):
        sequences = [0,3,4,5,6,7,9,10]
        cam_id = 0
        depth_model = "metric3d"
        model_variant="vit_small"
        num_images = 100
        batch_size = 4
        output="./evaluation/eval_results"
    def __repr__(self):
        return (f"DebugArgs(sequences={self.sequences}, cam_id={self.cam_id}, "
                f"depth_model={self.depth_model}, model_variant={self.model_variant}, "
                f"num_images={self.num_images}, batch_size={self.batch_size}, "
                f"output={self.output})")

    args = DebugArgs()

    main(args)
