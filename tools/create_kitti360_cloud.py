import argparse
from pathlib import Path

from modules.io.datasets import CustomDataset, KITTIDataset, KITTI360Dataset
from modules.depth.models import KITTI360DepthModel
from modules.core.backprojection import Backprojector
from modules.core.reconstructors import SimpleReconstructor


def main(args):
    seq_id = args.seq_id
    cam_id = args.cam_id
    start = args.start
    end = args.end
    max_d = args.max_d
    output_dir = Path(args.output_dir)

    dataset = KITTI360Dataset(seq=seq_id, cam_id=cam_id, start=start, end=end)
    depth_model = KITTI360DepthModel(dataset)
    backprojector = Backprojector(cfg={"dropout":0.00, "max_d": max_d}, intrinsics=dataset.intrinsics)
    recon = SimpleReconstructor(dataset, backprojector, depth_model, cfg={"batch_size": 4, "output_dir": output_dir, "map_name": "gt_map.ply"})
    recon.run()

    print("done!")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Creates ground truth pointclouds for KITTI360 datasets")

    parser.add_argument("--seq_id", type=int, required=True, help="Sequence id")
    parser.add_argument("--cam_id", type=int, required=True, help="Cam id")
    parser.add_argument("--start", type=int, default=0, help="Start frame id of the chosen sequence")
    parser.add_argument("--end", type=int, default=-1, help="End frame id of the chosen sequence")
    parser.add_argument("--output_dir", "-o", type=str, default="./debug", help="Directory to save the output ground truth point cloud")
    parser.add_argument("--max_d", type=float, default=50, help="Maximum depth threshold for backprojection")

    args = parser.parse_args()
    main(args)


