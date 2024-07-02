import argparse
from pathlib import Path

import open3d as o3d
from modules.io.datasets import ColmapDataset


def main(args):
    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError

    output_path = input_path.parent / "points3D.ply"
    pcd = ColmapDataset.read_colmap_pcd_o3d(input_path)
    o3d.io.write_point_cloud(str(output_path), pcd.to_legacy())
    print(f"Wrote ply pointcloud at {str(output_path)}")


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", "-i", type=str, required=True, help="Path to points3D.txt in your colmap reconstruction")
    args = parser.parse_args()

    main(args)

