import sys
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import NamedTuple
from argparse import ArgumentParser

from modules.io.utils import find_latest_number_in_dir


# TODO we need to make sure that we take care of the fact that 3DGS only works for simple pinhole cameras where the cx and cy are at the middle
def main(args, extra_args):
    root_dir = Path(args.root_dir)
    recon_name = args.recon_name
    splat_name = args.splat_name
    iterations = args.iterations
    colmap_dir = Path(args.colmap_dir) if args.colmap_dir is not None else None
    use_every_nth_as_val = args.use_every_nth_as_val

    if not root_dir.exists():
        raise RuntimeError(f"Your root_dir at '{str(root_dir)}' does not exist! Please make sure you give the right path.")

    image_dir = root_dir / "data" / "rgb"
    recon_dir = root_dir / Path(f"reconstructions/{recon_name}")
    splat_dir = root_dir / Path("splats")
    splat_dir.mkdir(exist_ok=True, parents=True)

    # To enumerate the files
    largest_idx = find_latest_number_in_dir(splat_dir)
    if splat_name is None:
        splat_name = f"{largest_idx + 1}_splat"
    else:
        splat_name = f"{largest_idx + 1}_{splat_name}"

    output_dir = splat_dir / Path(splat_name)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Setup logging
    log_path = output_dir / Path("log_splat.txt")
    log_time = datetime.now().strftime('%Y-%m-%d:%H-%M-%S')
    with open(log_path, 'w'): # to clear existing logs
        pass
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(levelname)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(f"Log file for '7_train_gaussians.py', created at (year-month-day:hour-minute-second): {log_time}")
    logging.info(f"Arguments: \n{json.dumps(vars(args), indent=4)}")

    if colmap_dir is None:
        input_path = str(recon_dir)
        image_path = str(image_dir)
    else:
        logging.info(f"COLMAP directory is given as '{str(colmap_dir)}', training the 3D Gaussian Splatting model on COLMAP data.")
        input_path = str(colmap_dir)
        image_path = str(colmap_dir / "images")

    output_path = str(output_dir)

    # To be able to call the script in the submodule
    sys.path.append("./submodules/gaussian-splatting/")
    gaussian_train_path = "./submodules/gaussian-splatting/train.py"

    command = [
        "python",
        gaussian_train_path,
        "--source_path",
        input_path,
        "--images",
        image_path,
        "--iterations",
        str(iterations),
        "-m",
        output_path,
        "--eval",
        "--scale_depths",
        "--use_inverse_depth",
        "--llffhold",
        str(use_every_nth_as_val),
        ]

    # Append extra arguments to the command
    command.extend(extra_args)

    logging.info(f"The command used to run 3DGS: {' '.join(command)}")
    subprocess.run(command, check=True)

    print("Done!")

if __name__ == "__main__":
    parser = ArgumentParser(description="Train a gaussian splatting model using the 'train.py' script in './submodules/gaussian-splatting'")

    parser.add_argument("--root_dir", "-r", type=str, required=True, help="Path to the root directory of your dataset")
    parser.add_argument("--recon_name", "-i", type=str, help="Name of the reconstruction you want to use in the './root_dir/reconstructions' directory")
    parser.add_argument("--splat_name", "-o", type=str, default=None, help="Name of the saved gaussian splat")
    parser.add_argument("--iterations", type=int, default=30000, help="Number of iterations to train gaussian model")
    parser.add_argument("--colmap_dir", type=str, default=None, help="Path to the COLMAP directory. If you provide this the other options will be ignored and 3DGS will be ran on default COLMAP results")
    parser.add_argument("--use_every_nth_as_val", type=int, default=10, help="Every nth image will be saved to form a validation set for evaluation.")

    args, extra_args = parser.parse_known_args()
    main(args, extra_args)


