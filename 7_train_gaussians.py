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

def main(args):
    root_dir = Path(args.root_dir)
    recon_name = args.recon_name
    splat_name = args.splat_name
    iterations = args.iterations

    if not root_dir.exists():
        raise RuntimeError(f"Your root_dir at '{str(root_dir)}' does not exist! Please make sure you give the right path.")

    recon_dir = root_dir / Path(f"reconstructions/{recon_name}")
    splat_dir = root_dir / Path("splats")
    splat_dir.mkdir(exist_ok=True, parents=True)

    # To enumerate the files
    largest_idx = find_latest_number_in_dir(splat_dir)
    if splat_name is None:
        splat_name = f"{largest_idx + 1}_splat"
    else:
        splat_name = f"{largest_idx + 1}_{splat_name}"

    output_dir = splat_dir / Path("splat_name")
    log_dir = output_dir / Path("log")

    # Setup logging
    log_path = log_dir.parent / Path("log_scale_alignment.txt")
    log_time = datetime.now().strftime('%Y-%m-%d:%H-%M-%S')
    with open(log_path, 'w'): # to clear existing logs
        pass
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(levelname)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(f"Log file for '7_train_gaussians.py', created at (year-month-day:hour-minute-second): {log_time}")
    logging.info(f"Arguments: \n{args}") # TODO remove after debug
    # logging.info(f"Arguments: \n{json.dumps(vars(args), indent=4)}")

    input_path = "/usr/stud/kaa/data/splats/custom_new/ds_combined_colmap"
    image_path = "/usr/stud/kaa/data/root/ds_combined/data/rgb"
    iterations = "2000"
    save_iterations = ["1000", "1500"]
    output_path = "./output/splat demo"

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
        iterations,
        "--save_iterations",
        *save_iterations,
        "-m",
        output_path
        ]

    # Append terms to the command depending on regularization options
    # TODO add the code here for the command options


    logging.info(f"The command used to run 3DGS: {' '.join(command)}")
    subprocess.run(command, check=True)

    print("Done!")

if __name__ == "__main__":
    parser = ArgumentParser(description="Train a gaussian splatting model using the 'train.py' script in './submodules/gaussian-splatting'")

    class DebugArgs(NamedTuple):
        root_dir = ""
    args = DebugArgs()
    main(args)


