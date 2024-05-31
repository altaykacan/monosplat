import logging
import shutil
from pathlib import Path
from typing import Union

import torch
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)

def save_image_torch(tensor: torch.Tensor, name: str = "debug", output_dir: Union[str, Path] = "."):
    """
    Saves a torch tensor representing an image into disk, useful for debugging.
    Only saves the first sample in batched input
    """
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    if not output_dir.is_dir():
        logging.warning(f"Output directory for saving tensors '{str(output_dir)}' does not exist. Creating it...")
        output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / Path(f"{name}.png")

    if len(tensor.shape) == 4: # ignore rest of the batch
        tensor = tensor[0, :, :, :]
    tensor = tensor.squeeze()

    if len(tensor.shape) == 3: # rgb image
        plt.imsave(output_path, tensor.detach().cpu().permute(1, 2,0).numpy())
    if len(tensor.shape) == 2: # binary mask or 1-channel image
        plt.imsave(output_path, tensor.detach().cpu().numpy())


def find_latest_number_in_dir(dir_path: Union[str, Path]) -> int:
    """
    Finds and returns the largest number in a directory that contains
    enumerated subdirectories with the naming convention `xx_some_directory_name`
    where `xx` represents the numbering of the run. Useful for keeping track
    of reconstruction or gaussian splat training runs.
    """
    largest_idx = 0

    if isinstance(dir_path, str):
        dir_path = Path(dir_path)

    for path in dir_path.iterdir():
        path_str = str(path)

        # Don't count hidden files
        if path_str[0] == ".":
            continue

        # Expecting xx_some_directory_name as a naming convention where xx is a number
        idx = int(path_str.split("_")[0])
        if idx > largest_idx:
            largest_idx = idx

    return largest_idx


def ask_to_clear_dir(dir_path: Union[str, Path]) -> bool:
    """
    Asks the user to delete existing files in a directory if there are existing
    files. If the user types in 'y' the files are deleted.
    Useful for scripts that compute some values and save them for later usage.
    Returns a boolean representing whether the operation should be continued
    or not.
    """
    if isinstance(dir_path, str):
        dir_path = Path(dir_path)

    not_empty = any(dir_path.iterdir())
    should_continue = True

    if not_empty:
        while True:
            answer = input(f"The directory you specified {str(dir_path)} is not empty, do you want to delete existing files before continuing? [y/n]: ")

            if answer.lower() == "y":
                print(f"Deleting existing files at {str(dir_path)}...")
                for file in dir_path.iterdir():
                    if file.is_file():
                        file.unlink()
                should_continue = True
                break
            elif answer.lower() == "n":
                print(f"Not deleting existing files at {str(dir_path)} and aborting...")
                should_continue = False
                break

            print("Please type 'y' or 'n'")

    return should_continue



