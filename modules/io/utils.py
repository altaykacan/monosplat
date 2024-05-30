from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt

def save_image_torch(tensor, name="debug"):
    """
    Saves a torch tensor representing an image into disk, useful for debugging
    Only saves the first sample in batched input
    """
    if len(tensor.shape) == 4: # ignore rest of the batch
        tensor = tensor[0, :, :, :]

    tensor = tensor.squeeze()

    if len(tensor.shape) == 3: # rgb image
        plt.imsave(f"{name}.png", tensor.detach().cpu().permute(1,2,0).numpy())
    if len(tensor.shape) == 2: # mask or 1-channel image
        plt.imsave(f"{name}.png", tensor.detach().cpu().numpy())


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
    files. Useful for scripts that compute some values and save them for later
    usage.
    """
    if isinstance(dir_path, str):
        dir_path = Path(dir_path)

    not_empty = any(dir_path.iterdir())

    if not_empty:
        while True:
            answer = input(f"The directory you specified {str(dir_path)} is not empty, do you want to delete existing files before continuing? [y/n]: ")

            if answer.lower() == "y":
                return True
            elif answer.lower() == "n":
                print(f"Not deleting existing files at {str(dir_path)}...")
                return False

            print("Please type 'y' or 'n'")



