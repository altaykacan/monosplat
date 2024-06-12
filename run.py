import subprocess

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(
    version_base=None,
    config_path="./configs/",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    # TODO add the commands for each of the scripts we have 1-7 and use the hydra
    # configs to get the command line arguments for them
    pass
    # Example usage of subprocess:
    # # Define the command
    # command = ["python3", "/path/to/script.py", "arg1", "arg2"]

    # # Run the command
    # subprocess.run(command)
    #
    # Commands
    # python 1_extract_frames_from_video.py --video_path /usr/stud/kaa/data/root/ds01/data/GX010061.MP4 --resize_factor 2
    # python 2_run_colmap.py  ...

if __name__=="__main__":
    main()