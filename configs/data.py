# Where your KITTI360 directory is
KITTI360_DIR = "/usr/stud/kaa/data/KITTI-360"

# Total character size of the filenames used to save/read images
PADDED_IMG_NAME_LENGTH = 5 # for colmap and custom datasets
# PADDED_IMG_NAME_LENGTH = 6 # for KITTI
# PADDED_IMG_NAME_LENGTH = 10 # for KITTI360

# Where you want the outputs to be saved
OUTPUT_DIR = "/usr/stud/kaa/data/root/ds01"

# List of supported datasets
SUPPORTED_DATASETS = ["custom", "kitti", "kitti360", "colmap", "tum_rgbd"]

# Number of decimal points to round up to when logging evaluation results
EVAL_DECIMAL_POINTS = 5