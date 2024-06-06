# Labels taken from mmsegmentation CityScapesDataset defaults here: https://github.com/open-mmlab/mmsegmentation/blob/v1.2.2/mmseg/datasets/cityscapes.py
CITYSCAPES_LABELS = (
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
)

# Colors for trajectory plotting
TRAJECTORY_COLORS = (
    "blue",
    "black",
    "magenta",
    "cyan",
    "yellow",
    "purple",
    "pink",
    "brown",
)

# Random seeds for evaluation
SEEDS = (
    123456789,
    987654321,
    42,
    13,
    7,
    0,
)