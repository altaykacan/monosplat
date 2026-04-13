# CLAUDE.md — monosplat

This file provides guidance for AI assistants working in this repository.

## Project Overview

**monosplat** is a research-grade Python library and collection of pipeline scripts that produce dense 3D point cloud reconstructions from monocular video, then use those point clouds to initialize 3D Gaussian Splatting (3DGS) models for photorealistic rendering.

Core idea: combine monocular pose estimates (COLMAP / SLAM) with metric depth predictions (Metric3Dv2) to overcome scale ambiguity and create metric-scale dense reconstructions.

## Repository Structure

```
monosplat/
├── 1_extract_frames_from_video.py    # Step 1 – extract & resize frames
├── 2_run_colmap.py                   # Step 2 – COLMAP poses & calibration
├── 3_precompute_depth_and_normals.py # Step 3 – depth & normal prediction
├── 4_estimate_poses.py               # Step 4 – TODO: SLAM wrapper
├── 5_align_scale.py                  # Step 5 – scale alignment
├── 6_create_pointcloud.py            # Step 6 – dense point cloud creation
├── 7_train_gaussians.py              # Step 7 – 3DGS training wrapper
├── run.py                            # TODO: Hydra-based pipeline entry point
├── configs/
│   ├── data.py                       # Global constants (DATA_DIR, etc.)
│   ├── config.yaml                   # TODO: central Hydra config
│   └── dataset/                      # Dataset-specific Hydra configs
├── modules/
│   ├── core/                         # Abstract interfaces, utilities, data structures
│   │   ├── interfaces.py             # BaseModel, BaseMap, BaseReconstructor, etc.
│   │   ├── models.py                 # RAFT optical flow wrapper
│   │   ├── reconstructors.py         # SimpleReconstructor, MovingObjectRemovalReconstructor
│   │   ├── backprojection.py         # Backprojector, DepthBasedDropoutBackprojector
│   │   ├── maps.py                   # PointCloud (Open3D-based)
│   │   ├── utils.py                  # compute_target_intrinsics(), resize_image_torch()
│   │   └── visualization.py
│   ├── depth/
│   │   └── models.py                 # Metric3Dv2, KITTI360DepthModel, PrecomputedDepthModel
│   ├── segmentation/
│   │   ├── models.py                 # SegFormer, MaskRCNN, PrecomputedSegModel
│   │   ├── moving_objects.py         # Moving object detection & filtering
│   │   └── utils.py
│   ├── scale_alignment/
│   │   ├── dense.py                  # Per-sequence dense scale alignment
│   │   └── sparse.py                 # Per-frame sparse scale alignment
│   ├── pose/
│   │   ├── alignment.py              # Pose alignment utilities
│   │   └── slam.py                   # TODO: SLAM wrapper
│   ├── io/
│   │   ├── datasets.py               # All dataset classes
│   │   └── utils.py                  # File I/O helpers
│   └── eval/
│       ├── metrics.py                # Depth & trajectory metrics
│       ├── tum_rgbd_tools/           # ATE/RPE evaluation
│       └── kitti360.py
├── submodules/
│   ├── gaussian-splatting/           # Modified 3DGS fork
│   ├── Metric3D/                     # Metric3Dv2 depth model
│   └── UniDepth/                     # Alternative depth model
├── evaluation/                       # Evaluation scripts & results
├── thesis/
│   ├── tests/                        # Ad-hoc debug scripts (not unit tests)
│   └── tools/                        # Visualization & utility helpers
├── demo/                             # Demo data (KITTI, KITTI-360, TUM RGB-D)
├── checkpoints/                      # Pretrained model weights
└── assets/                           # Images, GIFs for README
```

## Pipeline Overview

Each numbered script is a self-contained pipeline step:

| Step | Script | Purpose |
|------|--------|---------|
| 1 | `1_extract_frames_from_video.py` | Extract frames, resize, generate `rgb.txt` |
| 2 | `2_run_colmap.py` | Feature extraction/matching, pose & intrinsics estimation |
| 3 | `3_precompute_depth_and_normals.py` | Metric3Dv2 depth/normal prediction + moving-object masking |
| 4 | `4_estimate_poses.py` | *TODO:* ORB-SLAM3 wrapper |
| 5 | `5_align_scale.py` | Align depth scale with poses (dense or sparse) |
| 6 | `6_create_pointcloud.py` | Backproject depths → dense `.ply` point cloud |
| 7 | `7_train_gaussians.py` | Wrapper around 3DGS `train.py` |

Steps can be run individually. The unified `run.py` (Hydra-based) is not yet implemented.

## Expected Dataset Directory Layout

```
data_root_dir/
└── dataset_name/
    ├── data/
    │   ├── rgb/                    # PNG frames (zero-padded filenames)
    │   ├── depths/
    │   │   ├── arrays/             # .npy depth arrays
    │   │   └── images/             # 16-bit monochrome PNGs
    │   ├── normals/
    │   │   ├── arrays/
    │   │   └── images/
    │   ├── rgb.txt                 # TUM RGB-D format association file
    │   ├── depth.txt
    │   └── associations.txt
    ├── poses/
    │   ├── colmap/                 # COLMAP sparse reconstruction
    │   └── slam/                   # SLAM output (TUM RGB-D pose format)
    ├── reconstructions/
    │   └── <recon_name>/           # .ply + poses + intrinsics for 3DGS
    └── splats/
        └── <splat_name>/           # 3DGS output
```

## Environment Setup

Tested with Python 3.8, PyTorch 2.0.1, CUDA 11.8:

```bash
# Clone with submodules
git clone --recurse-submodules git@github.com:altaykacan/monosplat.git
# Or after cloning:
git submodule update --init --recursive

# Create conda environment
conda create -n monosplat python=3.8
conda activate monosplat

# PyTorch (must match CUDA version)
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia

# Metric3Dv2 dependencies
pip install -r submodules/Metric3D/requirements_v2.txt
pip install -U openmim
mim install mmengine
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
pip install mmsegmentation

# 3DGS CUDA extensions
pip install submodules/gaussian-splatting/submodules/diff-gaussian-rasterization-entropy
pip install submodules/gaussian-splatting/submodules/simple-knn

# Other dependencies
pip install open3d plyfile tensorboard hydra-core --upgrade moviepy scikit-learn
```

> If CUDA extension builds fail on a different machine, uninstall the packages, delete the build folders, and reinstall.

## Configuration

### `configs/data.py`

User-editable constants used throughout the codebase:

```python
DATA_DIR = "..."                    # Root directory for all datasets
KITTI360_DIR = "/path/to/KITTI-360"
PADDED_IMG_NAME_LENGTH = 5          # Filename zero-padding (5 for custom/colmap, 6 for KITTI, 10 for KITTI-360)
OUTPUT_DIR = "/path/to/outputs"
SUPPORTED_DATASETS = ["custom", "kitti", "kitti360", "colmap", "tum_rgbd"]
EVAL_DECIMAL_POINTS = 5
```

**Always update `DATA_DIR` and `KITTI360_DIR` before running any scripts.**

### Hydra Configs (partially implemented)

- `configs/config.yaml` — central pipeline config (TODO)
- `configs/dataset/*.yaml` — dataset-specific configs
- Intended to be loaded by the future `run.py` entry point

## Code Conventions

### Naming

| Element | Convention | Example |
|---------|-----------|---------|
| Functions | `snake_case` | `compute_target_intrinsics()` |
| Classes | `PascalCase` | `Metric3Dv2`, `PointCloud` |
| Constants | `UPPER_CASE` | `PADDED_IMG_NAME_LENGTH` |
| Private methods | `_method_name` | `_preprocess()`, `_predict()` |
| Module loggers | `log = logging.getLogger(__name__)` | — |

### Abstract Base Class Pattern

All major components inherit from abstract interfaces in `modules/core/interfaces.py`:

- **`BaseModel`** — any neural network model. Must implement `load()`, `unload()`, `_preprocess()`, `_predict()`, `_check_input()`, `_check_output()`. Public entry point is `predict(input_dict) -> output_dict`.
- **`BaseDataset`** — dataset wrappers. Must implement `load_image_paths_and_poses()`, `get_by_frame_ids()`, `__len__`, `__getitem__`.
- **`BaseReconstructor`** — point cloud reconstruction pipelines. Must implement `run()`, `step()`, `parse_config()`.
- **`BaseBackprojector`** — pixel-to-3D projection. Must implement `backproject()` and `compute_backprojection_masks()`.
- **`BaseMap`** — map data structures. Must implement `save()`, `transform()`, `increment()`, `postprocess()`.

When adding a new model, dataset, or reconstructor, subclass the appropriate base class and implement all abstract methods.

### Input/Output Dictionaries

Models communicate via `Dict` objects rather than positional arguments. This maintains flexibility across different model types. Check `_check_input()` / `_check_output()` in each class for expected keys.

### Tensor Conventions

| Data | Shape |
|------|-------|
| Batched images | `[N, C, H, W]` |
| Single image | `[C, H, W]` |
| Point cloud xyz | `[3, num_points]` |
| Point cloud with attributes | `[C, num_points]` |
| Pose matrix | `[4, 4]` |
| Batched masks | `[N, 1, H, W]` (bool, `True` = include) |

### Path Handling

- Always use `pathlib.Path`, not raw strings.
- Create directories with `path.mkdir(parents=True, exist_ok=True)`.
- Convert string inputs to `Path` at function boundaries: `path if isinstance(path, Path) else Path(path)`.

### Device & Memory Management

- Detect CUDA: `device = "cuda" if torch.cuda.is_available() else "cpu"`.
- Models use explicit `load()` / `unload()` for GPU memory management — do not load all models simultaneously unless memory allows.
- Check bfloat16 support before using it (ViT-based models).

### Logging

Every pipeline script uses a two-handler logger (file + console):

```python
log_path = output_dir / "log_<step>.txt"
with open(log_path, 'w'): pass  # clear existing log
logging.basicConfig(filename=log_path, level=logging.INFO, format='%(levelname)s - %(message)s')
logging.getLogger().addHandler(logging.StreamHandler())
logging.info(f"Arguments: \n{json.dumps(vars(args), indent=4)}")
```

Module-level loggers use `log = logging.getLogger(__name__)`.

### Import Order

```python
# 1. Standard library
import json, argparse, logging
from pathlib import Path
from typing import Union, Dict, Tuple, List

# 2. Third-party
import torch
import numpy as np
import cv2
from PIL import Image

# 3. Local modules
from configs.data import PADDED_IMG_NAME_LENGTH
from modules.core.utils import compute_target_intrinsics
from modules.io.utils import find_latest_number_in_dir
```

### Reconstruction Naming

Reconstructions are auto-numbered: `find_latest_number_in_dir(recon_root_dir)` returns the highest existing prefix, and new directories get prefix `latest + 1`. Format: `<N>_<recon_name>`.

### Scale Alignment Notes

- **Dense alignment** (`-a dense`): computes a single global scale factor for the whole sequence. Result saved as `scales_and_shifts.txt` next to the pose file.
- **Sparse alignment** (`-a sparse`): per-frame scale factors (optionally with shift via `--compute_shift_and_scale`).
- Scale factor is the value to **multiply predicted depths by**.
- `--pose_scale` is the reciprocal of `--depth_scale` when scaling poses instead of depths.

## Supported Datasets

| Dataset | Class | Notes |
|---------|-------|-------|
| `custom` | `CustomDataset` | Generic monocular video, TUM RGB-D pose format |
| `colmap` | `ColmapDataset` | COLMAP `images.txt` format |
| `colmap` (multi) | `CombinedColmapDataset` | Multiple COLMAP sequences |
| `kitti360` | `KITTI360Dataset` | KITTI-360 benchmark |
| `kitti` | `KITTIDataset` | KITTI Odometry |
| `tum_rgbd` | `TUMRGBDDataset` | TUM RGB-D benchmark |

The `--dataset` argument in most scripts selects the dataset class.

## Testing

There is no formal test suite. `thesis/tests/` contains ad-hoc debug scripts used during development. When adding new functionality, create a test script there following `test_template.py`.

To run a test script:
```bash
python thesis/tests/test_<module>.py
```

There is no `pytest`, `unittest`, CI/CD, or linting configuration.

## Key Submodules

| Submodule | Path | Purpose |
|-----------|------|---------|
| `gaussian-splatting` | `submodules/gaussian-splatting/` | Modified 3DGS with depth/normal rendering |
| `Metric3D` | `submodules/Metric3D/` | Metric3Dv2 depth & normal prediction |
| `UniDepth` | `submodules/UniDepth/` | Alternative depth model (less used) |

Always use `git submodule update --init --recursive` when switching branches that touch submodule commits.

## Known TODOs / Incomplete Areas

- `run.py` — unified Hydra pipeline orchestration (not implemented)
- `4_estimate_poses.py` — ORB-SLAM3 integration (not implemented)
- `modules/pose/slam.py` — SLAM wrapper (not implemented)
- `configs/config.yaml` — central Hydra config (template only)

Do not assume these are implemented. Scripts 1–3 and 5–7 are fully functional.

## Common Pitfalls

- **`mmcv` version** is tightly coupled to PyTorch and CUDA versions — always check the official mmcv release matrix.
- **3DGS CUDA extensions** must be rebuilt if the machine or CUDA version changes; delete the build directories and reinstall.
- **Submodule state** — after cloning or pulling, run `git submodule update --init --recursive` to avoid import errors.
- **`configs/data.py`** contains hardcoded paths (`/usr/stud/kaa/...`) that must be updated before use.
- **Depth PNG scale factor** — must match between `3_precompute_depth_and_normals.py` and any downstream SLAM system; default is `1000` (i.e., depth in mm stored as uint16).
- **Moving object masking** — by default enabled in `3_precompute_depth_and_normals.py`; provides `--skip_mask_depth_png` to disable.
