# 🎓 Master Thesis Project - Generalized Large-Scale Dense 3D Point Cloud Reconstruction from a Monocular Camera
This is the codebase for my master's thesis project where I investigated how state-of-the-art pose estimation systems (SfM and SLAM, COLMAP and ORB-SLAM3 specifically), generalizable metric depth prediction models (Metric3D), and point-based radiance field rendering methods (3D Gaussian Splatting) can be combined

## TODO
- [ ] Implement configs and `run.py` to connect the whole pipeline
- [ ] Implement `4_estimate_poses.py` and SLAM wrappers for ORB-SLAM3 to incorporate SLAM pose estimation
- [ ] Add demo data and example
- [ ] Finish up the README.md

## Installation
To clone:
```bash
git clone --recurse-submodules git@github.com:altaykacan/DEN-Splatting.git
```

If you forgot to include `--recurse-submodules` you can use the following after cloning:
```bash
git submodule update --init --recursive
```

Environment setup, currently only tested with PyTorch 2.0.1, CUDA 11.8, and Python 3.8:
```bash
conda env create -n thesis python=3.8
conda activate thesis

### For Metric3Dv2 ###
cd modules/depth/Metric3Dv2
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r submodules/Metric3D/requirements_v2.txt
pip install -U openmim
mim install mmengine
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html # this is very sensitive to the pytorch and CUDA versions, check official documentation
pip install mmsegmentation

### For 3DGS ###
# Make sure to have CUDA 11.8 or higher on your system
# If you get weird errors on different machines after building the pip packages, uninstall them, delete the build folders, and install them again using pip
pip install submodules/gaussian-splatting/submodules/diff-gaussian-rasterization-entropy
pip install submodules/gaussian-splatting/submodules/simple-knn

####################

# Generally needed
pip install open3d
pip install plyfile
pip install tensorboard
pip install hydra-core --upgrade # for configs
pip install moviepy # for video creation
```

## Acknowledgements:
- evo
- metric3d
- MMseg
- unidepth
- ...
*TODO: Add links!*
