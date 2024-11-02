# General Large-Scale Dense 3D Point Cloud Reconstruction from a Monocular Camera
This is a library and collection of scripts that allow combining pose predictions from a monocular SLAM system with depth predictions from a monocular depth neural network to create dense point clouds. The resulting point clouds can then be used to initialize 3D Gaussians for 3D Gaussian Splatting and photorealistic rendering.

The whole pipeline works in an offline fashion and each step can be run individually by running the numbered scripts. The script `run.py` runs the whole pipeline based on hydra configuration specified in `

## TODO
- [ ] Implement configs and `run.py` to connect the whole pipeline
- [ ] Implement `4_estimate_poses.py` and SLAM wrappers for ORB-SLAM3 to incorporate SLAM pose estimation
- [ ] Add demo data and example
- [ ] Finish up the README.md

## Setup
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

## Getting Started
This is a quick tutorial to guide you how to use the provided scripts for your own data.

## Dataset
If you are starting with a video input, create a dataset directory and put your video file in it- You also need to specify a dataset root directory where all of your datasets will be stored. The file structure in the beginning should look like this:

```plaintext
data_root_dir
├── dataset_name_1
│   └── data
│       └── your_video_1.mp4
├── dataset_name_2
│   └── data
│       └── your_video_2.mp4
...
```

Then you can either setup your config in `./configs/config.yaml` and let the whole pipeline run or you can run each step manually by executing the provided scripts in order. For the second option you should start with extracting images from your video. As an example, assume we are starting with a raw video and do not know the camera intrinsics. We would start with:

```bash
python 1_extract_frames_from_video.py --video_path data_root_dir/dataset_name_1/your_video_1.mp4 --target_size 576 1024
```

This will extract all frames from the video, resize them to `576x1024` pixels, and start saving the images on disk under `/data_root_dir/dataset_name_1/data/rgb`.

If you do not know your camera intrinsics, it is possible to use COLMAP with the images we extracted to get a rough estimate:

```bash
python 2_run_colmap.py -s /data_root_dir/dataset_name_1/poses/colmap --camera SIMPLE_PINHOLE
```

This would give you a good estimate of the intrinsics and also estimate poses. To see the intrinsics you should look at `/data_root_dir/dataset_name_1/poses/colmap/sparse/0/cameras.txt`. To see utility functions related to using COLMAP outputs please see `ColmapDataset` in `./modules/io/datasets.py`.

By default this script runs COLMAP with the exhaustive matcher. If you are using videos from a video (with sequential images) you can also use the `--use_sequential_matcher` option of the script. You would also need to download the vocabulary tree from the official COLMAP website.

## Pose Estimation
Once we have the intrinsics, we can either use the poses from COLMAP or run any SLAM system. We found that COLMAP usually performs well but SLAM systems returns much more accurate poses when the camera does a loop. This is expected as COLMAP isn't optimized for sequentially captured images (see https://github.com/colmap/colmap/issues/411 and https://github.com/colmap/colmap/issues/1521).

...

## Depth and Scale

## Pointcloud Creation

## Gaussian Splatting

## Evaluation



## Acknowledgements
- evo
- metric3d
- MMseg
- unidepth
- ...
*TODO: Add links and complete the list!*
