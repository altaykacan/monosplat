# General Large-Scale Dense 3D Point Cloud Reconstruction from a Monocular Camera
> ![Teaser Video](./assets/teaser_video.gif)
> ![Results](./assets/results.png)

This is a library and collection of scripts that allow combining pose predictions from a monocular SLAM system with depth predictions from a monocular depth neural network to create dense point clouds. The resulting point clouds can then be used to initialize 3D Gaussians for 3D Gaussian Splatting and photorealistic rendering.

The whole pipeline works in an offline fashion and each step can be run individually by running the enumerated scripts.



## TODO
- [x] Finish up the README.md
- [ ] Implement main hydra configs and `run.py` to connect the whole pipeline and run the scripts
- [ ] Implement `4_estimate_poses.py` and SLAM wrappers for ORB-SLAM3 to incorporate SLAM pose estimation
- [ ] Add demo data and example

## Setup
To clone:
```bash
git clone --recurse-submodules git@github.com:altaykacan/monosplat.git
```

If you forgot to include `--recurse-submodules` you can use the following after cloning:
```bash
git submodule update --init --recursive
```

Environment setup, currently only tested with PyTorch 2.0.1, CUDA 11.8, and Python 3.8:
```bash
conda env create -n monosplat python=3.8
conda activate monosplat

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
pip install moviepy # for animations
```

## Getting Started
This is a quick tutorial to guide you how to use the provided scripts for your own data.

This is the overall pipeline that is implemented to create 3D renderings from monocular video input (dotted lines indicate optional input):
> ![Pipeline](./assets/pipeline.png)

### Dataset
If you are starting with a video input, create a dataset directory and put your video file in it. You also need to specify a dataset root directory where all of your datasets will be stored. The file structure in the beginning should look like this:

```plaintext
data_root_dir
├── dataset_name_1
│   └── data
│       └── your_video_1.mp4
├── dataset_name_2
│   └── data
│       └── your_video_2.mp4
[...]
```

Then you can either setup your config in `./configs/config.yaml` and let the whole pipeline run (which is not implemented yet, see TODOs) or you can run each step manually by executing the provided scripts in order. For the second option you should start with extracting images from your video. As an example, assume we are starting with a raw video and do not know the camera intrinsics. We would start with:

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

You can also save segmentation masks for movable objects to ignore pixels belonging to them when reconstructing the point cloud, saving precomputed depths, or training a 3D Gaussian Splatting model. Please see `./thesis/tools/create_masks.py` for a script on how to do that.

### Pose Estimation
Once we have the intrinsics, we can either use the poses from COLMAP or run any SLAM system. We found that COLMAP usually performs well but SLAM systems returns much more accurate poses when the camera does a loop. This is expected as COLMAP isn't optimized for sequentially captured images (see [this](https://github.com/colmap/colmap/issues/411) and [this](https://github.com/colmap/colmap/issues/1521)).

There are two options to run SLAM to estimate poses:
1. Use purely visual SLAM,
2. Use RGB-D SLAM with depth predictions from a depth predictor network.

In the first case you just need to use a SLAM system capable of working with TUM RGB-D data format. The required `rgb.txt` is created under `/data_root_dir/dataset_name_1/data/` after running `1_extract_frames_from_video.py`. Please see the TUM RGB-D dataset documentation for more information on the expected format [link](https://cvg.cit.tum.de/data/datasets/rgbd-dataset).

For the second case, you would need to incorporate depth predictions of a deep neural net to your SLAM system. To keep things simple and flexible, we chose to do this asynchronously. First run `3_precompute_depth_and_normals.py` with a command like:

```bash
python 3_precompute_depth_and_normals.py -r /data_root_dir/dataset_name_1 --model metric3d_vit --intrinsics fx fy cx cy --max_depth_png 50 --scale_factor_depth_png 1000
```

By default, this script will save depth and normal predictions both as numpy arrays and png images. To use RGB-D SLAM with the depths, it is the simplest to keep the images as pngs as the TUM RGB-D dataset format expects it. Additionally, it allows you to easily view the model predictions by sacrificing some disk space. Since we are using the Metric3Dv2 family of models by default, we can both predict approximately metric depths and surface normals. If you do not have a model to predict normals or want to save disk space you can provide the `--skip_depth_png` and `--skip_normals` flags.

Since the TUM RGB-D dataset expects depths to be monochrome 16-bit png images, to capture decimal numbers in the predicted depths (which are floats) we need to scale the depth values. This is specified by the `--scale_factor_depth_png 1000` argument. **Make sure you are using the same depth png scale factor and intrinsics for your SLAM system**.

This script also uses a segmentation model to predict masks for movable objects and set their depths to a large constant value. This allows you to effectively mask out movable objects by simply setting a maximum depth in your SLAM system. The extracted feature points/estimated depths by the SLAM system will get filtered out when combined with the depth pseudo-measurements. You can provide the `--skip_mask_depth_png` flag to disable this behaviour.

After running the depth and normal computation script for `dataset_name_1` we would end up with:

```plaintext
data_root_dir
├── dataset_name_1
│   ├── data
│   │   ├── rgb                # has all your images
│   │   ├── depths
│   │   │   ├── arrays         # as numpy arrays
│   │   │   └── images         # as 16-bit monochrome png files
│   │   ├── normals
│   │   │   ├── arrays
│   │   │   └── images
│   │   ├── rgb.txt
│   │   ├── depth.txt
│   │   ├── associations.txt   # needed for RGB-D SLAM in TUM RGB-D format
│   │   └── your_video_1.mp4
│   └── poses
│       └── colmap             # where all the colmap results are
├── dataset_name_2
│   └── data
│       └── your_video_2.mp4
[...]
```

Once you run your SLAM system and get poses in the TUM RGB-D format, you can simply create a new directory under poses and put the text files there:

```plaintext
[...]
│   └── poses
│       ├── slam             # make sure to name this directory 'slam'
│       |    ├── poses_1.txt
│       |    ├── poses_2.txt
│       |    └── [...]
│       └── colmap
[...]
```

### Scale Alignment
To create pointclouds by backprojecting the predicted depths and combining them with the poses, we need to *align the scales of the depths and poses*. With monocular SLAM, the poses are only accurate up to an unknown scale factor (due to scale ambiguity) and even though the depth model we are using is trained to predict *metric* depth, it is usually not perfectly accurate. The model might have learned an internal scale value that is close to metric scale but might be off.

The script `5_align_scale.py` is exactly for this purpose and uses the different scale alignment methods as implemented in `./modules/scale_alignment`. You can use the script as following:

```bash
python 5_align_scale.py -r /data_root_dir/dataset_name_1 -p /data_root_dir/dataset_name_1/sparse/0/images.txt --intrinsics fx fy cx cy -a dense --dataset colmap --seg_model_type precomputed --exp_name your_experiment_name
```

There are many options on how to do scale alignment which are all explained in the arguments of the script, for in depth explanations and a list of arguments please run:

```bash
python 5_align_scale.py --help
```

The computed scale factors will be saved in the parent directory of the pose files you specify with the `-p` option, i.e. under `/data_root_dir/dataset_name_1/sparse/0/scales_and_shifts.txt`. With the `dense` alignment option, these values will all be the same and no shift factor will be computed. If you specify `-a sparse` instead, the script optimizes per-frame depth scale factors. If you want to compute per-frame shifts additionally, also provide the `--compute_shift_and_scale` flag.

The scale factor is the scale you need to *multiply the depths* with.

### Point cloud Creation
With scale-aligned poses and depth predictions, we can simply backproject each image and accumulate dense point clouds.

The command:
```bash
python 6_create_pointcloud.py -r /data_root_dir/dataset_name_1 --intrinsics fx fy cx cya --reconstructor simple --backprojector simple --dataset colmap --max_d 50 --depth_model precomputed --seg_model segformer --pose_scale 1.0 --use_every_nth 3 --batch_size 2 --dropout 0.90 --downsample_pointcloud_voxel_size 0.05 --init_cloud_path /data_root_dir/dataset_name_1/colmap/sparse/0/points3D.ply --add_skydome --recon_name your_reconstruction_name --clean_pointcloud --depth_scale your_scale_value
```

demonstrates the main functionality of `6_create_pointcloud.py`. Depending on what you specify for the reconstruction name, this script will save dense point cloud reconstructions (with associated poses, intrinsics, and logs to run Gaussian Splatting easily) under a new directory `reconstructions` in your dataset directory. Your directory structure should look something like this:

```plaintext
data_root_dir
├── dataset_name_1
│   ├── data
│   │   ├── rgb
│   │   ├── depths
│   │   │   ├── arrays
│   │   │   └── images
│   │   ├── normals
│   │   │   ├── arrays
│   │   │   └── images
│   │   ├── rgb.txt
│   │   ├── depth.txt
│   │   ├── associations.txt
│   │   └── your_video_1.mp4
│   ├── poses
│   │   ├── slam
│   │   └── colmap
│   └── reconstructions
│       └── reconstruction_name   # has .ply and everything else for 3dgs
├── dataset_name_2
│   └── data
│       └── your_video_2.mp4
[...]
```

The reconstruction (the `.ply` file) will get used to initialize the Gaussians if you want to use 3D Gaussian Splatting. To additionally include the initial point cloud (i.e. the sparse point cloud from COLMAP or any other point cloud you have) just specify the path with `--init_cloud_path`. It is important to make sure that the initial cloud you are adding, the poses, and the approximately metric depth predictions all have the same scale.

There are three ways to determine which scaling factors are used in the reconstruction. The `--depth_scale` is the value you get by running scale alignment in the previous step. The depth predictions from the neural network get multiplied by this value. You can alternatively provide `--scales_and_shifts_path` if you want to experiment with using per-frame scale estimates or scale estimates with shifts. With this option, every depth prediction would get multipled by its corresponding scale factor and a shift gets applied. Finally, you can set `--pose_scale` to multiply the poses by a constant factor. This value would be the reciprocal of the numerical value you use for the depth scale.

Please refer to the help options of `6_create_pointcloud.py` for more details and a complete list of arguments.

### Gaussian Splatting
The final step is to use the point clouds you created to initialize a set of Gaussians. The script `7_train_gaussians.py` itself has only a few arguments and forwards any other arguments it receives to the training script under `./submodules/gaussian-splatting/train.py`.

Here's an example command you can run for your own data:
```bash
python 7_train_gaussians.py -r /data_root_dir/dataset_name_1 --recon_name reconstruction_name -o splat_name --use_mask --mask_path /data_root_dir/dataset_name_1/data/masks --use_gt_depth --use_gt_normal --use_every_nth_as_val 10
```

The `--use_gt_normal` and `--use_gt_depth` flags specify whether normal and depth regularization should be used. The current version includes custom depth and normal rendering code which was the only option during the time this project was done. Now, the original 3DGS repository supports depth rendering which is recommended.

If you provide the `--colmap_dir` argument instead of a reconstruction name, the script will ignore the custom dense point cloud reconstructions and train 3DGS expecting the default colmap directory structure, i.e. this path should be `/data_root_dir/dataset_name_1/poses/colmap`.

You can specify all other arguments to the 3DGS `train.py` script through `7_train_gaussians.py` as in the original implementation. Please see [here](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file#running) for more details.

The trained splats will be saved under `/dataset_root_dir/dataset_name_1/splats/splat_name` in the default 3DGS format. Thus, you can use any viewer that supports viewing 3D Gaussians.

Here is an example fly-through for a dataset we captured ourselves:
> ![Example fly-through](./assets/ostspange.gif)

You can find (rough and hacky) scripts on how to create such videos under `./thesis/tools/create_video_*.py`. 

## Acknowledgements
Thanks to all of these great repositiories and research for making this project possible:

- TUM RGB-D dataset (https://cvg.cit.tum.de/data/datasets/rgbd-dataset)
- COLMAP (https://github.com/colmap/colmap)
- ORB-SLAM3 (https://github.com/UZ-SLAMLab/ORB_SLAM3)
- Metric3D (https://github.com/YvanYin/Metric3D)
- 3DGS (https://github.com/graphdeco-inria/gaussian-splatting)
- GaussianPro (https://github.com/kcheng1021/GaussianPro/tree/version1.0)
- evo (https://github.com/MichaelGrupp/evo)
- MMseg (https://github.com/open-mmlab/mmsegmentation)
