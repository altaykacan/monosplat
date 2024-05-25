# 🎓 Master Thesis Project - Generalized Large-Scale Dense 3D Point Cloud Reconstruction from a Monocular Camera

The current work on scale alignment is under the Python package `scale_alignment`. To see examples and visualizations please see `debug.ipynb`. The script `tests.py` include example usages of the developed functionality.

Image data (`data/images`) and debug results (`debug_results`) are not on GitHub, please contact me if you'd like a zip file.

To clone:
```bash
git clone --recurse-submodules git@github.com:altaykacan/Toolbox-Draft.git
```

If you forgot to include `--recurse-submodules` you can use the following after cloning:
```bash
git submodule update --init --recursive
```


Environment setup, tested with PyTorch 2.0.1, CUDA 11.8, and Python 3.8:
```bash
conda env create -n thesis python=3.8
conda activate thesis

### For Metric3Dv2 ###
cd modules/depth/Metric3Dv2
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements_v2.txt
pip install -U openmim
mim install mmengine
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html # this is very sensitive to the pytorch and CUDA versions, check official documentation
pip install mmsegmentation

### For UniDepth ###



### For DSINE ###


####################

# Generally needed
pip install open3d
pip install plyfile
pip install hydra-core --upgrade # for configs
pip install moviepy # for video creation

```
