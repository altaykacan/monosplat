"""
Custom classes for working with KITTI360 data. Most of the code is adapted
from kitti360scripts: https://github.com/autonomousvision/kitti360Scripts
"""
import os
from typing import List, Union

import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from kitti360scripts.devkits.commons.loadCalibration import loadCalibrationCameraToPose, loadCalibrationRigid
from kitti360scripts.helpers.project import CameraPerspective, CameraFisheye


class Kitti360Viewer3DRaw(object):
    """
    Main class to work with KITTI-360 laser scans.
    Code taken from the original KITTI-360 repository and slightly adapted
    to current framework.
    """

    # Constructor
    def __init__(self, seq=0, mode='velodyne', kitti360path=None):

        if kitti360path is None and 'KITTI360_DATASET' in os.environ:
            kitti360path = os.environ['KITTI360_DATASET']
        elif kitti360path is None:
            kitti360path = os.path.join(os.path.dirname(
                                os.path.realpath(__file__)), '..', '..')

        if mode=='velodyne':
            self.sensor_dir='velodyne_points'
        elif mode=='sick':
            self.sensor_dir='sick_points'
        else:
            raise RuntimeError('Unknown sensor type!')

        sequence = '2013_05_28_drive_%04d_sync' % seq
        self.raw3DPcdPath  = os.path.join(kitti360path, 'data_3d_raw', sequence, self.sensor_dir, 'data')

    def loadVelodyneData(self, frame=0):
        pcdFile = os.path.join(self.raw3DPcdPath, '%010d.bin' % frame)
        if not os.path.isfile(pcdFile):
            raise RuntimeError('%s does not exist!' % pcdFile)
        pcd = np.fromfile(pcdFile, dtype=np.float32)
        pcd = np.reshape(pcd,[-1,4])
        return pcd

    def loadSickData(self, frame=0):
        pcdFile = os.path.join(self.raw3DPcdPath, '%010d.bin' % frame)
        if not os.path.isfile(pcdFile):
            raise RuntimeError('%s does not exist!' % pcdFile)
        pcd = np.fromfile(pcdFile, dtype=np.float32)
        pcd = np.reshape(pcd,[-1,2])
        pcd = np.concatenate([np.zeros_like(pcd[:,0:1]), -pcd[:,0:1], pcd[:,1:2]], axis=1)
        return pcd

def get_kitti360_frames_with_poses(cam_id: int = 0, seq = 0, kitti360path: str = 0, start: int = 0, end: int =-1) -> List[int]:
    """
    Function to parse the `cam0_to_world.txt` files of the associated sequences
    from the KITTI360 dataset. Returns a list of frame_ids as integers.
    """
    if kitti360path is None and 'KITTI360_DATASET' in os.environ:
        kitti360path = os.environ['KITTI360_DATASET']
    elif kitti360path is None:
        kitti360path = os.path.join(os.path.dirname(
                                os.path.realpath(__file__)), '..', '..')

    sequence = '2013_05_28_drive_%04d_sync'%seq

    # poses.txt has the imu -> world transform, cam0_to_world.txt has cam0 -> world transform
    poses_path = os.path.join(kitti360path, "data_poses", sequence, "cam0_to_world.txt")

    if not os.path.isfile(poses_path):
        raise FileNotFoundError(f"cam0_to_world.txt can't be found in {poses_path}, please check your dataset!")

    frame_ids = []
    with open(poses_path, "r") as file:
        for line in file:
            cols = line.split(" ")
            id = int(cols[0]) # is an integer

            frame_ids.append(id)

    if end == -1:
        frame_ids = frame_ids[start:]
    else:
        frame_ids = frame_ids[start:end]

    return frame_ids


def cam2image(camera, points):
    """
    Perspective projection function from the original KITTI-360 authors,
    numpy `np.int` aliases are replaced with `int` to match newer versions of numpy.
    """
    ndim = points.ndim
    if ndim == 2:
        points = np.expand_dims(points, 0)
    points_proj = np.matmul(camera.K[:3,:3].reshape([1,3,3]), points)
    depth = points_proj[:,2,:]
    depth[depth==0] = -1e-6
    u = np.round(points_proj[:,0,:]/np.abs(depth)).astype(int)
    v = np.round(points_proj[:,1,:]/np.abs(depth)).astype(int)

    if ndim==2:
        u = u[0]; v=v[0]; depth=depth[0]
    return u, v, depth

def projectVeloToImage(cam_id, seq, kitti360path=None, max_d=200, image_name_padding=10):
    """
    Function to create depth maps from KITTI-360 ground truth values. Taken
    and slightly adapted from the original authors of kitti360scripts:
    https://github.com/autonomousvision/kitti360Scripts/blob/7d44b19446b92801cff403ac0eff5985986ff481/kitti360scripts/viewer/kitti360Viewer3DRaw.py#L66
    """
    if kitti360path is None and 'KITTI360_DATASET' in os.environ:
        kitti360path = os.environ['KITTI360_DATASET']
    elif kitti360path is None:
        kitti360path = os.path.join(os.path.dirname(
                                os.path.realpath(__file__)), '..', '..')

    sequence = '2013_05_28_drive_%04d_sync'%seq

    depth_path = os.path.join(kitti360path, "data_3d_raw", sequence, f"image_0{cam_id}", "depths")
    os.makedirs(depth_path, exist_ok=True)

    # perspective camera
    if cam_id in [0,1]:
        camera = CameraPerspective(kitti360path, sequence, cam_id)
    # fisheye camera
    elif cam_id in [2,3]:
        camera = CameraFisheye(kitti360path, sequence, cam_id)
    else:
        raise RuntimeError('Unknown camera ID!')

    # object for parsing 3d raw data
    velo = Kitti360Viewer3DRaw(mode='velodyne', seq=seq, kitti360path=kitti360path)

    # cam_0 to velo
    fileCameraToVelo = os.path.join(kitti360path, 'calibration', 'calib_cam_to_velo.txt')
    TrCam0ToVelo = loadCalibrationRigid(fileCameraToVelo)

    # all cameras to system center
    fileCameraToPose = os.path.join(kitti360path, 'calibration', 'calib_cam_to_pose.txt')
    TrCamToPose = loadCalibrationCameraToPose(fileCameraToPose)

    # velodyne to all cameras
    TrVeloToCam = {}
    for k, v in TrCamToPose.items():
        # Tr(cam_k -> velo) = Tr(cam_k -> cam_0) @ Tr(cam_0 -> velo)
        TrCamkToCam0 = np.linalg.inv(TrCamToPose['image_00']) @ TrCamToPose[k]
        TrCamToVelo = TrCam0ToVelo @ TrCamkToCam0
        # Tr(velo -> cam_k)
        TrVeloToCam[k] = np.linalg.inv(TrCamToVelo)

    # take the rectification into account for perspective cameras
    if cam_id==0 or cam_id == 1:
        TrVeloToRect = np.matmul(camera.R_rect, TrVeloToCam['image_%02d' % cam_id])
    else:
        TrVeloToRect = TrVeloToCam['image_%02d' % cam_id]

    # color map for visualizing depth map
    cm = plt.get_cmap('jet')

    # Get all frames that we have poses for
    frame_ids = get_kitti360_frames_with_poses(cam_id, seq, kitti360path)

    # for each frame, load the raw 3D scan and project to image plane
    print(f"Saving depth images from velodyne scans of sequence {seq} and camera {cam_id}...")
    for frame in tqdm(frame_ids):
        points = velo.loadVelodyneData(frame)
        points[:,3] = 1

        # transfrom velodyne points to camera coordinate
        pointsCam = np.matmul(TrVeloToRect, points.T).T
        pointsCam = pointsCam[:,:3]
        # project to image space
        u,v, depth = cam2image(camera, pointsCam.T) # Changing the default cam2image implementation to avoid deprecated numpy aliases
        u = u.astype(int)
        v = v.astype(int)

        # prepare depth map for visualization
        depthMap = np.zeros((camera.height, camera.width))
        depthImage = np.zeros((camera.height, camera.width, 3))
        mask = np.logical_and(np.logical_and(np.logical_and(u>=0, u<camera.width), v>=0), v<camera.height)
        # visualize points within 30 meters
        mask = np.logical_and(np.logical_and(mask, depth>0), depth<max_d)
        depthMap[v[mask],u[mask]] = depth[mask]

        # Saving every depth array that we project as numpy arrays with shape [H, W]
        depth_file_path = os.path.join(depth_path, f"{frame:0{image_name_padding}}.npy")
        np.save(depth_file_path, depthMap)

        # layout = (2,1) if cam_id in [0,1] else (1,2)
        # sub_dir = 'data_rect' if cam_id in [0,1] else 'data_rgb'

        # load RGB image for visualization
        # imagePath = os.path.join(kitti360path, 'data_2d_raw', sequence, 'image_%02d' % cam_id, sub_dir, '%010d.png' % frame)
        # if not os.path.isfile(imagePath):
        #     raise RuntimeError('Image file %s does not exist!' % imagePath)

        # colorImage = np.array(Image.open(imagePath)) / 255.
        # depthImage = cm(depthMap/depthMap.max())[...,:3]
        # colorImage[depthMap>0] = depthImage[depthMap>0]

        # fig, axs = plt.subplots(*layout, figsize=(18,12))
        # axs[0].imshow(depthMap, cmap='jet')
        # axs[0].title.set_text('Projected Depth')
        # axs[0].axis('off')
        # axs[1].imshow(colorImage)
        # axs[1].title.set_text('Projected Depth Overlaid on Image')
        # axs[1].axis('off')
        # plt.suptitle('Sequence %04d, Camera %02d, Frame %010d' % (seq, cam_id, frame))
        # plt.show()