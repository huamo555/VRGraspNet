import open3d as o3d
import scipy.io as scio
from PIL import Image
import os
import numpy as np
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from data_utils import get_workspace_mask, CameraInfo, create_point_cloud_from_depth_image

data_path = "C:/Users/Administrator/Desktop/dataset/"
scene_id = 'scene_0060'
ann_id = '0000'
camera_type = 'realsense'
color = np.array(Image.open("C:/Users/Administrator/Desktop/dataset/scene/scene_0060/realsense/rgb/0000.png"), dtype=np.float32) / 255.0
depth = np.array(Image.open("C:/Users/Administrator/Desktop/dataset/scene/scene_0060/realsense/depth/0000.png"))
seg = np.array(Image.open("C:/Users/Administrator/Desktop/dataset/scene/scene_0060/realsense/label/0000.png"))
meta = scio.loadmat("C:/Users/Administrator/Desktop/dataset/scene/scene_0060/realsense/meta/0000.mat")
intrinsic = meta['intrinsic_matrix']
factor_depth = meta['factor_depth']
camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
point_cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
depth_mask = (depth > 0)
camera_poses = np.load("C:/Users/Administrator/Desktop/dataset/scene/scene_0060/realsense/camera_poses.npy")
align_mat = np.load("C:/Users/Administrator/Desktop/dataset/scene/scene_0060/realsense/cam0_wrt_table.npy" )
trans = np.dot(align_mat, camera_poses[int(ann_id)])
workspace_mask = get_workspace_mask(point_cloud, seg, trans=trans, organized=True, outlier=0.02)
mask = (depth_mask & workspace_mask)
point_cloud = point_cloud[mask]
color = color[mask]
seg = seg[mask]

graspness_full = np.load("C:/Users/Administrator/Desktop/dataset/graspness_label/scene_0060/realsense/0060.npy").squeeze()
graspness_full[seg == 0] = 0.
print('graspness full scene: ', graspness_full.shape, (graspness_full > 0.1).sum())
color[graspness_full > 0.1] = [0., 1., 0.]


cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(point_cloud.astype(np.float32))
cloud.colors = o3d.utility.Vector3dVector(color.astype(np.float32))

o3d.io.write_point_cloud("/data2/gaoyuming/.cache/graspnet-baseline-main/view/temp.ply", cloud)

