import numpy as np
import os
from PIL import Image
import scipy.io as scio
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from data_utils import get_workspace_mask, CameraInfo, create_point_cloud_from_depth_image
from knn.knn_modules import knn
import torch
from graspnetAPI.utils.xmlhandler import xmlReader
from graspnetAPI.utils.utils import get_obj_pose_list, transform_points
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default=None, required=True)
parser.add_argument('--camera_type', default='kinect', help='Camera split [realsense/kinect]')

# 生成物体的抓取性标签（graspness label），进行了碰撞度检测
if __name__ == '__main__':
    cfgs = parser.parse_args() # parser.parse_args()将命令行参数进行解析，将解析结果保存在cfgs变量中
    dataset_root = cfgs.dataset_root   # set dataset root
    camera_type = cfgs.camera_type   # kinect / realsense
    save_path_root = os.path.join(dataset_root, 'graspness') # 设置保存路径的根目录。

    num_views, num_angles, num_depths = 300, 12, 4 # 三个变量num_views、num_angles和num_depths，分别赋值为300、12和4，用于设置视角数量、角度数量和深度数量。
    fric_coef_thresh = 0.8 # 设置摩擦系数的阈值
    point_grasp_num = num_views * num_angles * num_depths # 将num_views、num_angles和num_depths相乘，计算出point_grasp_num，表示每个点的抓取标签的数量
    for scene_id in range(100):
        save_path = os.path.join(save_path_root, 'scene_' + str(scene_id).zfill(4), camera_type)
        if not os.path.exists(save_path): # 检查是否存在save_path目录，如果不存在则使用os.makedirs()函数创建该目录
            os.makedirs(save_path)
        # # 使用np.load()函数加载碰撞标签文件
        labels = np.load( 
            os.path.join(dataset_root, 'collision_label', 'scene_' + str(scene_id).zfill(4), 'collision_labels.npz'))
        collision_dump = []
        # 通过循环遍历标签数据的长度，将每个标签数据labels['arr_{}'.format(j)]添加到collision_dump列表中。
        for j in range(len(labels)):
            collision_dump.append(labels['arr_{}'.format(j)])

        for ann_id in range(256): # 0 -> 256个视角
            # get scene point cloud
            # 这段代码用于生成场景，并获取场景相关的深度图、分割图、元数据和点云数据
            print('generating scene: {} ann: {}'.format(scene_id, ann_id))
            # Image.open()函数打开深度图像文件，将打开的图像文件转换为NumPy数组，并将其存储在depth变量中
            depth = np.array(Image.open(os.path.join(dataset_root, 'scenes', 'scene_' + str(scene_id).zfill(4),
                                                     camera_type, 'depth', str(ann_id).zfill(4) + '.png')))
            # Image.open()函数打开分割图像文件，将打开的图像文件转换为NumPy数组，并将其存储在seg变量中
            seg = np.array(Image.open(os.path.join(dataset_root, 'scenes', 'scene_' + str(scene_id).zfill(4),
                                                   camera_type, 'label', str(ann_id).zfill(4) + '.png')))
            # 使用Image.open()函数打开元数据文件，加载后的元数据保存在meta变量中
            meta = scio.loadmat(os.path.join(dataset_root, 'scenes', 'scene_' + str(scene_id).zfill(4),
                                             camera_type, 'meta', str(ann_id).zfill(4) + '.mat'))
            # 从元数据meta中提取相机内参矩阵intrinsic和深度因子factor_depth
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
            # 使用提取的相机内参矩阵intrinsic和深度因子factor_depth创建CameraInfo对象，用于表示相机信息
            camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
                                factor_depth)
            # 使用create_point_cloud_from_depth_image()函数从深度图像和相机信息生成点云数据。点云数据存储在cloud变量中。
            cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

            """背景检测,得到最终点云"""
            # remove outlier and get objectness label
            depth_mask = (depth > 0) # 创建深度掩码depth_mask，将其设置为布尔类型的数组，表示深度图中大于0的像素点
            # 使用np.load()函数加载相机姿态文件
            camera_poses = np.load(os.path.join(dataset_root, 'scenes', 'scene_' + str(scene_id).zfill(4),
                                                camera_type, 'camera_poses.npy'))
            # 从相机姿态camera_poses中获取当前姿态camera_pose，即camera_poses[ann_id]
            camera_pose = camera_poses[ann_id]
            # 使用np.load()函数加载对齐矩阵文件,加载后的对齐矩阵保存在align_mat变量中
            align_mat = np.load(os.path.join(dataset_root, 'scenes', 'scene_' + str(scene_id).zfill(4),
                                             camera_type, 'cam0_wrt_table.npy'))
            # 通过矩阵乘法计算变换矩阵trans，将对齐矩阵align_mat与相机姿态camera_pose相乘
            trans = np.dot(align_mat, camera_pose)
            # 使用get_workspace_mask()函数生成工作空间掩码workspace_mask,掩码根据点云和变换矩阵生成，用于筛选出位于工作空间内的点云
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            # 通过逻辑与运算符&将深度掩码depth_mask和工作空间掩码workspace_mask进行逐元素的逻辑与运算，生成最终的掩码mask。
            mask = (depth_mask & workspace_mask)
            cloud_masked = cloud[mask] # 从点云数据cloud中根据掩码mask筛选出符合条件的点云，存储在cloud_masked变量中
            objectness_label = seg[mask]# 从分割标签seg中根据掩码mask筛选出对应的分割标签，存储在objectness_label变量中

            """读取场景的注释信息和姿态向量，获取物体列表和姿态列表，并加载对应物体的抓取标签数据，将其存储在字典中供后续使用。"""
            # 使用xmlReader类从注释文件中加载场景的注释信息
            # 注释文件的路径由数据集根目录dataset_root、子目录scenes、子目录scene_XXXX、
            # 相机类型（kinect或realsense）、子目录annotations和文件名%04d.xml（使用ann_id进行格式化）组成。加载后的注释信息保存在scene_reader对象中
            scene_reader = xmlReader(os.path.join(dataset_root, 'scenes', 'scene_' + str(scene_id).zfill(4),
                                                  camera_type, 'annotations', '%04d.xml' % ann_id))
            # 使用getposevectorlist()方法从注释信息中获取姿态向量列表pose_vectors
            pose_vectors = scene_reader.getposevectorlist()
            # 使用get_obj_pose_list()函数根据相机姿态camera_pose和姿态向量列表pose_vectors获取物体列表obj_list和姿态列表pose_list
            obj_list, pose_list = get_obj_pose_list(camera_pose, pose_vectors)
            grasp_labels = {}
            # 通过循环遍历物体列表obj_list，依次加载对应物体的抓取标签数据
            for i in obj_list:
                file = np.load(os.path.join(dataset_root, 'grasp_label', '{}_labels.npz'.format(str(i).zfill(3))))
                grasp_labels[i] = (file['points'].astype(np.float32), file['offsets'].astype(np.float32),
                                   file['scores'].astype(np.float32))
                
            """这段代码用于处理抓取标签数据，根据一些条件筛选有效的抓取点，并对抓取点进行变换处理"""
            grasp_points = [] # 存储处理后的抓取点。
            grasp_points_graspness = [] # 存储处理后的抓取可信度
            # 通过循环遍历物体列表obj_list和姿态列表pose_list，依次处理每个物体的抓取标签数据
            for i, (obj_idx, trans_) in enumerate(zip(obj_list, pose_list)):
                sampled_points, offsets, fric_coefs = grasp_labels[obj_idx] # 首先从抓取标签数据字典grasp_labels中获取当前物体的抓取标签数据
                collision = collision_dump[i]  # 从碰撞信息数组collision_dump中获取当前物体的碰撞信息collision
                num_points = sampled_points.shape[0] # 计算采样点云的数量num_points
                
                # 通过一系列条件筛选生成有效抓取掩码valid_grasp_mask
                # 这些条件包括摩擦系数小于等于fric_coef_thresh、摩擦系数大于0，并且没有碰撞
                valid_grasp_mask = ((fric_coefs <= fric_coef_thresh) & (fric_coefs > 0) & ~collision)
                valid_grasp_mask = valid_grasp_mask.reshape(num_points, -1)
                # 然后，计算抓取可信度graspness，通过对有效抓取掩码沿着列方向求和，并将结果除以抓取点数量point_grasp_num
                graspness = np.sum(valid_grasp_mask, axis=1) / point_grasp_num
                # 通过变换函数transform_points()对采样点云sampled_points进行变换，变换矩阵为trans_，用于将采样点云从物体坐标系变换到场景坐标系
                target_points = transform_points(sampled_points, trans_)
                # 这一步是为了将点云从场景坐标系变换回相机坐标系。
                target_points = transform_points(target_points, np.linalg.inv(camera_pose))  # fix bug
                grasp_points.append(target_points)
                # 将变换后的点云target_points添加到grasp_points列表中，
                # 并将抓取可信度graspness转换为形状为(num_points, 1)的数组，并添加到grasp_points_graspness列表中
                grasp_points_graspness.append(graspness.reshape(num_points, 1))

            """之前处理得到的抓取点和抓取可信度进行进一步的处理和转换"""
            # 通过np.vstack()函数将列表grasp_points中的所有抓取点堆叠为一个二维数组，并将结果赋值给grasp_points变量
            grasp_points = np.vstack(grasp_points)
            # 使用np.vstack()函数将列表grasp_points_graspness中的所有抓取可信度堆叠为一个二维数组，并将结果赋值给grasp_points_graspness变量
            grasp_points_graspness = np.vstack(grasp_points_graspness)
            
            # 将grasp_points和grasp_points_graspness转换为PyTorch的Tensor，并将它们移动到CUDA设备上进行加速计算
            grasp_points = torch.from_numpy(grasp_points).cuda()
            grasp_points_graspness = torch.from_numpy(grasp_points_graspness).cuda()
            # # 这样的维度重排和扩展通常是为了与模型的输入要求相匹配。
            grasp_points = grasp_points.transpose(0, 1).contiguous().unsqueeze(0)
            
            # 获取变量cloud_masked的点云数量masked_points_num
            masked_points_num = cloud_masked.shape[0]
            cloud_masked_graspness = np.zeros((masked_points_num, 1))
            part_num = int(masked_points_num / 10000) # 计算变量masked_points_num除以10000的整数部分，并将结果赋值给part_num变量

            """点云中寻找与抓取点最近的点，并将对应的抓取可信度赋值给点云的抓取可信度"""
            for i in range(1, part_num + 2):   # lack of cuda memory
                if i == part_num + 1:
                    cloud_masked_partial = cloud_masked[10000 * part_num:]
                    if len(cloud_masked_partial) == 0:
                        break
                else:
                    cloud_masked_partial = cloud_masked[10000 * (i - 1):(i * 10000)]
                cloud_masked_partial = torch.from_numpy(cloud_masked_partial).cuda()
                cloud_masked_partial = cloud_masked_partial.transpose(0, 1).contiguous().unsqueeze(0)
                nn_inds = knn(grasp_points, cloud_masked_partial, k=1).squeeze() - 1
                cloud_masked_graspness[10000 * (i - 1):(i * 10000)] = torch.index_select(
                    grasp_points_graspness, 0, nn_inds).cpu().numpy()

            max_graspness = np.max(cloud_masked_graspness)
            min_graspness = np.min(cloud_masked_graspness)
            # 进行归一化
            cloud_masked_graspness = (cloud_masked_graspness - min_graspness) / (max_graspness - min_graspness)

            np.save(os.path.join(save_path, str(ann_id).zfill(4) + '.npy'), cloud_masked_graspness)
