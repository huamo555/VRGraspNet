
import os
import numpy as np
import scipy.io as scio
from PIL import Image
import torchvision.transforms as transform
import torch
import collections.abc as container_abcs
from torch.utils.data import Dataset
from tqdm import tqdm
import MinkowskiEngine as ME
from data_utils import CameraInfo, transform_point_cloud, create_point_cloud_from_depth_image, get_workspace_mask

import time

# 利用split对数据集进行了切分，默认是train，是可以选择的，选择train，test，test_seen，test_similar，test_novel
class  GraspNetDataset(Dataset):
    def __init__(self, root, grasp_labels=None, camera='kinect', split='train', num_points=20000,
                 voxel_size=0.005, remove_outlier=True, augment=False, load_label=True):
        assert (num_points <= 50000) # 这是一个断言，点云的点要小于50000
        self.root = root
        self.split = split
        self.voxel_size = voxel_size
        self.num_points = num_points
        self.remove_outlier = remove_outlier
        self.grasp_labels = grasp_labels
        self.camera = camera
        self.augment = augment
        self.load_label = load_label
        self.collision_labels = {}
        # self.view_list = []
        print("dataset111")
        if split == 'train':
            self.sceneIds = list(range(0 , 100))
        elif split == 'train1':
            self.sceneIds = list(range(22,23))
        elif split == 'test':
            self.sceneIds = list(range(100, 190))
        elif split == 'test_seen':
            self.sceneIds = list(range(100, 130))
        elif split == 'test_similar':
            self.sceneIds = list(range(130, 160))
        elif split == 'test_novel':
            self.sceneIds = list(range(160, 190))
        elif split == 'test_novel1':
            self.sceneIds = list(range(160, 161))
        # 这段代码的作用是将self.sceneIds列表中的元素转换为形如'scene_0001'、'scene_0002'等的字符串格式，
        # 并将转换后的字符串列表赋值给self.sceneIds。这样可以确保场景标识符的格式统一，并且保证字符串的长度一致。
        # 将当前元素x转换为字符串，并将其填充到'scene_'后面，保证字符串的长度为4位，不足的部分用零填充
        # 例如，如果self.sceneIds原本包含[1, 2, 3, 4]，
        # 经过这段代码的处理后，self.sceneIds将变为['scene_0001', 'scene_0002', 'scene_0003', 'scene_0004']
        self.sceneIds = ['scene_{}'.format(str(x).zfill(4)) for x in self.sceneIds]

        self.rgbpath = []
        self.depthpath = []
        self.labelpath = []
        self.metapath = []
        self.scenename = []
        self.frameid = []
        self.graspnesspath = []
        # 这段代码用于加载数据路径和碰撞标签，并将它们存储在相应的列表或字典中
        # for x in tqdm(self.sceneIds, desc='Loading data path and collision labels...'):：这是一个循环语句，
        # 遍历self.sceneIds列表中的每个场景标识符，并使用tqdm库显示进度条，描述为"Loading data path and collision labels..."。
        for x in tqdm(self.sceneIds, desc='Loading data path and collision labels...'):
            # 遍历从0到255的整数，用于表示每个场景中的图像编号
            for img_num in range(256):
                # 将深度图像的路径添加到self.depthpath列表中。
                # 路径由root（根目录）、scenes（场景文件夹）、场景标识符x、相机参数camera、depth（深度图像文件夹）和图像编号生成。
                self.rgbpath.append(os.path.join(root, 'scenes', x, camera, 'rgb', str(img_num).zfill(4) + '.png'))
                self.depthpath.append(os.path.join(root, 'scenes', x, camera, 'depth', str(img_num).zfill(4) + '.png'))
                # self.depthpath.append(os.path.join("/data3/gaoyuming/mutilview_project/graspness_depthguji/logs/0919_re_depth/", x, camera, 'depth', str(img_num).zfill(4) + '.png'))
                # self.depthpath.append(os.path.join("/data3/gaoyuming/mutilview_project/graspness_depthguji/logs/1013_kn_depth/", x, camera, 'depth', str(img_num).zfill(4) + '.png'))
                
                # ：将标签图像的路径添加到self.labelpath列表中。
                self.labelpath.append(os.path.join(root, 'scenes', x, camera, 'label', str(img_num).zfill(4) + '.png'))
                # 元数据文件的路径添加到self.metapath列表中，文件扩展名变为.mat
                self.metapath.append(os.path.join(root, 'scenes', x, camera, 'meta', str(img_num).zfill(4) + '.mat'))
                # 将抓取性能标签文件的路径添加到self.graspnesspath列表中，文件扩展名变为.npy

                self.graspnesspath.append(os.path.join(root, 'graspness_label', x, camera, str(img_num).zfill(4) + '.npy'))
                # self.graspnesspath.append(os.path.join(root, 'graspness_0919_graspness', x, camera, str(img_num).zfill(4) + '.npy'))
                
                # self.graspnesspath.append(os.path.join(root, 'graspness_1003_graspness', x, camera, str(img_num).zfill(4) + '.npy'))
                
                self.scenename.append(x.strip())# 将场景标识符x的字符串形式添加到self.scenename列表中。
                self.frameid.append(img_num) # 将图像编号img_num添加到self.frameid列表中。
            if self.load_label:# 表示需要加载碰撞标签。
                # 加载碰撞标签文件的路径，并将其保存在collision_labels变量中。
                collision_labels = np.load(os.path.join(root, 'collision_label', x.strip(), 'collision_labels.npz'))
                self.collision_labels[x.strip()] = {}# 加载碰撞标签文件的路径，并将其保存在collision_labels变量中。
                for i in range(len(collision_labels)):
                    # 将碰撞标签数据存储在self.collision_labels字典中，键为场景标识符x.strip()，值为一个以索引i为键的碰撞标签数组
                    self.collision_labels[x.strip()][i] = collision_labels['arr_{}'.format(i)]

    # 该方法可以用于获取数据集中所有场景的名称
    def scene_list(self):
        return self.scenename
    # 方法通常在迭代数据集时使用，以确定数据集的大小。
    def __len__(self):
        return len(self.depthpath)
    
    # 这段代码是一个数据增强的函数augment_data，
    # 它接收点云数据和物体姿态列表作为输入，并对其进行一系列的随机变换，以增加数据的多样性：
    def augment_data(self, point_clouds, object_poses_list):
        # 沿着YZ平面进行翻转。
        if np.random.random() > 0.5:
            flip_mat = np.array([[-1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]])
            point_clouds = transform_point_cloud(point_clouds, flip_mat, '3x3')
            for i in range(len(object_poses_list)):
                object_poses_list[i] = np.dot(flip_mat, object_poses_list[i]).astype(np.float32)

        # Rotation along up-axis/Z-axis：沿着上轴（Z轴）进行旋转。
        rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6  # -30 ~ +30 degree
        c, s = np.cos(rot_angle), np.sin(rot_angle)
        rot_mat = np.array([[1, 0, 0],
                            [0, c, -s],
                            [0, s, c]])
        point_clouds = transform_point_cloud(point_clouds, rot_mat, '3x3')
        for i in range(len(object_poses_list)):
            object_poses_list[i] = np.dot(rot_mat, object_poses_list[i]).astype(np.float32)

        return point_clouds, object_poses_list
    

    # 根据代码中的逻辑，load_label是一个布尔值，表示是否加载标签信息。如果load_label为True，
    # 则调用get_data_label方法获取带有标签的数据样本；
    # 如果load_label为False，则调用get_data方法获取不带标签的数据样本
    def __getitem__(self, index):
        if self.load_label:
            return self.get_data_label(index)
        else:
            return self.get_data(index)


    # 得到点云数据
    def get_data(self, index, return_raw_cloud=False):
        # 通过索引index读取深度图像（depth）和标签图像（seg）
        depth = np.array(Image.open(self.depthpath[index]))
        rgb = np.array(Image.open(self.rgbpath[index])) / 255.0
        rgb_resize = Image.open(self.rgbpath[index])
        transform_resize = transform.Resize([360, 640])
        rgb_resize = transform_resize(rgb_resize)
        rgb_resize = np.array(rgb_resize) / 255.0

        point_clouds_idx_in_rgbd = np.arange(720 * 1280).reshape(720, 1280)


        seg = np.array(Image.open(self.labelpath[index]))
        # 使用scio.loadmat加载元数据（meta）
        meta = scio.loadmat(self.metapath[index])
        # 获取场景名称（scene）
        scene = self.scenename[index]
        # 尝试从元数据中获取相机内参（intrinsic）和深度因子（factor_depth），并将它们赋值给相机信息对象（camera）。
        try:
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e))
            print(scene)
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
                            factor_depth)

        # 使用深度图像和相机信息生成有序的点云（cloud）
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
        # get valid points 
        # 创建深度掩码（depth_mask），其中大于0的像素被标记为True
        depth_mask = (depth > 0)

        # 如果设置了remove_outlier标志，加载相机姿态和对齐矩阵，并使用它们计算变换矩阵（trans）。
        # 然后，使用点云和标签图像以及变换矩阵生成工作空间掩码（workspace_mask），
        # 其中点云中与工作空间外部或离群值相关的点被排除。最后，通过将深度掩码和工作空间掩码进行逻辑与操作，生成最终的掩码（mask）。
        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[self.frameid[index]])
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)
        else:
            mask = depth_mask
        cloud_masked = cloud[mask]
        point_clouds_idx_in_rgbd_masked = point_clouds_idx_in_rgbd[mask]


        # 将掩码应用于点云，得到经过掩码过滤的点云（cloud_masked）
        # 如果设置了return_raw_cloud标志，直接返回经过掩码过滤的点云（cloud_masked）
        if return_raw_cloud:
            return cloud_masked
        # 如果经过掩码过滤的点云数量大于等于num_points，
        # 则从中随机选择num_points个点，否则，从中随机选择剩余点数，并将两部分点云合并为idxs。
        if len(cloud_masked) >= self.num_points:
            idxsa = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points - len(cloud_masked), replace=True)
            idxsa = np.concatenate([idxs1, idxs2], axis=0)

        # 从经过掩码过滤的点云中提取相应索引的点，得到最终的采样点云（cloud_sampled）
        cloud_sampled = cloud_masked[idxsa]
        point_clouds_idx_in_rgbd__sampled = point_clouds_idx_in_rgbd_masked[idxsa]

        idxs_hang = np.floor_divide(point_clouds_idx_in_rgbd__sampled, 1280)
        idxs_lie = point_clouds_idx_in_rgbd__sampled % 1280
        # 获得在缩小图上的位置
        idxs_hang = np.floor_divide(idxs_hang, 2)
        idxs_lie = np.floor_divide(idxs_lie, 2)
        # 获得在缩小图上的索引
        point_clouds_idx_in_rgbd__sampled = idxs_hang * 640 + idxs_lie
        
        # 构建一个字典（ret_dict），包含采样点云（point_clouds）、坐标（coors）和特征（feats）。
        # 返回字典（ret_dict）作为数据样本
        ret_dict = {'point_clouds': cloud_sampled.astype(np.float32),
                    'coors': cloud_sampled.astype(np.float32) / self.voxel_size,
                    'feats': np.ones_like(cloud_sampled).astype(np.float32),
                    'depth': depth.astype(np.int64),
                    'rgb': rgb.astype(np.int64),
                    'point_clouds_idx_in_rgbd' : point_clouds_idx_in_rgbd__sampled.astype(np.int64),
                    'idx': idxsa.astype(np.int64),
                    'rgb_resize': rgb_resize.astype(np.int64),
                    }
        return ret_dict


    def get_data_label(self, index):
        # 通过索引index读取深度图像（depth）和标签图像（seg）
        depth = np.array(Image.open(self.depthpath[index]))
        rgb = np.array(Image.open(self.rgbpath[index])) / 255.0
        rgb_resize = Image.open(self.rgbpath[index])
        transform_resize = transform.Resize([360, 640])
        rgb_resize = transform_resize(rgb_resize)
        rgb_resize = np.array(rgb_resize) / 255.0

        point_clouds_idx_in_rgbd = np.arange(720 * 1280).reshape(720, 1280)

        seg = np.array(Image.open(self.labelpath[index]))
        # 使用scio.loadmat加载元数据（meta）
        meta = scio.loadmat(self.metapath[index])
        # 从路径self.graspnesspath[index]加载抓取度（graspness），该值对应工作空间掩码点云中的每个点
        graspness = np.load(self.graspnesspath[index])  # for each point in workspace masked point cloud
        # 获取场景名称（scene）
        scene = self.scenename[index]

        # 尝试从元数据中获取目标物体的索引（obj_idxs）、姿态（poses）、相机内参（intrinsic）和深度因子（factor_depth），
        # 并将它们赋值给相机信息对象（camera）
        try:
            obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
            poses = meta['poses']
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e))
            print(scene)
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
                            factor_depth)

        # generate cloud
        cloud  = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # get valid points 将掩码应用于点云和标签图像，得到经过掩码过滤的点云（cloud_masked）和标签图像（seg_masked）
        depth_mask = (depth > 0)
        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[self.frameid[index]])
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)
        else:
            mask = depth_mask
        cloud_masked = cloud[mask]
        seg_masked = seg[mask]
        point_clouds_idx_in_rgbd_masked = point_clouds_idx_in_rgbd[mask]


        # sample points 从经过掩码过滤的点云和标签图像中提取相应索引的点，
        # 得到最终的采样点云（cloud_sampled）、采样标签图像（seg_sampled）和采样抓取度（graspness_sampled）
        if len(cloud_masked) >= self.num_points:
            idxsa = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points - len(cloud_masked), replace=True)
            idxsa = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxsa]
        seg_sampled = seg_masked[idxsa]
        graspness_sampled = graspness[idxsa]
        point_clouds_idx_in_rgbd__sampled = point_clouds_idx_in_rgbd_masked[idxsa]
        objectness_label = seg_sampled.copy()
        
        # 将采样标签图像复制为目标性标签（objectness_label），超过1的值被替换为1
        objectness_label[objectness_label > 1] = 1

        # 初始化空列表，用于存储每个目标物体的
        # 姿态（object_poses_list）、抓取点（grasp_points_list）、抓取宽度（grasp_widths_list）和抓取分数（grasp_scores_list）。
        object_poses_list = []
        grasp_points_list = []
        grasp_widths_list = []
        grasp_scores_list = []

        view_list = []
        index = index % 256
        index_np = np.array([index])

        # 将 NumPy 数组转换为 torch.Tensor 并添加到列表中
        view_list.append(torch.from_numpy(index_np))
        view_list_np = np.asarray(view_list)

        empty_arr_list = np.zeros_like(idxsa)
        # 遍历目标物体索引（obj_idxs）和对应的姿态（poses）
        for i, obj_idx in enumerate(obj_idxs):
            # 如果采样标签图像中与目标物体索引相等的像素数量小于50，则跳过当前目标物体
            if (seg_sampled == obj_idx).sum() < 50:
                continue
            # 将目标物体的姿态添加到姿态列表中（object_poses_list）
            object_poses_list.append(poses[:, :, i])
            # 从预先计算的抓取点（points）、抓取宽度（widths）和抓取分数（scores）中随机选择一部分点作为采样点。
            points, widths, scores = self.grasp_labels[obj_idx]
            # 获取与当前目标物体相关的碰撞标签（collision）。
            collision = self.collision_labels[scene][i]  # (Np, V, A, D)
            
            # 从抓取分数中排除与碰撞标签对应的点，并将结果赋值给抓取分数（scores）,scores[collision] = 0
            # 将采样的抓取点、抓取宽度和抓取分数分别添加到对应的列表中

           
            idxs = np.random.choice(len(points), min(max(int(len(points) / 4), 300), len(points)), replace=False)
            grasp_points_list.append(points[idxs])
            grasp_widths_list.append(widths[idxs])
            collision = collision[idxs].copy()
            scores = scores[idxs].copy()
            scores[collision] = 0
            grasp_scores_list.append(scores)
            
        unique_elements = list(set(empty_arr_list))
        # 如果设置了数据增强（augment）标志，对采样的点云和目标物体姿态进行数据增强操作，然后返回增强后的点云和姿态列表。
        if self.augment:
            cloud_sampled, object_poses_list = self.augment_data(cloud_sampled, object_poses_list)
        # 构建包含所有数据的字典（ret_dict）

        idxs_hang = np.floor_divide(point_clouds_idx_in_rgbd__sampled, 1280)
        idxs_lie = point_clouds_idx_in_rgbd__sampled % 1280
        # 获得在缩小图上的位置
        idxs_hang = np.floor_divide(idxs_hang, 2)
        idxs_lie = np.floor_divide(idxs_lie, 2)
        # 获得在缩小图上的索引
        point_clouds_idx_in_rgbd__sampled = idxs_hang * 640 + idxs_lie

        ret_dict = {
                    'point_clouds': cloud_sampled.astype(np.float32),
                    'coors': cloud_sampled.astype(np.float32) / self.voxel_size,
                    'feats': np.ones_like(cloud_sampled).astype(np.float32),
                    'graspness_label': graspness_sampled.astype(np.float32),
                    'objectness_label': objectness_label.astype(np.int64),
                    'object_poses_list': object_poses_list,
                    'grasp_points_list': grasp_points_list,
                    'grasp_widths_list': grasp_widths_list,
                    'grasp_scores_list': grasp_scores_list,
                    'view_nums': view_list_np.astype(np.int64),
                    'depth': depth.astype(np.int64),
                    'rgb': rgb.astype(np.int64),
                    'idx': idxsa.astype(np.int64),
                    'rgb_resize': rgb_resize.astype(np.int64),
                    'point_clouds_idx_in_rgbd' : point_clouds_idx_in_rgbd__sampled.astype(np.int64),#用于从cnn输出的前景标签中获得对应的点云标签  
                    }
        return ret_dict
# 用于加载抓取标签数据。
def load_grasp_labels(root):
    # 函数首先创建了一个包含数字1到88的列表obj_names，这些数字代表不同的物体名称或类别
    obj_names = list(range(1, 89))
    grasp_labels = {}
    # 函数使用np.load加载与当前物体名称对应的抓取标签数据文件，
    # 文件路径通过os.path.join函数构建得到。加载的数据以字典的形式存储在变量label中。
    # 函数将加载的抓取标签数据存储在一个名为grasp_labels的字典中，以物体名称作为键。
    # 字典的值是一个包含三个数组的元组，分别是抓取点坐标（points）、抓取宽度（width）和抓取分数（scores）。
    # 数组的数据类型被转换为np.float32
    for obj_name in tqdm(obj_names, desc='Loading grasping labels...'):
        label = np.load(os.path.join(root, 'grasp_label_simplified', '{}_labels.npz'.format(str(obj_name - 1).zfill(3))))
        grasp_labels[obj_name] = (label['points'].astype(np.float32), label['width'].astype(np.float32),
                                  label['scores'].astype(np.float32))

    return grasp_labels

# 这个函数用于对数据样本进行批处理操作，将稀疏坐标和特征进行拼接和量化，并返回处理后的数据字典
def minkowski_collate_fn(list_data):
    coordinates_batch, features_batch = ME.utils.sparse_collate([d["coors"] for d in list_data],
                                                                [d["feats"] for d in list_data])
    coordinates_batch, features_batch, _, quantize2original = ME.utils.sparse_quantize(
        coordinates_batch, features_batch, return_index=True, return_inverse=True)
    res = {
        "coors": coordinates_batch,
        "feats": features_batch,
        "quantize2original": quantize2original
    }
    # 对批量数据进行处理和拼接。将原始数据转换为可以输入深度学习模型的形式
    def collate_fn_(batch):
        if type(batch[0]).__module__ == 'numpy':
            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        elif isinstance(batch[0], container_abcs.Sequence):
            return [[torch.from_numpy(sample) for sample in b] for b in batch]
        elif isinstance(batch[0], container_abcs.Mapping):
            for key in batch[0]:
                if key == 'coors' or key == 'feats':
                    continue
                res[key] = collate_fn_([d[key] for d in batch])
            return res
    res = collate_fn_(list_data)

    return res
