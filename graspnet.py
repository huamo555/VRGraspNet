""" GraspNet baseline model definition.
    Author: chenxi-wang
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import MinkowskiEngine as ME
import random
from torch_geometric.nn import DynamicEdgeConv
import plistlib as plt
import open3d as o3d

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from backbone_resunet14 import MinkUNet14D
from modules import ApproachNet, GraspableNet, CloudCrop, SWADNet
from loss_utils import GRASP_MAX_WIDTH, NUM_VIEW, NUM_ANGLE, NUM_DEPTH, GRASPNESS_THRESHOLD, M_POINT
from label_generation import process_grasp_labels, match_grasp_view_and_label, batch_viewpoint_params_to_matrix
from pointnet2_utils import furthest_point_sample, gather_operation
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import torch
import math
import torch.nn.functional as F
from obj_cls.model.PointNet_PAConv import PAConv  # 假设你的PAConv类在your_module模块中

from SE_resUnet import SeResUNet 

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return F.relu(self.fc(x))


def cross_attention(q, k, v):
    # q: 查询 (点云特征), 形状 [batch_size, num_queries, feature_dim]
    # k: 键 (图像特征), 形状 [batch_size, num_keys, feature_dim]
    # v: 值 (图像特征), 形状 [batch_size, num_values, feature_dim]

    # 确保 k 和 v 的数量是相同的，这里我们假设 num_keys = num_values
    batch_size, num_keys, feature_dim = k.shape

    # 计算相似度矩阵，使用点积作为相似度度量
    similarity_matrix = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(feature_dim)
    # Softmax 归一化
    attention_weights = F.softmax(similarity_matrix, dim=-1)
    
    # 加权 Value
    weighted_v = torch.matmul(attention_weights, v)
    
    return weighted_v 





args = {
    'k_neighbors': 20,
    'calc_scores': 'softmax',
    'num_matrices': [8, 8, 8],
    'dropout': 0.5 
}


def generate_loacl_region(rgb, row_bat_fps_graspable, col_bat_fps_graspable):
    # 假设 row 和 col 是 [2, 1024] 的张量，包含每个点的行和列索引

    B, feature = row_bat_fps_graspable.shape
    row = row_bat_fps_graspable
    col = col_bat_fps_graspable
    patches_feature = []

    for ii in range(B):
        row_ii = row[ii]
        col_ii = col[ii]
        patches_ii = rgb[ii][ :, row_ii, col_ii]
        patches_feature.append(patches_ii)

    return patches_feature

# class GraspNet(nn.Module):
#     def __init__(self, cylinder_radius=0.05, seed_feat_dim=512, is_training=True):
#         super().__init__()
#         self.is_training = is_training
#         self.seed_feature_dim = seed_feat_dim
#         self.num_depth = NUM_DEPTH
#         self.num_angle = NUM_ANGLE
#         self.M_points = M_POINT
#         self.num_view = NUM_VIEW
     
#         # 名为backbone的MinkUNet14D模型，用于提取点云的特征
#         # D=3表示数据的输入是三维的
#         self.backbone = MinkUNet14D(in_channels=3, out_channels=self.seed_feature_dim, D=3)
#         # 名为graspable的GraspableNet模型，用于预测点云中每个点的可抓取性
#         self.graspable = GraspableNet(seed_feature_dim=self.seed_feature_dim)
#         # ApproachNet模型，用于预测点云中每个点的抓取方向 每个点预测300个视角，预测得分
#         self.rotation = ApproachNet(self.num_view, seed_feature_dim=self.seed_feature_dim, is_training=self.is_training)
#         # CloudCrop模型，用于从点云中裁剪出与抓取相关的局部区域
#         self.crop = CloudCrop(nsample=16, cylinder_radius=cylinder_radius, seed_feature_dim=self.seed_feature_dim)
#         # SWADNet模型，用于预测抓取的得分和宽度。
#         self.swad = SWADNet(num_angle=self.num_angle, num_depth=self.num_depth)

#         self.fuse_multi_scale = nn.Conv1d(256 + 256, 256, 1)

#         # 前拼接
#         self.fuse_multi_scale_ronghe = nn.Conv1d(256 + 512, 512, 1) 

#         # self.fuse_multi_scale_attn_hou = nn.Conv1d(512 + 256, 256, 1) 

#         self.rgbd_feature_dim = 512  #1  2 3
#         self.rgbd_backbone = SeResUNet(3, out_channels=self.rgbd_feature_dim)

#         self.mlp_point = MLP(1024, 1024)
#         self.mlp_rgb = MLP(1024, 1024)

#     def forward(self, end_points):
#         # 获取输入的点云坐标和特征，并通过backbone模型提取点云的特征
#         seed_xyz = end_points['point_clouds']  # use all sampled point cloud, B*Ns*3
#         idxa = end_points['idx']
#         rgb = end_points['rgb']
#         rgb_resize = end_points['rgb_resize']
#         depth = end_points['depth']

#         idx_640_360 = end_points['point_clouds_idx_in_rgbd']

#         B, point_num, _ = seed_xyz.shape  # batch _size
#         # point-wise features
#         coordinates_batch = end_points['coors']  # 从end_points字典中获取coors坐标数据N*3。
#         features_batch = end_points['feats']     # 从end_points字典中获取feats特征数据B*C。
#         mink_input = ME.SparseTensor(features_batch, coordinates=coordinates_batch)# 创建一个稀疏张量mink_input ,用上述提到的 coors坐标数据 和 feats特征数据。 
#         """[B*Ns*(C+3) --> B*Ns*512]"""
#         seed_features = self.backbone(mink_input).F # mink_input [B*Ns*（C+3）--> B*Ns*512] 输入到backbone模型中，并获取输出的特征数据seed_features
#         # 根据end_points['quantize2original']对seed_features进行索引，
#         # 以将其转换为原始点云坐标的顺序。然后，将其形状重新调整为(B, point_num, -1)，其中B表示批次大小，point_num表示点的数量
#         # 通过transpose(1, 2)操作将seed_features的维度转置，将点的数量维度放在第二个维度上
#         seed_features = seed_features[end_points['quantize2original']].view(B, point_num, -1).transpose(1, 2)
        
#         # 将提取到的特征输入到graspable模型中进行可抓取性预测
#         # 根据可抓取性预测结果和阈值，生成可抓取性掩码。
#         end_points = self.graspable(seed_features, end_points)# 将 seed_features 与 end_points 一起作为参数传递到 self.graspable 函数中。
#         seed_features_flipped = seed_features.transpose(1, 2)  # B*Ns*feat_dim --> B*Ns*512
#         # # 从 end_points 中获取 objectness_score 和 graspness_score。
#         objectness_score = end_points['objectness_score']
#         graspness_score = end_points['graspness_score'].squeeze(1)
#         objectness_pred = torch.argmax(objectness_score, 1)    # objectness_score 的第一个维度上执行最大值索引,得到是物体还是是背景
#         objectness_mask = (objectness_pred == 1)               # 进行mask
#         graspness_mask = graspness_score > GRASPNESS_THRESHOLD # 抓取得分大于阈值
#         graspable_mask = objectness_mask & graspness_mask      # 指示哪些抓取点既被预测为属于物体，又具有足够的可抓取程度
        
#         """FPS 从中取出1024给点, 随机取的"""
#         # 根据可抓取性掩码对点云的特征进行筛选，得到可抓取点的特征和坐标。
#         # 通过最远点采样（furthest point sampling）选取一定数量（M_POINT）的采样点，并将其特征和坐标保存下来
#         seed_features_graspable = []
#         seed_xyz_graspable = []
#         idxa_bat_fps_graspable = []
#         idx_640_360_fps_graspable = []
#         graspable_num_batch = 0.
        

#         for i in range(B):
            
#             cur_mask = graspable_mask[i]        # 从graspable_mask中获取当前样本的掩码(cur_mask)，用于筛选出可抓取的点

#             graspable_num_batch += cur_mask.sum() # 对当前样本中可抓取的点进行统计，计算可抓取点的数量，并将其累加到graspable_num_batch变量中
#             cur_feat = seed_features_flipped[i][cur_mask]  # Ns*feat_dim 从seed_features_flipped中获取当前样本的特征(cur_feat)，并根据掩码进行筛选，得到可抓取点对应的特征
#             cur_seed_xyz = seed_xyz[i][cur_mask]  # *Ns3  从seed_xyz中获取当前样本的坐标(cur_seed_xyz)，并根据掩码进行筛选，得到可抓取点的坐标
#             idxa_bat = idxa[i][cur_mask]
#             idx_640_360_bat = idx_640_360[i][cur_mask]

#             cur_seed_xyz = cur_seed_xyz.unsqueeze(0) # cur_seed_xyz表示当前的种子点坐标 1*M*3 增加一个维度
#             fps_idxs = furthest_point_sample(cur_seed_xyz, self.M_points) # fps_idxs 是通过调用函数 furthest_point_sample 对 cur_seed_xyz 进行最远点采样得到的索引
#             # 将 cur_seed_xyz 的维度从 (1, 3, M) 转置为 (1, M, 3) 后得到的结果。
#             # 这里使用 cur_seed_xyz.transpose(1, 2) 进行转置操作，
#             # 然后使用 contiguous() 函数保证内存的连续性。
#             cur_seed_xyz_flipped = cur_seed_xyz.transpose(1, 2).contiguous()  # 1*3*M
#             # cur_seed_xyz_flipped 中选取最远点采样得到的索引对应的坐标点
#             cur_seed_xyz = gather_operation(cur_seed_xyz_flipped, fps_idxs).transpose(1, 2).squeeze(0).contiguous() # M*3
#             # 从 cur_feat_flipped 中选取最远点采样得到的索引对应的特征点
#             cur_feat_flipped = cur_feat.unsqueeze(0).transpose(1, 2).contiguous()  # 1*feat_dim*M
#             cur_feat = gather_operation(cur_feat_flipped, fps_idxs).squeeze(0).contiguous() # feat_dim*M
            
#             # 将采样得到的特征 cur_feat 和坐标 cur_seed_xyz 添加到对应的列表 seed_features_graspable 和 seed_xyz_graspable 中
#             seed_features_graspable.append(cur_feat)
#             seed_xyz_graspable.append(cur_seed_xyz)
#             fps_idxs = fps_idxs.squeeze()

#             fps_idxs = fps_idxs.long()
#             idxa_bat_fps = idxa_bat[fps_idxs]
#             idx_640_360_bat_fps = idx_640_360_bat[fps_idxs]
#             idxa_bat_fps_graspable.append(idxa_bat_fps)
#             idx_640_360_fps_graspable.append(idx_640_360_bat_fps)


#         # 通过调用 torch.stack(seed_xyz_graspable, 0) 将列表中的张量沿着新创建的维度（索引为 0）进行堆叠，
#         # 得到形状为 (B, M, 3) 的张量，其中 B 是列表中元素的数量，即批次大小
#         seed_xyz_graspable = torch.stack(seed_xyz_graspable, 0)  # B*M*3
#         # 将列表中的张量沿着新创建的维度进行堆叠，得到形状为 (B, feat_dim, Ns) 的张量，其中 B 是列表中元素的数量，即批次大小。
#         seed_features_graspable = torch.stack(seed_features_graspable)  # B*feat_dim*M
#         idxa_bat_fps_graspable = torch.stack(idxa_bat_fps_graspable)
#         idx_640_360_fps_graspable = torch.stack(idx_640_360_fps_graspable)

#         rgb_resize = rgb_resize.permute(0, 3, 1, 2)
#         rgb_resize = rgb_resize.float()
#         self.rgbd_backbone = self.rgbd_backbone.to('cuda')
#         rgbd_features = self.rgbd_backbone(rgb_resize)


#         row_640_360_bat_fps_graspable = idx_640_360_fps_graspable // 640
#         col_640_360_bat_fps_graspable = idx_640_360_fps_graspable % 640
#         local_patches_rgb = generate_loacl_region(rgbd_features,row_640_360_bat_fps_graspable,col_640_360_bat_fps_graspable)
#         local_patches_rgb = torch.stack(local_patches_rgb)
#         local_patches_rgb1 = local_patches_rgb.to(device)

#         # 将处理后的可抓取点坐标 seed_xyz_graspable 存储在 end_points 字典中，使用键 'xyz_graspable'
#         end_points['xyz_graspable'] = seed_xyz_graspable
#         # graspable_num_batch 是一个表示每个批次中可抓取点数量的标量张量。
#         # 通过除以批次大小 B，计算出每个批次中平均的可抓取点数量，并将其存储在 end_points 字典中，使用键 'graspable_count_stage1'
#         end_points['graspable_count_stage1'] = graspable_num_batch / B
        
#         """self.rotation 是一个旋转模型，它接收可抓取点的特征和坐标，并返回预测的抓取方向以及其他相关信息"""
#         end_points, res_feat = self.rotation(seed_features_graspable, end_points) # 将选取的可抓取点的特征和坐标输入到rotation模型中进行抓取方向预测
#         # res_feat 是旋转模型输出的特征增量。将 res_feat 加到 seed_features_graspable 上，更新可抓取点的特征
#         seed_features_graspable = seed_features_graspable + res_feat 

#         adaptive_weight_point = torch.sigmoid(self.mlp_point(seed_features_graspable))
#         adaptive_weight_image = torch.sigmoid(self.mlp_rgb(local_patches_rgb1))

#         fused_features = adaptive_weight_point * seed_features_graspable + adaptive_weight_image * local_patches_rgb1

#         # 如果是训练阶段，还会进行一些处理操作，如处理抓取标签和匹配抓取视图和标签
#         if self.is_training:
#             end_points = process_grasp_labels(end_points) # 生成抓取标签，这个是根据得到的抓取点，抓取角度，抓取在旋转矩阵，抓取视角生成的
#             grasp_top_views_rot, end_points = match_grasp_view_and_label(end_points) # 匹配抓取视图和标签
#         else:
#             grasp_top_views_rot = end_points['grasp_top_view_rot']

#         # 将选取的可抓取点的特征和坐标以及抓取方向作为输入，通过crop模型裁剪出与抓取相关的局部区域
#         #  ################前拼接
#         group_features = self.crop(seed_xyz_graspable.contiguous(), fused_features.contiguous(), grasp_top_views_rot)

#         # PAconv
#         paconv_net = PAConv(args)
#         paconv_net = paconv_net.to('cuda')
#         seed_xyz_graspable_huan = seed_xyz_graspable.permute(0, 2, 1)
#         PA_seed_features_graspable = paconv_net(seed_xyz_graspable_huan)

#         new_features_1024_all  =  torch.cat((PA_seed_features_graspable, group_features), dim=1)
#         new_features_1024_all = self.fuse_multi_scale(new_features_1024_all) 
#         end_points = self.swad(new_features_1024_all, end_points)

#         return end_points




class GraspNet(nn.Module):
    def __init__(self, cylinder_radius=0.05, seed_feat_dim=512, is_training=True):
        super().__init__()
        self.is_training = is_training
        self.seed_feature_dim = seed_feat_dim
        self.num_depth = NUM_DEPTH
        self.num_angle = NUM_ANGLE
        self.M_points = M_POINT
        self.num_view = NUM_VIEW
     
        # 名为backbone的MinkUNet14D模型，用于提取点云的特征
        # D=3表示数据的输入是三维的
        self.backbone = MinkUNet14D(in_channels=3, out_channels=self.seed_feature_dim, D=3)
        # 名为graspable的GraspableNet模型，用于预测点云中每个点的可抓取性
        self.graspable = GraspableNet(seed_feature_dim=self.seed_feature_dim)
        # ApproachNet模型，用于预测点云中每个点的抓取方向 每个点预测300个视角，预测得分
        self.rotation = ApproachNet(self.num_view, seed_feature_dim=self.seed_feature_dim, is_training=self.is_training)
        # CloudCrop模型，用于从点云中裁剪出与抓取相关的局部区域
        self.crop = CloudCrop(nsample=16, cylinder_radius=cylinder_radius, seed_feature_dim=self.seed_feature_dim)
        # SWADNet模型，用于预测抓取的得分和宽度。
        self.swad = SWADNet(num_angle=self.num_angle, num_depth=self.num_depth)

        # self.fuse_multi_scale = nn.Conv1d(256 + 256 + 512, 256, 1)

        self.fuse_multi_scale = nn.Conv1d(256 + 256, 256, 1)

        self.rgbd_feature_dim = 512  #1  2 3
        self.rgbd_backbone = SeResUNet(3, out_channels=self.rgbd_feature_dim)
        self.mlp_point = MLP(1024, 1024)
        self.mlp_rgb = MLP(1024, 1024)

    def forward(self, end_points):
        # 获取输入的点云坐标和特征，并通过backbone模型提取点云的特征
        seed_xyz = end_points['point_clouds']  # use all sampled point cloud, B*Ns*3
        idxa = end_points['idx']
        rgb = end_points['rgb']
        rgb_resize = end_points['rgb_resize']
        depth = end_points['depth']

        idx_640_360 = end_points['point_clouds_idx_in_rgbd']

        B, point_num, _ = seed_xyz.shape  # batch _size
        # point-wise features
        coordinates_batch = end_points['coors']  # 从end_points字典中获取coors坐标数据N*3。
        features_batch = end_points['feats']     # 从end_points字典中获取feats特征数据B*C。
        mink_input = ME.SparseTensor(features_batch, coordinates=coordinates_batch)# 创建一个稀疏张量mink_input ,用上述提到的 coors坐标数据 和 feats特征数据。 
        """[B*Ns*(C+3) --> B*Ns*512]"""
        seed_features = self.backbone(mink_input).F # mink_input [B*Ns*（C+3）--> B*Ns*512] 输入到backbone模型中，并获取输出的特征数据seed_features
        # 根据end_points['quantize2original']对seed_features进行索引，
        # 以将其转换为原始点云坐标的顺序。然后，将其形状重新调整为(B, point_num, -1)，其中B表示批次大小，point_num表示点的数量
        # 通过transpose(1, 2)操作将seed_features的维度转置，将点的数量维度放在第二个维度上
        seed_features = seed_features[end_points['quantize2original']].view(B, point_num, -1).transpose(1, 2)
        
        # 将提取到的特征输入到graspable模型中进行可抓取性预测
        # 根据可抓取性预测结果和阈值，生成可抓取性掩码。
        end_points = self.graspable(seed_features, end_points)# 将 seed_features 与 end_points 一起作为参数传递到 self.graspable 函数中。
        seed_features_flipped = seed_features.transpose(1, 2)  # B*Ns*feat_dim --> B*Ns*512
        # # 从 end_points 中获取 objectness_score 和 graspness_score。
        objectness_score = end_points['objectness_score']
        graspness_score = end_points['graspness_score'].squeeze(1)
        objectness_pred = torch.argmax(objectness_score, 1)    # objectness_score 的第一个维度上执行最大值索引,得到是物体还是是背景
        objectness_mask = (objectness_pred == 1)               # 进行mask
        graspness_mask = graspness_score > GRASPNESS_THRESHOLD # 抓取得分大于阈值
        graspable_mask = objectness_mask & graspness_mask      # 指示哪些抓取点既被预测为属于物体，又具有足够的可抓取程度
        
        """FPS 从中取出1024给点, 随机取的"""
        # 根据可抓取性掩码对点云的特征进行筛选，得到可抓取点的特征和坐标。
        # 通过最远点采样（furthest point sampling）选取一定数量（M_POINT）的采样点，并将其特征和坐标保存下来
        seed_features_graspable = []
        seed_xyz_graspable = []
        idxa_bat_fps_graspable = []
        idx_640_360_fps_graspable = []
        graspable_num_batch = 0.
        

        for i in range(B):
            
            cur_mask = graspable_mask[i]        # 从graspable_mask中获取当前样本的掩码(cur_mask)，用于筛选出可抓取的点

            graspable_num_batch += cur_mask.sum() # 对当前样本中可抓取的点进行统计，计算可抓取点的数量，并将其累加到graspable_num_batch变量中
            cur_feat = seed_features_flipped[i][cur_mask]  # Ns*feat_dim 从seed_features_flipped中获取当前样本的特征(cur_feat)，并根据掩码进行筛选，得到可抓取点对应的特征
            cur_seed_xyz = seed_xyz[i][cur_mask]  # *Ns3  从seed_xyz中获取当前样本的坐标(cur_seed_xyz)，并根据掩码进行筛选，得到可抓取点的坐标
            idxa_bat = idxa[i][cur_mask]
            idx_640_360_bat = idx_640_360[i][cur_mask]

            cur_seed_xyz = cur_seed_xyz.unsqueeze(0) # cur_seed_xyz表示当前的种子点坐标 1*M*3 增加一个维度
            fps_idxs = furthest_point_sample(cur_seed_xyz, self.M_points) # fps_idxs 是通过调用函数 furthest_point_sample 对 cur_seed_xyz 进行最远点采样得到的索引
            # 将 cur_seed_xyz 的维度从 (1, 3, M) 转置为 (1, M, 3) 后得到的结果。
            # 这里使用 cur_seed_xyz.transpose(1, 2) 进行转置操作，
            # 然后使用 contiguous() 函数保证内存的连续性。
            cur_seed_xyz_flipped = cur_seed_xyz.transpose(1, 2).contiguous()  # 1*3*M
            # cur_seed_xyz_flipped 中选取最远点采样得到的索引对应的坐标点
            cur_seed_xyz = gather_operation(cur_seed_xyz_flipped, fps_idxs).transpose(1, 2).squeeze(0).contiguous() # M*3
            # 从 cur_feat_flipped 中选取最远点采样得到的索引对应的特征点
            cur_feat_flipped = cur_feat.unsqueeze(0).transpose(1, 2).contiguous()  # 1*feat_dim*M
            cur_feat = gather_operation(cur_feat_flipped, fps_idxs).squeeze(0).contiguous() # feat_dim*M
            
            # 将采样得到的特征 cur_feat 和坐标 cur_seed_xyz 添加到对应的列表 seed_features_graspable 和 seed_xyz_graspable 中
            seed_features_graspable.append(cur_feat)
            seed_xyz_graspable.append(cur_seed_xyz)
            fps_idxs = fps_idxs.squeeze()

            fps_idxs = fps_idxs.long()
            idxa_bat_fps = idxa_bat[fps_idxs]
            idx_640_360_bat_fps = idx_640_360_bat[fps_idxs]
            idxa_bat_fps_graspable.append(idxa_bat_fps)
            idx_640_360_fps_graspable.append(idx_640_360_bat_fps)


        # 通过调用 torch.stack(seed_xyz_graspable, 0) 将列表中的张量沿着新创建的维度（索引为 0）进行堆叠，
        # 得到形状为 (B, M, 3) 的张量，其中 B 是列表中元素的数量，即批次大小
        seed_xyz_graspable = torch.stack(seed_xyz_graspable, 0)  # B*M*3
        # 将列表中的张量沿着新创建的维度进行堆叠，得到形状为 (B, feat_dim, Ns) 的张量，其中 B 是列表中元素的数量，即批次大小。
        seed_features_graspable = torch.stack(seed_features_graspable)  # B*feat_dim*M
        idxa_bat_fps_graspable = torch.stack(idxa_bat_fps_graspable)
        idx_640_360_fps_graspable = torch.stack(idx_640_360_fps_graspable)

        rgb_resize = rgb_resize.permute(0, 3, 1, 2)
        rgb_resize = rgb_resize.float()
        self.rgbd_backbone = self.rgbd_backbone.to('cuda')
        rgbd_features = self.rgbd_backbone(rgb_resize)


        row_640_360_bat_fps_graspable = idx_640_360_fps_graspable // 640
        col_640_360_bat_fps_graspable = idx_640_360_fps_graspable % 640
        local_patches_rgb = generate_loacl_region(rgbd_features,row_640_360_bat_fps_graspable,col_640_360_bat_fps_graspable)
        local_patches_rgb = torch.stack(local_patches_rgb)
        local_patches_rgb1 = local_patches_rgb.to(device)

        # 将处理后的可抓取点坐标 seed_xyz_graspable 存储在 end_points 字典中，使用键 'xyz_graspable'
        end_points['xyz_graspable'] = seed_xyz_graspable
        # graspable_num_batch 是一个表示每个批次中可抓取点数量的标量张量。
        # 通过除以批次大小 B，计算出每个批次中平均的可抓取点数量，并将其存储在 end_points 字典中，使用键 'graspable_count_stage1'
        end_points['graspable_count_stage1'] = graspable_num_batch / B
        
        """self.rotation 是一个旋转模型，它接收可抓取点的特征和坐标，并返回预测的抓取方向以及其他相关信息"""
        end_points, res_feat = self.rotation(seed_features_graspable, end_points) # 将选取的可抓取点的特征和坐标输入到rotation模型中进行抓取方向预测
        # res_feat 是旋转模型输出的特征增量。将 res_feat 加到 seed_features_graspable 上，更新可抓取点的特征
        seed_features_graspable = seed_features_graspable + res_feat 

        adaptive_weight_point = torch.sigmoid(self.mlp_point(seed_features_graspable))
        adaptive_weight_image = torch.sigmoid(self.mlp_rgb(local_patches_rgb1))

        print(adaptive_weight_point.shape)

        adaptive_weight_point = (adaptive_weight_point + 1).clamp(min=0.5, max=1.0)
        adaptive_weight_image = (adaptive_weight_image - 1).clamp(min=0.0, max=0.5)

        fused_features = (adaptive_weight_point * seed_features_graspable + adaptive_weight_image * local_patches_rgb1) / (adaptive_weight_point + adaptive_weight_image)


        # 如果是训练阶段，还会进行一些处理操作，如处理抓取标签和匹配抓取视图和标签
        if self.is_training:
            end_points = process_grasp_labels(end_points) # 生成抓取标签，这个是根据得到的抓取点，抓取角度，抓取在旋转矩阵，抓取视角生成的
            grasp_top_views_rot, end_points = match_grasp_view_and_label(end_points) # 匹配抓取视图和标签
        else:
            grasp_top_views_rot = end_points['grasp_top_view_rot']

        # 将选取的可抓取点的特征和坐标以及抓取方向作为输入，通过crop模型裁剪出与抓取相关的局部区域
        #  ################前拼接
        group_features = self.crop(seed_xyz_graspable.contiguous(), fused_features.contiguous(), grasp_top_views_rot)

        # PAconv
        # paconv_net = PAConv(args)
        # paconv_net = paconv_net.to('cuda')
        # seed_xyz_graspable_huan = seed_xyz_graspable.permute(0, 2, 1)
        # PA_seed_features_graspable = paconv_net(seed_xyz_graspable_huan)

        # new_features_1024_all  =  torch.cat((PA_seed_features_graspable, group_features, fused_features), dim=1)
        # new_features_1024_all = self.fuse_multi_scale(new_features_1024_all) 
        # end_points = self.swad(new_features_1024_all, end_points)

        paconv_net = PAConv(args)
        paconv_net = paconv_net.to('cuda')
        seed_xyz_graspable_huan = seed_xyz_graspable.permute(0, 2, 1)
        PA_seed_features_graspable = paconv_net(seed_xyz_graspable_huan)

        new_features_1024_all  =  torch.cat((PA_seed_features_graspable, group_features), dim=1)
        new_features_1024_all = self.fuse_multi_scale(new_features_1024_all) 
        end_points = self.swad(new_features_1024_all, end_points)


        return end_points

def pred_decode(end_points):
    # 首先获取批量大小（batch_size），然后创建一个空列表grasp_preds用于存储解码后的抓取预测结果。
    batch_size = len(end_points['point_clouds'])
    grasp_preds = []
    for i in range(batch_size):
        # 对于每个样本，依次进行解码操作。首先获取抓取中心点（grasp_center）的预测结果，并将其转换为浮点型
        grasp_center = end_points['xyz_graspable'][i].float()
        # 获取抓取得分（grasp_score）的预测结果，并进行相应的处理
        grasp_score = end_points['grasp_score_pred'][i].float()
        grasp_score = grasp_score.view(M_POINT, NUM_ANGLE*NUM_DEPTH)
        grasp_score, grasp_score_inds = torch.max(grasp_score, -1)  # [M_POINT]
        grasp_score = grasp_score.view(-1, 1)
        # 获取抓取宽度（grasp_width）的预测结果，并进行相应的处理
        grasp_angle = (grasp_score_inds // NUM_DEPTH) * np.pi / 12
        grasp_depth = (grasp_score_inds % NUM_DEPTH + 1) * 0.01
        grasp_depth = grasp_depth.view(-1, 1)
        grasp_width = 1.2 * end_points['grasp_width_pred'][i] / 10.
        grasp_width = grasp_width.view(M_POINT, NUM_ANGLE*NUM_DEPTH)
        grasp_width = torch.gather(grasp_width, 1, grasp_score_inds.view(-1, 1))
        grasp_width = torch.clamp(grasp_width, min=0., max=GRASP_MAX_WIDTH)
        # 获取抓取旋转矩阵（grasp_rot）的预测结果，并进行相应的处理
        approaching = -end_points['grasp_top_view_xyz'][i].float()
        grasp_rot = batch_viewpoint_params_to_matrix(approaching, grasp_angle)
        grasp_rot = grasp_rot.view(M_POINT, 9)

        # merge preds 根据预测的抓取结果，组合得到完整的抓取预测结果
        grasp_height = 0.02 * torch.ones_like(grasp_score)
        obj_ids = -1 * torch.ones_like(grasp_score)
        grasp_preds.append(
            torch.cat([grasp_score, grasp_width, grasp_height, grasp_depth, grasp_rot, grasp_center, obj_ids], axis=-1))
    return grasp_preds

