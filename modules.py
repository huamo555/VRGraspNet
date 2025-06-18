import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import pytorch_utils as pt_utils
from pointnet2_utils import CylinderQueryAndGroup
from loss_utils import generate_grasp_views, batch_viewpoint_params_to_matrix

import torch
import torch.nn.functional as F
from obj_cls.model.PointNet_PAConv import PAConv  # 假设你的PAConv类在your_module模块中
import json




# 输入的维度是seed_feature_dim(512)，表示输入的种子特征的维度 
# seed_features 是输入的种子特征，end_points 是一个字典，用于存储中间结果
class GraspableNet(nn.Module):
    def __init__(self, seed_feature_dim):
        super().__init__()
        self.in_dim = seed_feature_dim
        self.conv_graspable = nn.Conv1d(self.in_dim, 3, 1)

    def forward(self, seed_features, end_points):
        graspable_score = self.conv_graspable(seed_features)  # (B, 3, num_seed)
        end_points['objectness_score'] = graspable_score[:, :2] # 将 graspable_score 的前两个通道作为 'objectness_score'
        end_points['graspness_score'] = graspable_score[:, 2]   # 第三个通道作为 'graspness_score'
        return end_points

"""预测抓取的方向"""
class ApproachNet(nn.Module):
    def __init__(self, num_view, seed_feature_dim, is_training=True):
        super().__init__()
        self.num_view = num_view
        self.in_dim = seed_feature_dim
        self.is_training = is_training
        # 两个卷积层self.conv1和self.conv2
        self.conv1 = nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv2 = nn.Conv1d(self.in_dim, self.num_view, 1) # num_view-->300 300个视角

    def forward(self, seed_features, end_points):
        # seed_features表示输入的种子特征，形状为(B, C, num_seed)，其中B是批次大小，C是特征通道数，num_seed是种子特征的数量
        B, _, num_seed = seed_features.size()
        # 两次卷积,得到300个视角下各个视角的得分，最终的结果存储在end_points字典中的'view_score'键下
        res_features = F.relu(self.conv1(seed_features), inplace=True)
        features = self.conv2(res_features)
        view_score = features.transpose(1, 2).contiguous() # (B, num_seed, num_view)
        end_points['view_score'] = view_score
        
        """对view_score进行归一化处理,将其范围缩放到0到1之间"""
        if self.is_training:
            # normalize view graspness score to 0~1
            view_score_ = view_score.clone().detach() # 创建一个view_score_的副本，并使用clone().detach()进行分离，以避免梯度传播
            view_score_max, _ = torch.max(view_score_, dim=2) # 计算view_score_在第2维上的最大值保存在view_score_max中
            view_score_min, _ = torch.min(view_score_, dim=2) # 计算view_score_在第2维上的最小值保存在view_score_min中
            view_score_max = view_score_max.unsqueeze(-1).expand(-1, -1, self.num_view) # 将view_score_max扩展为与view_score_相同的形状
            view_score_min = view_score_min.unsqueeze(-1).expand(-1, -1, self.num_view) # 将view_score_min扩展为与view_score_相同的形状
            # 对view_score_进行归一化计算：(view_score_ - view_score_min) / (view_score_max - view_score_min + 1e-8)
            view_score_ = (view_score_ - view_score_min) / (view_score_max - view_score_min + 1e-8)

            top_view_inds = [] # 创建一个空列表top_view_inds用于存储每个样本的选择的视角索引
            for i in range(B): # 对于每个样本，从归一化后的view_score_中根据概率进行采样
                # 对于每个样本，使用torch.multinomial函数根据每个视角的概率进行采样，并选择一个样本
                top_view_inds_batch = torch.multinomial(view_score_[i], 1, replacement=False)
                top_view_inds.append(top_view_inds_batch) # 将采样得到的索引添加到top_view_inds列表中
            # 将采样得到的索引添加到top_view_inds列表中
            top_view_inds = torch.stack(top_view_inds, dim=0).squeeze(-1)  # B, num_seed
        else: 
            """这段代码根据最佳视角索引,生成了每个种子特征对应的视角坐标和旋转矩阵,并将其存储在end_points字典中"""
            _, top_view_inds = torch.max(view_score, dim=2)  # (B, num_seed) torch.max函数找到每个样本中视角得分最高的索引，表示每个种子特征对应的最佳视角索引
            
            # 从预定义的视角模板中根据top_view_inds_的索引选取对应的视角。视角模板的形状为(1, 1, num_view, 3)，
            # 通过expand()将其扩展为(B, num_seed, num_view, 3)的形状，以便与top_view_inds_的形状匹配。
            top_view_inds_ = top_view_inds.view(B, num_seed, 1, 1).expand(-1, -1, -1, 3).contiguous()
            # 通过torch.gather函数根据top_view_inds_的索引从视角模板中选取对应的视角，得到形状为(B, num_seed, 3)的vp_xyz张量，表示每个种子特征对应的视角坐标
            template_views = generate_grasp_views(self.num_view).to(features.device)  # (num_view, 3)
            template_views = template_views.view(1, 1, self.num_view, 3).expand(B, num_seed, -1, -1).contiguous()
            # 对vp_xyz进行形状变换，将其变为(B*num_seed, 3)的形状，并创建一个形状相同的全零张量batch_angle
            vp_xyz = torch.gather(template_views, 2, top_view_inds_).squeeze(2)  # (B, num_seed, 3)
            vp_xyz_ = vp_xyz.view(-1, 3)
            batch_angle = torch.zeros(vp_xyz_.size(0), dtype=vp_xyz.dtype, device=vp_xyz.device)
            # 通过batch_viewpoint_params_to_matrix函数将-vp_xyz_和batch_angle作为参数，
            # 生成视角旋转矩阵vp_rot。vp_rot的形状为(B, num_seed, 3, 3)，表示每个种子特征对应的视角旋转矩阵
            vp_rot = batch_viewpoint_params_to_matrix(-vp_xyz_, batch_angle).view(B, num_seed, 3, 3)
            end_points['grasp_top_view_xyz'] = vp_xyz
            end_points['grasp_top_view_rot'] = vp_rot

        end_points['grasp_top_view_inds'] = top_view_inds
        return end_points, res_features

def knn(x,k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    
    idx = pairwise_distance.topk(k=k, dim=-1)[1]        #(batch_size, num_points, k)
    return idx
    

def get_graph_feature(x, k=20, idx=None, dim9 = False):
    # x: (batch_size, 3, num_points)
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')
    
    idx_base = torch.arange(0, batch_size, device=device).view(-1,1,1)*num_points
    
    idx = idx + idx_base
    
    idx = idx.view(-1)
    
    _,num_dims,_ = x.size()
    
    x = x.transpose(2,1).contiguous()   # (batch_size, num_points, num_dims) --> (batch_size*num_points, num_dims)
    feature = x.view(batch_size*num_points, -1)[idx, :]     # KNN
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1,1,k,1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    
    return feature      # (batch_size, 2*num_dims, num_points, k)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



"""该模型用于对点云进行裁剪和特征提取"""
"""CloudCrop模型通过圆柱体查询和分组操作,对输入的点云进行裁剪并提取局部特征"""
class CloudCrop(nn.Module):
    def __init__(self, nsample, seed_feature_dim, cylinder_radius=0.05, hmin=-0.02, hmax=0.04):
        super().__init__()
        self.nsample = nsample
        self.in_dim = seed_feature_dim
        self.cylinder_radius = cylinder_radius
        mlps = [3 + self.in_dim, 256, 256]   # use xyz, so plus 3

        self.grouper = CylinderQueryAndGroup(radius=cylinder_radius, hmin=hmin, hmax=hmax, nsample=nsample,
                                             use_xyz=True, normalize_xyz=True)
        self.mlps = pt_utils.SharedMLP(mlps, bn=True)

    def forward(self, seed_xyz_graspable, seed_features_graspable, vp_rot):
        grouped_feature = self.grouper(seed_xyz_graspable, seed_xyz_graspable, vp_rot,
                                       seed_features_graspable)  # B*3 + feat_dim*M*K
        new_features = self.mlps(grouped_feature)  # (batch_size, mlps[-1], M, K)
        new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])  # (batch_size, mlps[-1], M, 1)
        new_features = new_features.squeeze(-1)   # (batch_size, mlps[-1], M)
        return new_features
    


    

"""SWADNet模型通过卷积层对输入特征进行处理和提取,预测抓取得分和宽度,并将预测结果保存在end_points字典中"""
class SWADNet(nn.Module):
    def __init__(self, num_angle, num_depth):
        super().__init__()
        self.num_angle = num_angle # num_angle（视角数量）
        self.num_depth = num_depth # num_depth（宽度数量）

        self.conv1 = nn.Conv1d(256, 256, 1)  # input feat dim need to be consistent with CloudCrop module
        self.conv_swad = nn.Conv1d(256, 2*num_angle*num_depth, 1)

    def forward(self, vp_features, end_points):
        B, _, num_seed = vp_features.size()
        # 首先获取输入的vp_features，它是经过CloudCrop模块裁剪后的局部区域的特征
        vp_features = F.relu(self.conv1(vp_features), inplace=True)
        # 通过一个1D卷积层conv1对vp_features进行特征提取，得到一个维度为256的特征表示
        vp_features = self.conv_swad(vp_features)
        # 通过另一个1D卷积层conv_swad对提取的特征进行抓取得分和宽度的预测
        vp_features = vp_features.view(B, 2, self.num_angle, self.num_depth, num_seed)
        # 将输出的特征进行维度变换和排列，得到最终的抓取得分和宽度的预测结果
        vp_features = vp_features.permute(0, 1, 4, 2, 3)

        # split prediction
        end_points['grasp_score_pred'] = vp_features[:, 0]  # B * num_seed * num angle * num_depth
        end_points['grasp_width_pred'] = vp_features[:, 1]
        # 将预测结果保存在end_points字典中，并返回该字典作为模型的输出。
        return end_points
