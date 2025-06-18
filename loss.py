import torch.nn as nn
import torch
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

number_list_simple_two = [
    [0],
    [1, 2, 3, 4, 5],
    [26, 27, 28, 29, 30],
    [31, 32, 33, 34, 35],
    [56, 57, 58, 59, 60],
    [61, 62, 63, 64, 65],
    [86, 87, 88, 89, 90],
    [91, 92, 93, 94, 95],
    [116, 117, 118, 119, 120],
    [121, 122, 123, 124, 125],
    [146, 147, 148, 149, 150],
    [151, 152, 153, 154, 155],
    [176, 177, 178, 179, 180],
    [181, 182, 183, 184, 185],
    [206, 207, 208, 209, 210],
    [211, 212, 213, 214, 215],
    [236, 237, 238, 239, 240],
    [241, 242, 243, 244, 245]
]
number_list_simple_one = [
    0, 
    1, 2, 3, 4, 5,
    26, 27, 28, 29, 30,
    31, 32, 33, 34, 35,
    56, 57, 58, 59, 60,
    61, 62, 63, 64, 65,
    86, 87, 88, 89, 90,
    91, 92, 93, 94, 95,
    116, 117, 118, 119, 120,
    121, 122, 123, 124, 125,
    146, 147, 148, 149, 150,
    151, 152, 153, 154, 155,
    176, 177, 178, 179, 180,
    181, 182, 183, 184, 185,
    206, 207, 208, 209, 210,
    211, 212, 213, 214, 215,
    236, 237, 238, 239, 240,
    241, 242, 243, 244, 245
]
number_list_difficult_one = [
    6, 7, 8, 9, 10,
    21, 22, 23, 24, 25,
    36, 37, 38, 39, 40,
    51, 52, 53, 54, 55,
    66, 67, 68, 69, 70,
    81, 82, 83, 84, 85,
    96, 97, 98, 99, 100,
    111, 112, 113, 114, 115,
    126, 127, 128, 129, 130,
    141, 142, 143, 144, 145,
    156, 157, 158, 159, 160,
    171, 172, 173, 174, 175,
    186, 187, 188, 189, 190,
    201, 202, 203, 204, 205,
    216, 217, 218, 219, 220,
    231, 232, 233, 234, 235,
    246, 247, 248, 249, 250
]
number_list_infernal_one = [
    11, 12, 13, 14, 15,
    16, 17, 18, 19, 20,
    41, 42, 43, 44, 45,
    46, 47, 48, 49, 50,
    71, 72, 73, 74, 75,
    76, 77, 78, 79, 80,
    101, 102, 103, 104, 105,
    106, 107, 108, 109, 110,
    131, 132, 133, 134, 135,
    136, 137, 138, 139, 140,
    161, 162, 163, 164, 165,
    166, 167, 168, 169, 170,
    191, 192, 193, 194, 195,
    196, 197, 198, 199, 200,
    221, 222, 223, 224, 225,
    226, 227, 228, 229, 230,
    251, 252, 253, 254, 255
]
number_list_difficult_two = [
    [6, 7, 8, 9, 10],
    [21, 22, 23, 24, 25],
    [36, 37, 38, 39, 40],
    [51, 52, 53, 54, 55],
    [66, 67, 68, 69, 70],
    [81, 82, 83, 84, 85],
    [96, 97, 98, 99, 100],
    [111, 112, 113, 114, 115],
    [126, 127, 128, 129, 130],
    [141, 142, 143, 144, 145],
    [156, 157, 158, 159, 160],
    [171, 172, 173, 174, 175],
    [186, 187, 188, 189, 190],
    [201, 202, 203, 204, 205],
    [216, 217, 218, 219, 220],
    [231, 232, 233, 234, 235],
    [246, 247, 248, 249, 250]
]
number_list_infernal_two = [
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [41, 42, 43, 44, 45],
    [46, 47, 48, 49, 50],
    [71, 72, 73, 74, 75],
    [76, 77, 78, 79, 80],
    [101, 102, 103, 104, 105],
    [106, 107, 108, 109, 110],
    [131, 132, 133, 134, 135],
    [136, 137, 138, 139, 140],
    [161, 162, 163, 164, 165],
    [166, 167, 168, 169, 170],
    [191, 192, 193, 194, 195],
    [196, 197, 198, 199, 200],
    [221, 222, 223, 224, 225],
    [226, 227, 228, 229, 230],
    [251, 252, 253, 254, 255]
]
def find_position_in_2d_array(target, array_2d):
    for row_index, row in enumerate(array_2d):
        for col_index, element in enumerate(row):
            if element == target:
                return row_index, col_index  # 返回行索引和列索引
    return None  # 如果没有找到，返回None

            
def get_viewweigh(end_points):
    final_weigh = torch.tensor([], device=device)  # 初始化为一个空的一维张量
    view = end_points['view_nums']

    for i in range(len(view)):
        if view[i] in number_list_simple_one:
            position = find_position_in_2d_array(view[i], number_list_simple_two) 
            if position[0] == 0 & position[1] == 0:
                scaled_weights_tensor = torch.tensor([0.75], device=device)  # 一维张量
                final_weigh = torch.cat((final_weigh, scaled_weights_tensor), dim=0)
            else:
                # scaled_weights_tensor = torch.tensor([0.6], device=device)
                scaled_weights = 0.5 * np.exp(0.322 * position[1] - 1.25) + 0.5
                scaled_weights_tensor = torch.tensor([scaled_weights], device=device)  # 注意这里使用方括号创建一维张量
                final_weigh = torch.cat((final_weigh, scaled_weights_tensor), dim=0)

        if view[i] in number_list_difficult_one:
            position = find_position_in_2d_array(view[i], number_list_difficult_two)     
            if position[0] == 0 & position[1] == 0:
                scaled_weights_tensor = torch.tensor([1.1], device=device)  # 一维张量
                final_weigh = torch.cat((final_weigh, scaled_weights_tensor), dim=0)
            else:
                # scaled_weights_tensor = torch.tensor([0.6], device=device)
                scaled_weights = 0.5 * np.exp(0.111 * position[1] +0.354) + 0.5
                scaled_weights_tensor = torch.tensor([scaled_weights], device=device)  # 注意这里使用方括号创建一维张量
                final_weigh = torch.cat((final_weigh, scaled_weights_tensor), dim=0)
    
        if view[i] in number_list_infernal_one:
            position = find_position_in_2d_array(view[i], number_list_infernal_two)    
            if position[0] == 0 & position[1] == 0:
                scaled_weights_tensor = torch.tensor([1.7], device=device)  # 一维张量
                final_weigh = torch.cat((final_weigh, scaled_weights_tensor), dim=0)
            else:
                # scaled_weights_tensor = torch.tensor([0.6], device=device)
                scaled_weights = 0.5 * np.exp(0.067 * position[1] + 0.96) + 0.5
                scaled_weights_tensor = torch.tensor([scaled_weights], device=device)  # 注意这里使用方括号创建一维张量
                final_weigh = torch.cat((final_weigh, scaled_weights_tensor), dim=0)
        
    final_weigh_n = torch.empty((len(final_weigh), 1024), device=device)

    # 遍历 final_weigh 中的每个元素，并创建填充了该元素值的 1024 个一维张量
    for i, value in enumerate(final_weigh):
        final_weigh_n[i] = torch.full((1024,), value, dtype=torch.float32, device=device)
    return final_weigh_n            
            

def get_loss(end_points):
    weigh = get_viewweigh(end_points)
    objectness_loss, end_points = compute_objectness_loss(end_points)
    graspness_loss, end_points = compute_graspness_loss(end_points)

    view_loss, end_points = compute_view_graspness_loss(end_points,weigh)   # 抓取方向
    score_loss, end_points = compute_score_loss(end_points,weigh)           # 抓取最终得分
    width_loss, end_points = compute_width_loss(end_points,weigh)           # 抓取宽度

    loss = objectness_loss + 10 * graspness_loss + 100 * view_loss + 10 * score_loss + 10 * width_loss
    end_points['loss/overall_loss'] = loss
    return loss, end_points


def compute_objectness_loss(end_points):
    criterion = nn.CrossEntropyLoss(reduction='mean')
    objectness_score = end_points['objectness_score']
    objectness_label = end_points['objectness_label']
    loss = criterion(objectness_score, objectness_label)
    end_points['loss/stage1_objectness_loss'] = loss

    objectness_pred = torch.argmax(objectness_score, 1)
    end_points['stage1_objectness_acc'] = (objectness_pred == objectness_label.long()).float().mean()
    end_points['stage1_objectness_prec'] = (objectness_pred == objectness_label.long())[
        objectness_pred == 1].float().mean()
    end_points['stage1_objectness_recall'] = (objectness_pred == objectness_label.long())[
        objectness_label == 1].float().mean()
    return loss, end_points


def compute_graspness_loss(end_points):
    criterion = nn.SmoothL1Loss(reduction='none')
    graspness_score = end_points['graspness_score'].squeeze(1)
    graspness_label = end_points['graspness_label'].squeeze(-1)
    loss_mask = end_points['objectness_label'].bool()
    loss = criterion(graspness_score, graspness_label)
    loss = loss[loss_mask]
    loss = loss.mean()
    
    graspness_score_c = graspness_score.detach().clone()[loss_mask]
    graspness_label_c = graspness_label.detach().clone()[loss_mask]
    graspness_score_c = torch.clamp(graspness_score_c, 0., 0.99)
    graspness_label_c = torch.clamp(graspness_label_c, 0., 0.99)
    rank_error = (torch.abs(torch.trunc(graspness_score_c * 20) - torch.trunc(graspness_label_c * 20)) / 20.).mean()
    end_points['stage1_graspness_acc_rank_error'] = rank_error

    end_points['loss/stage1_graspness_loss'] = loss
    return loss, end_points


def compute_view_graspness_loss(end_points,final_weigh): # 抓取读视角损失
    criterion = nn.SmoothL1Loss(reduction='mean')
    view_score = end_points['view_score']
    view_label = end_points['batch_grasp_view_graspness']
    loss = criterion(view_score, view_label)

    criterion = nn.SmoothL1Loss(reduction='none')
    view_score = end_points['view_score']
    view_label = end_points['batch_grasp_view_graspness']
    loss = criterion(view_score, view_label)

    weighted_loss = torch.mul(loss, final_weigh.unsqueeze(-1).expand_as(loss))
    weighted_mean_loss = torch.mean(weighted_loss)
    end_points['loss/stage2_view_loss'] = weighted_mean_loss
    return weighted_mean_loss, end_points

 
def compute_score_loss(end_points,final_weigh): # 最终得分
    criterion = nn.SmoothL1Loss(reduction='none')
    grasp_score_pred = end_points['grasp_score_pred']
    grasp_score_label = end_points['batch_grasp_score']
    loss = criterion(grasp_score_pred, grasp_score_label)

    weighted_loss = torch.mul(loss, final_weigh.unsqueeze(-1).unsqueeze(-1).expand_as(loss))
    weighted_mean_loss = torch.mean(weighted_loss)
    end_points['loss/stage3_score_loss'] = weighted_mean_loss
    return weighted_mean_loss, end_points


def compute_width_loss(end_points,final_weigh): # 爪子宽度 
    criterion = nn.SmoothL1Loss(reduction='none')
    grasp_width_pred = end_points['grasp_width_pred']
    grasp_width_label = end_points['batch_grasp_width'] * 10
    loss = criterion(grasp_width_pred, grasp_width_label)

    weighted_loss = torch.mul(loss, final_weigh.unsqueeze(-1).unsqueeze(-1).expand_as(loss))

    grasp_score_label = end_points['batch_grasp_score']
    loss_mask = grasp_score_label > 0
    weighted_loss = weighted_loss[loss_mask].mean()
    
    end_points['loss/stage3_width_loss'] = weighted_loss
    return weighted_loss, end_points