import os
import sys
import numpy as np
from datetime import datetime
import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))

from graspnet import GraspNet
from loss import get_loss
from loss_novel import get_loss_novel
from loss_novel_new import get_loss_novel_new
from graspnet_dataset import GraspNetDataset, minkowski_collate_fn, load_grasp_labels

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default=None, required=True)
parser.add_argument('--camera', default='kinect', help='Camera split [realsense/kinect]')
parser.add_argument('--checkpoint_path', help='Model checkpoint path', default=None)
parser.add_argument('--model_name', type=str, default=None)
parser.add_argument('--log_dir', default='/data2/gaoyuming/.cache/graspness_implementation-main/output')
parser.add_argument('--num_point', type=int, default=19998, help='Point Number [default: 20000]')
parser.add_argument('--seed_feat_dim', default=512, type=int, help='Point wise feature dim')
parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel Size to process point clouds ')
parser.add_argument('--max_epoch', type=int, default=35, help='Epoch to run [default: 18]')
parser.add_argument('--batch_size', type=int, default=2, help='Batch Size during training [default: 2]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--resume', action='store_true', default=False, help='Whether to resume from checkpoint')
cfgs = parser.parse_args()
# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
EPOCH_CNT = 0
# 判断cfgs.log_dir是否存在，如果不存在则创建该目录。cfgs.log_dir是用于存储训练过程中的日志文件的目录
CHECKPOINT_PATH = cfgs.checkpoint_path if cfgs.checkpoint_path is not None and cfgs.resume else None
if not os.path.exists(cfgs.log_dir):
    os.makedirs(cfgs.log_dir)

LOG_FOUT = open(os.path.join(cfgs.log_dir, 'log_train.txt'), 'a')
LOG_FOUT.write(str(cfgs) + '\n')

# 名为log_string的函数，用于将输出字符串out_str写入LOG_FOUT中并刷新缓冲区，同时也将该字符串打印到控制台上
def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


# Init datasets and dataloaders 
# 名为my_worker_init_fn的函数，用作PyTorch的DataLoader在多线程情况下的初始化函数
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass

print("train")
# 加载数据标签
grasp_labels = load_grasp_labels(cfgs.dataset_root)
# 加载训练数据
TRAIN_DATASET = GraspNetDataset(cfgs.dataset_root, grasp_labels=grasp_labels, camera=cfgs.camera, split='train1',
                                num_points=cfgs.num_point, voxel_size=cfgs.voxel_size,
                                remove_outlier=True, augment=True, load_label=True)
print('train dataset length: ', len(TRAIN_DATASET))
# 对训练数据用DataLoader
TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=cfgs.batch_size, shuffle=True,
                              num_workers=0, worker_init_fn=my_worker_init_fn, collate_fn=minkowski_collate_fn)
print('train dataloader length: ', len(TRAIN_DATALOADER))


# 网络是GraspNet，并将数据移动至 cuda：0
net = GraspNet(seed_feat_dim=cfgs.seed_feat_dim, is_training=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

# Load the Adam optimizer 优化器用Adam优化器
optimizer = optim.Adam(net.parameters(), lr=cfgs.learning_rate)

start_epoch = 0
# 判断CHECKPOINT_PATH是否不为None且指定的文件路径存在。
# 如果满足条件，则表示存在之前保存的检查点文件，可以进行模型参数和优化器状态的加载
# 使用torch.load()函数加载CHECKPOINT_PATH指定的检查点文件，将其存储在checkpoint变量中
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    # 加载模型参数model_state_dict，和优化器参数optimizer_state_dict
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] #表示从该轮开始继续训练。
    # 使用SummaryWriter创建一个名为TRAIN_WRITER的TensorBoard可视化对象，用于记录训练过程中的指标和可视化结果
    log_string("-> loaded checkpoint %s (epoch: %d)" % (CHECKPOINT_PATH, start_epoch))
# TensorBoard Visualizers
TRAIN_WRITER = SummaryWriter(os.path.join(cfgs.log_dir, 'train'))

# 学习率设置 学习率按照指数衰减的方式进行更新。每经过一轮训练，学习率会乘以0.95，即每轮学习率减小5%
def get_current_lr(epoch):
    lr = cfgs.learning_rate
    lr = lr * (0.95 ** epoch)
    return lr

# 根据当前轮数更新优化器中的学习率，以实现学习率的动态调整
def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_one_epoch():
    stat_dict = {}  # collect statistics
    # 调用adjust_learning_rate(optimizer, EPOCH_CNT)函数，根据当前轮数EPOCH_CNT调整优化器optimizer的学习率
    adjust_learning_rate(optimizer, EPOCH_CNT)
    net.train()
    batch_interval = 20
    # 对于每个批次的数据和标签，将其移动到设备（如GPU）上进行计算，以便利用硬件加速
    for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):
        for key in batch_data_label:
            if 'list' in key:
                for i in range(len(batch_data_label[key])):
                    for j in range(len(batch_data_label[key][i])):
                        batch_data_label[key][i][j] = batch_data_label[key][i][j].to(device)
            else:
                batch_data_label[key] = batch_data_label[key].to(device)
        # 模型net对批次数据进行前向传播，得到输出end_points
        
        # print(batch_data_label['object_list'].shape)
        
        end_points = net(batch_data_label)
        if end_points is None:
            continue
        print("本次实验，A+B+C+D，kn相机.1127")

        # get_loss(end_points)函数计算损失，并返回损失值和更新后的end_points
        loss, end_points = get_loss(end_points)
        print(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # 统计并输出每个批次的训练指标，并将指标写入TensorBoard日志文件以进行可视化
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'prec' in key or 'recall' in key or 'count' in key:
                if key not in stat_dict:
                    stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        if (batch_idx + 1) % batch_interval == 0:
            log_string(' ----epoch: %03d  ---- batch: %03d ----' % (EPOCH_CNT, batch_idx + 1))
            for key in sorted(stat_dict.keys()):
                TRAIN_WRITER.add_scalar(key, stat_dict[key] / batch_interval,
                                        (EPOCH_CNT * len(TRAIN_DATALOADER) + batch_idx) * cfgs.batch_size)
                log_string('mean %s: %f' % (key, stat_dict[key] / batch_interval))
                stat_dict[key] = 0

# 多轮训练
def train(start_epoch):
    global EPOCH_CNT
    for epoch in range(start_epoch, cfgs.max_epoch):
        EPOCH_CNT = epoch
        log_string('**** EPOCH %03d ****' % epoch)
        log_string('Current learning rate: %f' % (get_current_lr(epoch)))
        log_string(str(datetime.now()))
        # Reset numpy seed.
        # REF: https://github.com/pytorch/pytorch/issues/5059
        np.random.seed()
        train_one_epoch()

        save_dict = {'epoch': epoch + 1, 'optimizer_state_dict': optimizer.state_dict(),
                     'model_state_dict': net.state_dict()}
        torch.save(save_dict, os.path.join(cfgs.log_dir, cfgs.model_name + '_epoch' + str(epoch + 1).zfill(2) + '.tar'))


if __name__ == '__main__':
    train(start_epoch)
