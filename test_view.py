import os
import sys
import numpy as np
import argparse
import time
import torch
from torch.utils.data import DataLoader
""" from graspnetAPI.graspnet_eval import GraspGroup, GraspNetEval """
from graspnetAPI.graspnet_eval_view import GraspGroup, GraspNetEval_view
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))


from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset, minkowski_collate_fn
from collision_detector import ModelFreeCollisionDetector

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default=None, required=True)
parser.add_argument('--checkpoint_path', help='Model checkpoint path', default=None, required=True)
parser.add_argument('--dump_dir', help='Dump dir to save outputs', default=None, required=True)
parser.add_argument('--seed_feat_dim', default=512, type=int, help='Point wise feature dim')
parser.add_argument('--camera', default='kinect', help='Camera split [realsense/kinect]')
parser.add_argument('--num_point', type=int, default=19998, help='Point Number [default: 15000]')
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during inference [default: 1]')
parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel Size for sparse convolution') # 表示体素的大小。cfgs.voxel_size是一个配置参数，用于指定体素的分辨率。
parser.add_argument('--collision_thresh', type=float, default=0.01,
                    help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size_cd', type=float, default=0.01, help='Voxel Size for collision detection')
parser.add_argument('--infer', action='store_true', default=False)
parser.add_argument('--eval', action='store_true', default=False)
cfgs = parser.parse_args()



# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
if not os.path.exists(cfgs.dump_dir):
    os.mkdir(cfgs.dump_dir)


# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass


def inference():
    # 创建了一个名为test_dataset的数据集对象，用于在GraspNet数据集上进行测试
    # remove_outlier=True：表示是否移除离群点。如果设置为True，会进行离群点的移除操作
    # augment=False：表示是否进行数据增强。如果设置为True，会对数据进行增强操作，如旋转、平移或缩放等
    # load_label=False：表示是否加载标签信息。如果设置为True，会加载样本的标签信息，否则只加载点云数据
    test_dataset = GraspNetDataset(cfgs.dataset_root, split='test', camera=cfgs.camera, num_points=cfgs.num_point,
                                   voxel_size=cfgs.voxel_size, remove_outlier=True, augment=False, load_label=False)
    print('Test dataset length: ', len(test_dataset))
    scene_list = test_dataset.scene_list()
    # 创建一个数据加载器（test_dataloader），用于批量加载测试数据。根据提供的配置参数，设置批量大小、是否打乱数据、工作进程数量等
    test_dataloader = DataLoader(test_dataset, batch_size=cfgs.batch_size, shuffle=False,
                                 num_workers=0, worker_init_fn=my_worker_init_fn, collate_fn=minkowski_collate_fn)
    print('Test dataloader length: ', len(test_dataloader))
    # Init the model 初始化模型（net）并将其移动到可用的设备（GPU或CPU）上
    net = GraspNet(seed_feat_dim=cfgs.seed_feat_dim, is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # Load checkpoint 加载预训练的模型参数（checkpoint）
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)" % (cfgs.checkpoint_path, start_epoch))
    
    # 批量处理的间隔（batch_interval）和模型的评估模式（net.eval()）
    batch_interval = 100
    net.eval()
    tic = time.time()
    # 遍历测试数据加载器，逐批加载数据（batch_data），
    # 将其移动到设备上，并进行前向传播。使用模型对数据进行推断，得到抓取预测结果（grasp_preds）
    for batch_idx, batch_data in enumerate(test_dataloader):
        for key in batch_data:
            if 'list' in key:
                for i in range(len(batch_data[key])):
                    for j in range(len(batch_data[key][i])):
                        batch_data[key][i][j] = batch_data[key][i][j].to(device)
            else:
                batch_data[key] = batch_data[key].to(device)

        # Forward pass
        with torch.no_grad():
            end_points = net(batch_data)
            grasp_preds = pred_decode(end_points)

        # Dump results for evaluation
        # 对每个样本进行处理。将抓取预测结果转换为NumPy数组，并创建一个GraspGroup对象（gg）来表示抓取
        for i in range(cfgs.batch_size):
            data_idx = batch_idx * cfgs.batch_size + i
            preds = grasp_preds[i].detach().cpu().numpy()

            gg = GraspGroup(preds)
            # collision detection 
            # 根据配置参数，进行碰撞检测操作，通过与点云数据进行碰撞检测，过滤掉与点云碰撞的抓取结果。将过滤后的抓取结果保存到文件中。
            if cfgs.collision_thresh > 0:
                cloud = test_dataset.get_data(data_idx, return_raw_cloud=True)
                mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size_cd)
                collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
                gg = gg[~collision_mask]

            # save grasps
            save_dir = os.path.join(cfgs.dump_dir, scene_list[data_idx], cfgs.camera)
            save_path = os.path.join(save_dir, str(data_idx % 256).zfill(4) + '.npy')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            gg.save_npy(save_path)
        # 根据配置参数的设定，定期输出评估批次的信息，包括批次索引和处理时间
        if (batch_idx + 1) % batch_interval == 0:
            toc = time.time()
            print('Eval batch: %d, time: %fs' % (batch_idx + 1, (toc - tic) / batch_interval))
            tic = time.time()


def evaluate(dump_dir):
    ge = GraspNetEval_view(root=cfgs.dataset_root, camera=cfgs.camera, split='test',view_model="simple")
    res, ap = ge.eval_all(dump_folder=dump_dir, proc=6)  
    ge = GraspNetEval_view(root=cfgs.dataset_root, camera=cfgs.camera, split='test',view_model="difficult")
    res, ap = ge.eval_all(dump_folder=dump_dir, proc=6)
    ge = GraspNetEval_view(root=cfgs.dataset_root, camera=cfgs.camera, split='test',view_model="infernal")
    res, ap = ge.eval_all(dump_folder=dump_dir, proc=6) 
    
    save_dir = os.path.join(cfgs.dump_dir, 'ap_{}.npy'.format(cfgs.camera))
    np.save(save_dir, res)


if __name__ == '__main__':
    if cfgs.infer:
        inference()
    if cfgs.eval:
        evaluate(cfgs.dump_dir)
