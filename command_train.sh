CUDA_VISIBLE_DEVICES=1 python train.py --camera realsense --log_dir "/data3/gaoyuming/mutilview_project/graspness_A+B+C+D/graspness_yuan/output/re_1124_A+B+C+D/" --batch_size 4 --learning_rate 0.001 --model_name minkuresunet --dataset_root  "/data3/gaoyuming/project/datasets/datasets/dataset-data/" --resume --checkpoint_path "/data3/gaoyuming/mutilview_project/graspness_A+B+C+D/graspness_yuan/output/re_1124_A+B+C+D/minkuresunet_epoch24.tar"

#"--camera","kinect",
#                "--dataset_root","/data2/gaoyuming/.cache/datasets/dataset-data/",
#                "--model_name", "minkuresunet"