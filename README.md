<p align="center">
 <img src="https://capsule-render.vercel.app/api?type=rect&height=150&color=0:38bdf8,50:60a5fa,100:c084fc&text=VRGraspNet" alt="VRGraspNet banner">
</p>

<h3 align="center">
筛选文件

</p>

<h3 align="center">
  VRGraspNet: Toward Viewpoint Robust 6-DoF Grasp Pose Estimation
</h3>

<p align="center">
  <a href="https://github.com/huamo555/VRGraspNet"><img src="https://img.shields.io/badge/Project-VRGraspNet-2563eb.svg" alt="Project"></a>
  <img src="https://img.shields.io/badge/IEEE%20TCSVT-2025-f97316.svg" alt="TCSVT 2025">
  <img src="https://img.shields.io/badge/Task-6--DoF%20Grasp%20Pose%20Estimation-7c3aed.svg" alt="Task">
  <img src="https://img.shields.io/badge/Python-80.6%25-3776ab.svg" alt="Python">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-See%20LICENSE-16a34a.svg" alt="License"></a>
</p>

<p align="center">
  <a href="#news">News</a> |
  <a href="#overview">Overview</a> |
  <a href="#method">Method</a> |
  <a href="#installation">Installation</a> |
  <a href="#quick-start">Quick Start</a> |
  <a href="#results">Results</a> |
  <a href="#citation">Citation</a>
</p>

<p align="center">
  Official implementation of <b>VRGraspNet</b>, published in <b>IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), 2025</b>.
</p>

---

## News

- **2025**: VRGraspNet was published in **IEEE TCSVT**.
- **2025**: The official implementation was released.
- **Coming soon**: Pretrained checkpoints, detailed benchmark tables, and more visualization examples.

## Overview

VRGraspNet is a viewpoint-robust framework for **6-DoF grasp pose estimation** in cluttered robotic manipulation scenes. Single-view RGB-D observations are often incomplete and sensitive to camera viewpoint changes, which can cause unstable grasp predictions. VRGraspNet is designed to improve the robustness of grasp detection under different viewpoints by learning stronger geometric representations from point clouds and local grasp-aware features.

## Installation

### 1. Clone

```bash
git clone https://github.com/huamo555/VRGraspNet.git
cd VRGraspNet
```

### 2. Create Environment

```bash
conda create -n vrgraspnet python=3.8 -y
conda activate vrgraspnet
pip install -r requirements.txt
```

### 3. Compile CUDA Extensions

```bash
python setup.py install
```

If compilation fails, please check the compatibility of PyTorch, CUDA, GCC, and your GPU driver.

## Dataset Preparation

This project follows the **GraspNet-1Billion** benchmark setting. Please download the dataset from the official GraspNet website and organize it as follows:

```text
data/
`-- graspnet/
    |-- scenes/
    |-- models/
    |-- dex_models/
    |-- grasp_label/
    `-- collision_label/
```

Then update the dataset root path in the corresponding scripts or configuration files.

## Quick Start

### Training

```bash
bash command_train.sh
```

### Testing

```bash
bash command_test.sh
```

## Citation

If you find this project useful, please consider citing our paper:

```bibtex
@article{vrgraspnet2025,
  title   = {VRGraspNet: Toward Viewpoint Robust 6-DoF Grasp Pose Estimation},
  author  = {Gao, Yuming and Wang, Lichun and Zheng, Jiaqi and Xu, Kai and Yao, Huayang and Yin, Baocai},
  journal = {IEEE Transactions on Circuits and Systems for Video Technology},
  year    = {2025}
}
```

## Acknowledgements

This project is built upon the GraspNet benchmark and related open-source 6-DoF grasp pose estimation projects. We sincerely thank the authors and contributors for their valuable work.

## Contact

For questions, suggestions, or collaboration, please open an issue in this repository.

## License

This repository is released under the license specified in [LICENSE](LICENSE).
