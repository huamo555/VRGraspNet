<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=rect&height=150&color=0:0f172a,55:2563eb,100:8b5cf6&text=VRGraspNet" alt="VRGraspNet banner">
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

This repository provides the research code for training, testing, graspness generation, visualization, collision checking, and GraspNet-style AP / APu evaluation.

## Highlights

<table>
  <tr>
    <td><b>Viewpoint robustness</b></td>
    <td>Improves grasp pose estimation under diverse camera viewpoints and partial observations.</td>
  </tr>
  <tr>
    <td><b>6-DoF grasp prediction</b></td>
    <td>Generates full spatial grasp poses for cluttered scenes.</td>
  </tr>
  <tr>
    <td><b>Graspness-aware learning</b></td>
    <td>Includes tools for graspness generation and visualization.</td>
  </tr>
  <tr>
    <td><b>Point cloud backbone support</b></td>
    <td>Provides PointNet++ / KNN / ResUNet-related modules for 3D feature extraction.</td>
  </tr>
  <tr>
    <td><b>Complete experimental pipeline</b></td>
    <td>Covers training, testing, visualization, collision checking, and metric computation.</td>
  </tr>
</table>

## Method

VRGraspNet targets the viewpoint sensitivity problem in 6-DoF grasp pose estimation. The framework extracts scene-level and local geometric features from RGB-D-derived point clouds, estimates graspness-related cues, and predicts reliable grasp poses even when the observation viewpoint changes.

The overall pipeline contains:

1. **RGB-D / point cloud preprocessing** for cluttered scene representation.
2. **Backbone feature extraction** using point cloud and convolutional modules.
3. **Viewpoint-robust grasp representation learning** for stable local geometry encoding.
4. **6-DoF grasp pose decoding** with grasp score estimation.
5. **Collision-aware post-processing and benchmark evaluation**.

### Framework

Place the framework figure at `doc/framework.png`; GitHub will render it below.

<p align="center">
  <img src="doc/framework.png" width="90%" alt="VRGraspNet framework">
</p>

## Repository Structure

```text
VRGraspNet/
|-- knn/                         # KNN CUDA/C++ extensions
|-- pointnet2/_ext_src/          # PointNet++ CUDA/C++ extension source
|-- SE_resUnet.py                # SE-ResUNet module
|-- backbone_resunet14.py        # Backbone network
|-- collision_detector.py        # Collision checking
|-- data_utils.py                # Data processing utilities
|-- generate_graspness.py        # Graspness label / score generation
|-- get_AP_and_APu.py            # AP / APu metric computation
|-- get_AP_and_APu.sh            # Evaluation script
|-- graspnet.py                  # Main grasp network
|-- graspnet_dataset.py          # Dataset loader
|-- infer_vis_grasp.py           # Inference and visualization
|-- knn_modules.py               # KNN modules
|-- label_generation.py          # Label generation
|-- loss.py                      # Training losses
|-- loss_utils.py                # Loss utilities
|-- modules.py                   # Core model modules
|-- pointnet2_modules.py         # PointNet++ modules
|-- pointnet2_utils.py           # PointNet++ utilities
|-- pytorch_utils.py             # PyTorch helper utilities
|-- resnet.py                    # ResNet modules
|-- setup.py                     # Extension setup
|-- simplify_dataset.py          # Dataset simplification utility
|-- train.py                     # Training entry point
|-- test.py                      # Testing entry point
|-- test_view.py                 # Viewpoint testing entry point
|-- vis_graspness.py             # Graspness visualization
|-- command_train.sh             # Training script
|-- command_test.sh              # Testing script
|-- command_testview.sh          # Viewpoint robustness testing script
`-- requirements.txt
```

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

### Generate Graspness

```bash
python generate_graspness.py
```

### Training

```bash
bash command_train.sh
```

You can also launch training directly:

```bash
python train.py
```

### Testing

```bash
bash command_test.sh
```

### Viewpoint Robustness Testing

```bash
bash command_testview.sh
```

or:

```bash
python test_view.py
```

### Evaluation

```bash
bash get_AP_and_APu.sh
```

or:

```bash
python get_AP_and_APu.py
```

### Visualization

```bash
python infer_vis_grasp.py
python vis_graspness.py
```

## Results

Final benchmark numbers will be updated after release materials are organized.

### GraspNet-1Billion

| Method | Seen AP | Similar AP | Novel AP | APu |
| --- | ---: | ---: | ---: | ---: |
| Baseline | TBD | TBD | TBD | TBD |
| VRGraspNet | TBD | TBD | TBD | TBD |

### Viewpoint Robustness

| Setting | AP | APu | Notes |
| --- | ---: | ---: | --- |
| Original viewpoint | TBD | TBD | Standard evaluation |
| Changed viewpoint | TBD | TBD | Viewpoint robustness evaluation |
| VRGraspNet | TBD | TBD | Full model |

## Model Zoo

| Model | Dataset | Metric | Checkpoint |
| --- | --- | --- | --- |
| VRGraspNet | GraspNet-1Billion | TBD | Coming soon |

## Roadmap

- [ ] Release pretrained checkpoints.
- [ ] Add final benchmark results.
- [ ] Add viewpoint robustness evaluation details.
- [ ] Add qualitative grasp visualization examples.
- [ ] Add project page and demo video.

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
