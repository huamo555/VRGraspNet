# VRGraspNet

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&height=180&color=0:0ea5e9,100:8b5cf6&text=VRGraspNet" alt="VRGraspNet banner">
</p>

<h3 align="center">
  Toward Viewpoint Robust 6-DoF Grasp Pose Estimation
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
