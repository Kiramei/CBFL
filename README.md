
# **Learning Behavior-Aware Features Across Spaces for Improved 3D Human Motion Prediction**

This repository contains the official implementation of the paper:
**Learning Behavior-Aware Features Across Spaces for Improved 3D Human Motion Prediction**

![Pipeline](./doc/assets/pipeline.jpg)

---

## 📄 Abstract

3D skeleton-based human motion prediction is an essential and challenging task for human-machine interactions, aiming to forecast future poses given a history of previous motions. However, most existing works model human motion dependencies exclusively in Euclidean space, neglecting the human motion representation in Euclidean space leads to distortions and loss of information when representation dimensions increase. In this paper, we propose **Cross-space Behavior-aware Feature Learning (CBFL)** Networks that can not only exploit the spatial-temporal kinematic correlations in Euclidean space, but also capture effect and compact dependencies and motion dynamics in Geometric algebra space. Specifically, we develop a Geometric Algebra Dependency-Aware Extractor, incorporating Geometric Algebra-based Full Connection layers to adapt to geometric algebraic space, thus enabling the extraction of human action dependency representations. Additionally, we design an Euclidean Kinematic-Aware Extractor utilizing temporal-wise Kinematic-Aware Attention and spatial-wise Kinematic-Aware Feature Extraction. These two modules enhance and complement each other, leading to effective human motion prediction. Extensive experiments are conducted to quantitatively and qualitatively verify that our proposed CBFL consistently boosts existing methods by **4.3%** of MPJPE on average on Human3.6M datasets.

---

## 📂 Dataset Preparation

### Human3.6M

Download Human3.6M dataset from their [official website](http://vision.imar.ro/human3.6m/description.php)

```
datasets/
└── H3.6m/
    ├── S1/
    ├── S5/
    ├── S6/
    └── ...
```

---

### AMASS

Download AMASS dataset from their [official website](https://amass.is.tue.mpg.de/en).

```
datasets/
└── amass/
    ├── ACCAD/
    ├── BioMotionLab_NTroje/
    ├── CMU/
    └── ...
```

---

### 3DPW

Download 3DPW dataset from their [official website](https://virtualhumans.mpi-inf.mpg.de/3DPW/).

```
datasets/
└── 3dpw/
    ├── imageFiles/
    │   ├── courtyard_arguing_00/
    │   ├── courtyard_backpack_00/
    │   └── ...
    └── sequenceFiles/
        ├── test/
        ├── train/
        └── validation/
```

---

## ⚙️ Prerequisites

### Environment setup

1️⃣ Clone the repository:

```bash
git clone https://github.com/Kiramei/CBFL.git
cd CBFL
```

2️⃣ (Optional but recommended) Create a virtual environment:

```bash
conda create -n cbfl python=3.10
conda activate cbfl
```

or use `mamba` for faster installation:

```bash
mamba create -n cbfl python=3.10
mamba activate cbfl
```

3️⃣ Install [PyTorch](https://pytorch.org/get-started/locally/) and torchvision according to your system & CUDA version.

4️⃣ Install other dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Training

### Human3.6M

```bash
python exp_h36m_3d.py --kernel_size 10 --dct_n 20 --input_n 50 --output_n 10 --skip_rate 1 --batch_size 32 --test_batch_size 32 --in_features 66
```

### 3DPW

```bash
python exp_3dpw_3d.py --kernel_size 10 --dct_n 20 --input_n 50 --output_n 10 --skip_rate 1 --batch_size 32 --test_batch_size 32 --in_features 66
```

### CMU MOCAP

```bash
python exp_cmu_3d.py --kernel_size 10 --dct_n 20 --input_n 50 --output_n 10 --skip_rate 1 --batch_size 128 --test_batch_size 128 --in_features 75 --epoch 100 --num_stage 18
```

---

## 📝 Evaluation

### Human3.6M

```bash
python exp_h36m_3d.py --kernel_size 10 --dct_n 20 --input_n 50 --output_n 25 --skip_rate 1 --batch_size 32 --test_batch_size 32 --in_features 66 --is_eval --ckpt PATH_TO_YOUR_CHECKPOINT.pth.tar
```

### 3DPW

```bash
python exp_3dpw_3d.py --kernel_size 10 --dct_n 20 --input_n 50 --output_n 25 --skip_rate 1 --batch_size 32 --test_batch_size 32 --in_features 66 --is_eval --ckpt PATH_TO_YOUR_CHECKPOINT.pth.tar
```

### CMU MOCAP

```bash
python exp_cmu_3d.py --kernel_size 10 --dct_n 20 --input_n 50 --output_n 10 --skip_rate 1 --batch_size 128 --test_batch_size 128 --in_features 75 --epoch 100 --num_stage 18 --is_eval --ckpt PATH_TO_YOUR_CHECKPOINT.pth.tar
```

---

## 📧 Citation & Contact

If you find this work helpful, please cite our paper (paper citation coming soon).
For questions or feedback, feel free to open an issue or contact us.

1. Code Citation:
    ```bibtex
    @software{kiramei_2025_15809246,
      author       = {キラメイ},
      title        = {Kiramei/CBFL: Release the Code for the Paper},
      month        = jul,
      year         = 2025,
      publisher    = {Zenodo},
      version      = {v1},
      doi          = {10.5281/zenodo.15809246},
      url          = {https://doi.org/10.5281/zenodo.15809246},
    }
    ```
