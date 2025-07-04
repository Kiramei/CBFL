# **Learning Behavior Aware Features Across Spaces for Improved 3D Human Motion Prediction**
This repository is the official implementation of the paper "Learning Behavior Aware Features Across Spaces for Improved 3D Human Motion Prediction"

![The structure of CBFL](./doc/assets/pipeline.jpg)



## Abstract

3D skeleton-based human motion prediction is an essential and challenging task for human-machine interactions, aiming to forecast future poses given a history of previous motions. However, most existing works model human motion dependencies exclusively in Euclidean space, neglecting the human motion representation in Euclidean space leads to distortions and loss of information when representation dimensions increase. In this paper, we propose **Cross-space Behavior-aware Feature Learning** Networks that can not only exploit the spatial-temporal kinematic correlations in Euclidean space, but also capture effect and compact dependencies and motion dynamics in Geometric algebra space. Specifically, we develop a Geometric Algebra Dependency-Aware Extractor, incorporating Geometric Algebra-based Full Connection layers to adapt to geometric algebraic space, thus enabling the extraction of human action dependency representations. Additionally, we design an Euclidean Kinematic-Aware Extractor utilizing temporal-wise Kinematic-Aware Attention and spatial-wise Kinematic-Aware Feature Extraction. These two modules enhance and complement each other, leading to effective human motion prediction. Extensive experiments are conducted to quantitatively and qualitatively verify that our proposed CBFL consistently boosts existing methods by **4.3%** of MPJPE on average on Human 3.6M datasets.  



> The repo will be updated after the code reconstruction is finished.
