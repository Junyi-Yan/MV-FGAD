# MV-FGAD
[ICML'26] MV-FGAD: Towards Efficient and Effective Federated Graph Anomaly Detection via Multi-view Learning

This repository is the official implementation of "[MV-FGAD: Towards Efficient and Effective Federated Graph Anomaly Detection via Multi-view Learning](https://openreview.net/pdf?id=yBcY0bY45t)", accepted by ICML'26.

![](https://github.com/Junyi-Yan/Junyi-Yan.github.io/blob/main/Picture/ICML2026.PNG)

# Overview
Our implementation for STIM is based on PyTorch. 

# Datasets
Datasets can be obtained from [google drive link](https://drive.google.com/drive/folders/1q-i6ANJ5YKsgLfL4L3MI18q1aO2rHCi-). 

# Requirments
This code requires the following:

- Python==3.9
- PyTorch==2.0.1+cu118
- Pytorch Geometric==2.6.1
- Numpy==1.23.0
- Scipy==1.13.1
- Scikit-learn==1.6.1
- NetworkX==3.2.1
- OGB==1.3.6
- DGL==1.1.2+cu118

# Usage
```
python main.py
```

# Cite
If you compare with, build on, or use aspects of this work, please cite the following:

```
@inproceedings{yanmv,
  title={MV-FGAD: Towards Efficient and Effective Federated Graph Anomaly Detection via Multi-view Learning},
  author={Yan, Junyi and Liang, Ke and Yu, Hao and Liu, Meng and Tan, Hao and Liu, Tianrui and Huang, Jun-Jie and Liu, Xinwang},
  booktitle={Forty-third International Conference on Machine Learning},
  year={2026}
}
```
