# MV-FGAD
This repository is the official implementation of "[MV-FGAD: Towards Efficient and Effective Federated Graph Anomaly Detection via Multi-view Learning](https://openreview.net/pdf?id=yBcY0bY45t)", accepted by ICML'26.

![](https://github.com/Junyi-Yan/Junyi-Yan.github.io/blob/main/Picture/ICML2026.PNG)

# Overview
Our implementation for STIM is based on PyTorch. 

# Datasets
Datasets can be obtained from [google drive link](https://drive.google.com/drive/folders/1q-i6ANJ5YKsgLfL4L3MI18q1aO2rHCi-). For each dataset, please create a separate subfolder under `datasets/` with the same name as the dataset, and place the corresponding `.mat` file inside it (e.g., `datasets/Amazon/Amazon.mat`).

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

# Baselines
All baselines and their URLs are as follows:  
- FedAvg [[Paper](https://proceedings.mlr.press/v54/mcmahan17a?ref=https://githubhelp.com)] [[Code](https://github.com/zj-jayzhang/FedAvg)]
- FedProx [[Paper](https://proceedings.mlsys.org/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html)] [[Code](https://github.com/litian96/FedProx)]
- GCFL+ [[Paper](https://proceedings.neurips.cc/paper_files/paper/2021/file/9c6947bd95ae487c81d4e19d3ed8cd6f-Paper.pdf)] [[Code](https://github.com/Oxfordblue7/GCFL)]
- FedAux [[Paper](https://proceedings.neurips.cc/paper_files/paper/2025/file/2360da01c2ed6592bb691326424de184-Paper-Conference.pdf)] [[Code](https://github.com/JhuoW/FedAux)]
- LG-FGAD [[Paper](https://www.ijcai.org/proceedings/2024/0416.pdf)] [[Code](https://github.com/wownice333/LG-FGAD)]
- FGAD [[Paper](https://dl.acm.org/doi/abs/10.1145/3664647.3681415)] [[Code](https://github.com/wownice333/FGAD)]
- FedCLGN [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/35458)] [[Code](https://github.com/Theodore-Silas/FedCLGN)]


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
