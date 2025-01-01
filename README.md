# Reconnecting Top-𝑙 Relationships (RT𝑙R) query

This rerpository supports the following papaer:

> Reconnecting the Estranged Relationships: Optimizing the
Influence Propagation in Evolving Networks. [\[PDF\]](https://ieeexplore.ieee.org/abstract/document/10254336)

## Requirements

Lastest tested combination: Python 3.9.7 + PyTorch 1.10.1 + PyTorch_Geometric 2.0.3

Install [PyTorch](https://pytorch.org/)

Install [PyTorch_Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

## Usages

### SEAL link pred

Download Dataset from [SNAP](https://snap.stanford.edu/data). We except the directory structure to be the following:

```plaintext
code_root/data/SEALDataset/
└── dataset1/
  └── raw/
    └── dataset1.gz
└── dataset2/
  └── raw/
    └── dataset2.gz
```

Runing `python SEAL_link_pred.py` in shell and result saved in `path/data/SEALDataset/dataset/T{}_pred_edge.pt`, in which the `T` param meams the number of snapshots of dataset.

### RT𝑙R query

```shell
python RTlR_query.py
```

## License

This work is licensed under the Creative Commons BY-NC-ND 4.0 International
License. Visit [license](https://creativecommons.org/licenses/by-nc-nd/4.0/) to view a copy of this license.

## Reference

If you find the code useful, please cite our papers.

```bibtex
@ARTICLE{10254336,
  author={Cai, Taotao and Lei, Qi and Sheng, Quan Z. and Cui, Ningning and Yang, Shuiqiao and Yang, Jian and Zhang, Wei Emma and Mahmood, Adnan},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  title={Reconnecting the Estranged Relationships: Optimizing the Influence Propagation in Evolving Networks}, 
  year={2024},
  volume={36},
  number={5},
  pages={2151-2165},
  keywords={Social networking (online);Topology;Seminars;Network topology;Integrated circuit modeling;Companies;Australia;Evolving networks;graph query;influence maximization;link prediction},
  doi={10.1109/TKDE.2023.3316268}}

```
