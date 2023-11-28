
# Pytorch implementation of HiCAP: Hierarchical Clustering-based Attention Pooling for Graph Representation Learning

A novel method called [HiCAP (Hierarchical Clustering-Based Attention Pooling)](https://ieeexplore.ieee.org/document/10326268) is proposed to address the limitations of existing graph pooling approaches.

## Requirements

    * Python >= 3.6
    * Pytorch >= 2.0.1+cu118
    * Pytorch_geometric >= 2.4.0
    * Numpy
    * Networkx

## Usage

```Python
python3 main.py
```
In case of using CUDA:
```python
CUDA_LAUNCH_BLOCKING=1 python3 main.py
```
NOTE: there are various arguments which you can modify, please see ```main.py``` for further details.

## Cite
```
@INPROCEEDINGS{10326268,
  author={Haddadian, Parsa and Abedian, Rooholah and Moeini, Ali},
  booktitle={2023 13th International Conference on Computer and Knowledge Engineering (ICCKE)}, 
  title={HiCAP: Hierarchical Clustering-based Attention Pooling for Graph Representation Learning}, 
  year={2023},
  volume={},
  number={},
  pages={463-468},
  doi={10.1109/ICCKE60553.2023.10326268}}
```

## Contact Us
Please contact [Parsa](mailto:p.haddadian@ut.ac.ir) with any questions.



