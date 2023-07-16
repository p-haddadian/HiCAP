
# Pytorch implementation of Hierarchial Clustering-based Attention Pooling for Graph Representation Learning

A novel method called HiCAP (Hierarchical Cluster-Based Attention Pooling) is proposed to address the limitations of existing graph pooling approaches. By combining the strengths of node clustering and node dropping techniques, HiCAP establishes a hierarchical framework. Initially, a soft cluster assignment matrix is learned through the application of a GNN. Subsequently, the matrix undergoes a transformation into a hard assignment by incorporating structural considerations. Attention-based scoring is subse- quently employed to select representative nodes within each cluster.

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


## Contact Us
Please contact [Parsa](mailto:p.haddadian@ut.ac.ir) with any questions.



