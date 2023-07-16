from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool.topk_pool import topk,filter_adj
from torch.nn import Parameter
from torch.nn import Softmax
from torch.autograd.variable import Variable
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import subgraph
from torch_geometric.utils import to_networkx
from torch_geometric.utils import unbatch
from utils import find_cluster_edges
from utils import normalize_edge_index
from utils import find_max_batch
import torch.nn.functional as F
import torch

import numpy as np


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, lin=True):
        super(GNN, self).__init__()
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        self.convs.append(GCNConv(in_channels, hidden_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        self.convs.append(GCNConv(hidden_channels, hidden_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        self.convs.append(GCNConv(hidden_channels, out_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(out_channels))


    def forward(self, x, adj, mask=None):
        # batch_size, num_nodes, in_channels = x.size()
        
        for step in range(len(self.convs)):
            x = F.relu(self.convs[step](x, adj))
        

        return x


class HiCAP(torch.nn.Module):
    def __init__(self, in_channels, next_layer_nodes, score_conv = GCNConv, assign_Conv = GNN, non_lin = torch.tanh):
        super(HiCAP, self).__init__()
        self.in_channels = in_channels
        self.next_layer_nodes = next_layer_nodes
        self.score_layer = score_conv(in_channels, 1)
        self.assign_layer = assign_Conv(in_channels, 256, self.next_layer_nodes)
        self.non_lin = non_lin

    def forward(self, x, adj, edge_attr = None, batch = None, mask = None):
        # construct a batch (assume one batch if not existed)
        device = 'cuda:0'
        if batch is None:
            batch = adj.new_zeros(x.size(0))

        # generating soft assign tensor
        self.assign_tensor = self.assign_layer(x, adj)
        self.assign_tensor = Softmax(dim=1)(self.assign_tensor)
        # print('assign tensor shape', self.assign_tensor.shape)
        
        ################### Hard clustering #######################
        print('assign tensor: ', self.assign_tensor[0])
        temp_assign = torch.tensor(self.assign_tensor, requires_grad=False)
        temp_assign = self.neighboring_hard_assignment(adj)

        ###########################################################

        num_clusters = self.assign_tensor.shape[1]
        new_adj = list()
        new_x = list()
        nodes = list()
        new_batch = list()
        clusters_nonzero = list()
        empty_clusters = 0
        for i in range(num_clusters):
            cluster = temp_assign[:, i]
            selected = (cluster == 1)
            sub_adj, _ = subgraph(selected, adj)
            sub_x = x[selected]
            sub_batch = batch[selected]
            # print('sub x shape {} sub batch shape {}'.format(sub_x.shape, sub_batch.shape))
            # Drop the empty clusters (We wish to minimize the number of empty clusters)
            if sub_adj.shape[1] == 0:
                empty_clusters += 1
                continue
            clusters_nonzero.append(i)
            
            # A post-processing is done on edge index in order to start node IDs from zero.
            flat_edge_index = sub_adj.flatten()
            unique_node_ids = torch.unique(flat_edge_index)

            # Create a mapping from old node IDs to new node IDs to start all IDs from zero
            node_id_map = {old_id.item(): new_id for new_id, old_id in enumerate(unique_node_ids)}

            # Adjust the node IDs in the edge index tensor
            sub_adj_modified = torch.tensor([[node_id_map[old_id.item()] for old_id in edge_index_row] for edge_index_row in sub_adj]).to(device)

            # Scoring layer to find the most representative node in sub_adj. 
            score = self.score_layer(sub_x, sub_adj_modified).squeeze()
            score = self.non_lin(score)
            
            # max score and its index
            alpha, index = find_max_batch(score, sub_batch)
            # print(sub_x.shape[0], index)
            # print('alpha', alpha)
            # print('sub x index', sub_x[index].shape, alpha.shape)
            alpha = alpha.to(device)
            sub_x = sub_x[index] * alpha[:, None]
            sub_batch = sub_batch[index]
            
            # print('sub x shape', sub_x.shape)
            # sub_x = sub_x.reshape(1, -1)
            new_x.append(sub_x)
            new_batch.append(sub_batch)
            
            # A list containing all the nodes in the final induced graph
            nodes.extend(torch.unique(sub_adj.flatten()))

        # print('percent of empty clusters:', empty_clusters / num_clusters)
        
        # Edges between clusters
        new_adj = find_cluster_edges(adj, temp_assign, clusters_nonzero)
        new_adj = normalize_edge_index(new_adj)
        
        new_x = torch.cat(new_x, dim = 0)
        new_batch = torch.cat(new_batch, dim = 0)
        # print('batch and x shapes:', new_x.shape, new_batch.shape)

        return new_x, new_adj, new_batch, self.assign_tensor

    # Neighboring hard assignment (based on neighbors and threshold)
    def neighboring_hard_assignment(self, adj):
        S = self.assign_tensor
        threshold = 1 /  S.size(dim = 1)
        # eps = threshold / 50

        threshold = Variable(torch.Tensor([threshold]))
        S1 = (S > (threshold)).float() * 1
        S2 = (S == threshold).float() * threshold
        S = S1 + S2
        
        for row in range(S.size(0)):
            non_zeros = torch.nonzero((S[row] == 1))
            selected = np.random.choice(non_zeros.reshape(-1))
            
            S[row] = 0
            S[row, selected] = 1

        uncertains = (S == threshold)
        indices = uncertains.nonzero()
        # print(indices.shape)

        # A modification for times when batch is presented is rquired
        for index in indices:
            i = index[0]
            j = index[1]
            S[i] = self.neighboring_label(S, adj, i)
        return S
    
    def neighboring_label(self, S, adj, i):
        data = Data(edge_index=adj)
        graph = to_networkx(data)
        
        neighbors = [n for n in graph[i]]
        num_clusters = S.size(-1)
        clusters_list = torch.zeros(num_clusters)

        for k in range(num_clusters):
            selected = (S[neighbors, k] == 1).float() * 1
            total = torch.sum(selected)
            clusters_list[k] = total
    
        maxi = torch.argmax(clusters_list)
        S[i] = 0
        S[i, maxi] = 1

        return S[i]
    