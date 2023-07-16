import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from layers import SAGPool
from layers import AGPool
from layers import GNN
from torch_geometric.utils import to_dense_adj



class Net(torch.nn.Module):
    def __init__(self,args):
        super(Net, self).__init__()
        self.args = args
        self.num_nodes = args.max_nodes
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        
        self.conv1 = GCNConv(self.num_features, self.nhid)
        num_nodes = int(args.max_nodes * self.pooling_ratio)
        self.pool1 = AGPool(self.nhid, num_nodes)
        self.conv2 = GCNConv(self.nhid, self.nhid)
        num_nodes = int(num_nodes * self.pooling_ratio)
        self.pool2 = AGPool(self.nhid, num_nodes)
        self.conv3 = GCNConv(self.nhid, self.nhid)
        num_nodes = int(num_nodes * self.pooling_ratio)
        self.pool3 = AGPool(self.nhid, self.num_nodes)

        self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid//2)
        self.lin3 = torch.nn.Linear(self.nhid//2, self. num_classes)

    def forward(self, data):
        '''
        forward method
        '''
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # print('x {} edge_index {} batch {}'.format(x.shape, edge_index.shape, batch.shape))
        # for the situation where adj matrix is given execute the following code
        # edge_index = edge_index.nonzero().t().contiguous()
        num_nodes = x.shape[0]

        # print('x before', x.shape)
        x = F.relu(self.conv1(x, edge_index))
        # print('x conv1', x.shape)
        x, self.edge_index, batch, self.s = self.pool1(x, edge_index, None, batch)
        self.edge_index = self.edge_index.to('cuda:0')
        # print('x after', x.shape)
        # print('adj after', edge_index)
        x_read = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        self.batch = batch
        x = F.relu(self.conv2(x, self.edge_index))
        if edge_index.shape[1] > 0:
            x, self.edge_index, batch, self.s = self.pool2(x, self.edge_index, None, batch)
            x_read = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
            self.edge_index = self.edge_index.to('cuda:0')
        # print('shape x read', x_read.shape)
        # x = F.relu(self.conv3(x, edge_index))
        # x, edge_index = self.pool3(x, edge_index, None, batch)
        # x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # x = x1 + x2 + x3

        x = F.relu(self.lin1(x_read))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        self.x = x
        return x

    def loss(self, pred, label, adj = None):
        s = self.s
        loss = F.nll_loss(pred, label)

        # print('s trans shape', s.transpose(0, 1).shape)
        mat = torch.matmul(s, s.transpose(0, 1))
        # print('batch shape', self.batch)
        # print('edge index shape', self.edge_index)
        # dense_adj = to_dense_adj(self.edge_index, batch = self.batch)
        num_nodes = mat.shape[0]
        edge_sparse = torch.sparse_coo_tensor(self.edge_index, torch.ones(self.edge_index.shape[1], device = 'cuda:0'), (num_nodes, num_nodes))
        dense_adj = edge_sparse.to_dense()
        # print('adj shape: {}, s shape {}'.format(dense_adj.shape, s.shape))
        link_loss = dense_adj - mat
        link_loss = torch.norm(link_loss, p = 'fro')
        link_loss = link_loss / adj.numel()

        ent_loss = (-s * torch.log(s + 1e-15)).sum(dim = -1).mean()

        return loss + link_loss + ent_loss