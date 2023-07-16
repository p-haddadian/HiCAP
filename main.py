import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.loader import DenseDataLoader
from torch_geometric import utils
from networks import  Net
import torch.nn.functional as F
import torch_geometric.transforms as T
import argparse
import os
from torch.utils.data import random_split
from utils import num_nodes_extraction
from utils import max_node_find

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=1234,
                    help='seed')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.0002,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001,
                    help='weight decay')
parser.add_argument('--nhid', type=int, default=256,
                    help='hidden size')
parser.add_argument('--pooling_ratio', type=float, default=0.1,
                    help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.5,
                    help='dropout ratio')
parser.add_argument('--dataset', type=str, default='DD',
                    help='DD/PROTEINS/NCI1/NCI109/Mutagenicity')
parser.add_argument('--epochs', type=int, default=10000,
                    help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=50,
                    help='patience for earlystopping')
parser.add_argument('--pooling_layer_type', type=str, default='GCNConv',
                    help='DD/PROTEINS/NCI1/NCI109/Mutagenicity')
parser.add_argument('--max_nodes', type=int, default=1000,
                    help='Maximum number of the nodes (Ignore the graphs exceeding this limit)')

args = parser.parse_args()
args.device = 'cpu'

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:0'

# A filter to only consider the nodes with the following condition [Caution]: the pre_filter argument in TUDataset needs to take a data object
# to check whether it satisfies a certain condition.
class MyFilter(object):
    def __call__(self, data):
        return data.num_nodes <= args.max_nodes
    

dataset = TUDataset(os.path.join('data',args.dataset),name=args.dataset)
args.dataset = dataset
# print('Dataset info:', args.dataset[0])
args.num_classes = dataset.num_classes
args.num_features = dataset.num_features

print('**** Dataset: {}\nNum classes: {}\nNum features: {}'.format(args.dataset, args.num_classes, args.num_features))

# 80/10/10 split TODO: use an appropriate function for splitting the dataset and the ratio to main arguments
num_training = int(len(dataset)*0.8)
num_val = int(len(dataset)*0.1)
num_test = len(dataset) - (num_training+num_val)
training_set,validation_set,test_set = random_split(dataset,[num_training,num_val,num_test])
# print('dataset type: training {}, valid {}.'.format(type(training_set), type(validation_set)))


# Loading the data using a DataLoader and making a Net object as our model
train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle= True)
val_loader = DataLoader(validation_set,batch_size=args.batch_size, shuffle= True)
test_loader = DataLoader(test_set,batch_size=1, shuffle= True)

# finding the number of nodes in each batch and maximum number of nodes
args.num_nodes = num_nodes_extraction(train_loader)
args.max_nodes = max_node_find(train_loader)
print('max nodes in graph:', args.max_nodes)


model = Net(args).to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print('parameter name: {} data:{}'.format(name, param.data))

# raise 'force stop'

def test(model,loader):
    model.eval()
    correct = 0.
    loss = 0.
    for data in loader:
        data = data.to(args.device)
        out = model(data)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss += model.loss(out, data.y, data.edge_index).item()
    return correct / len(loader.dataset),loss / len(loader.dataset)

min_loss = 1e10
patience = 0



best_val_acc = test_acc = 0
for epoch in range(args.epochs):
    model.train()
    correct = 0
    loss_all = 0
    for i, data in enumerate(train_loader):
        # print('***data shape: ', data.x.shape)
        data = data.to(args.device)
        out = model(data)
        # loss = F.nll_loss(out, data.y)
        loss = model.loss(out, data.y, data.edge_index)
        # loss = F.cross_entropy(out, data.y)
        # print(out, data.y)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss_all += data.y.size(0) * loss.item()
        print("Training loss:{0:.4f} Training acc: {1:.4f}".format(loss, correct / len(train_loader.dataset)))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    val_acc,val_loss = test(model,val_loader)
    train_loss = loss_all / len(training_set)
    print("Epoch: {0}, Train loss: {1:.4f}, Validation loss:{2:.4f} Validation accuracy:{3:.4f}".format(epoch, train_loss, val_loss, val_acc))
    if val_loss < min_loss:
        torch.save(model.state_dict(),'latest.pth')
        print("Model saved at epoch{}".format(epoch))
        min_loss = val_loss
        patience = 0
    else:
        patience += 1
    if patience > args.patience:
        break 



model = Net(args).to(args.device)
model.load_state_dict(torch.load('latest.pth'))
test_acc,test_loss = test(model,test_loader)
print("Test accuarcy:{}".format(test_acc))
