from torch_geometric.loader import DataLoader
import torch

def num_nodes_extraction(loader:DataLoader):
    '''
    extract the number of nodes in a dataloader containing multiple graphs
    '''
    num_nodes_list = list()
    for data in loader:
        num_nodes = data.x.shape[0]
        num_nodes_list.append(num_nodes)
    return num_nodes_list

def max_node_find(loader: DataLoader):
    max = 0
    for data in loader:
        if data.x.shape[0] > max:
            max = data.x.shape[0]
    
    return max

# Finds the edges between clusters via iterating all the edges in the graph
def find_cluster_edges(edge_index, assignment_matrix, nonzero_clusters):
    # create a list to store the edges exist between clusters
    new_edges = list()

    # Iterate over all the edges and see whether there exists a link between clusters and add to the final graph
    for i in range(edge_index.size(1)):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()

        # print('assignment matrix src: ', assignment_matrix[src])
        src_cluster = torch.nonzero(assignment_matrix[src]).item()
        dst_cluster = torch.nonzero(assignment_matrix[dst]).item()

        # If clsuters are not the same add an edge between cluster for the new graph
        if src_cluster != dst_cluster:
            if src_cluster in nonzero_clusters and dst_cluster in nonzero_clusters:
                new_edges.append((src_cluster, dst_cluster))

    new_edges = list(set(new_edges))
    # convert the list of edges to edge index format
    if len(new_edges) > 0:
        new_edge_index = torch.tensor(new_edges, dtype=torch.long).t().contiguous()
    else:
        new_edge_index = torch.empty((2, 0), dtype=torch.long)

    return new_edge_index

# Normalizes edge index in order to start from zero
def normalize_edge_index(edge_index):
    # Create a set of unique node IDs
    unique_nodes = set(edge_index.flatten().tolist())

    # Create a mapping from old node IDs to new node IDs
    node_mapping = {node: i for i, node in enumerate(unique_nodes)}

    # Create a new edge index with normalized node IDs
    normalized_edge_index = torch.tensor([
        [node_mapping[edge_index[0, i].item()] for i in range(edge_index.size(1))],
        [node_mapping[edge_index[1, i].item()] for i in range(edge_index.size(1))]
    ], dtype=torch.long)

    return normalized_edge_index


def find_max_batch(score, batch):
    if score.shape[0] != batch.shape[0]:
        raise "Score and batch must have the same size at dimension 0"
    
    num_batch = torch.max(batch) + 1
    # print('num batch', batch)

    maxes = list()
    indexes = list()
    for batch_id in range(num_batch):
        # times -2 because the range of score is (-1, 1)
        selected = (batch != batch_id) * -2 + (batch == batch_id) * score
        if torch.all(selected == -2).item():
            continue
        max = torch.max(selected).item()
        index = torch.argmax(selected).item()
        
        maxes.append(max)
        indexes.append(index)
    maxes = torch.Tensor(maxes)

    return maxes, indexes
