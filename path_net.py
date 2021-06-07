import torch
import torch.nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool, global_max_pool, global_mean_pool
from torch_cluster import random_walk


class PathSimplifiedGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_gc_layers, activation, readout):
        super(PathSimplifiedGCN, self).__init__()
        self.activation = activation()
        self.readout = readout
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(hidden_dim)] * num_gc_layers)
    
        for i in range(num_gc_layers):
            if i == 0:
                mlp = torch.nn.Linear(input_dim, hidden_dim)
            else:
                mlp = torch.nn.Linear(hidden_dim, hidden_dim)
            self.convs.append(mlp)

    def forward(self, x, edge_index, batch):
        z = x
        g = []

        row = edge_index[0].cuda()
        col = edge_index[1].cuda()
        start = torch.tensor(list(range(x.shape[0]))).cuda()
        walk2 = torch.transpose(random_walk(row, col, start, walk_length=2), 0, 1)
        walk3 = torch.transpose(random_walk(row, col, start, walk_length=3), 0, 1)
        walk4 = torch.transpose(random_walk(row, col, start, walk_length=4), 0, 1)

        for i, conv in enumerate(self.convs):
            # matrix operation
            z = conv(z)
            path2_emb = z[walk2[0]] + z[walk2[1]] + z[walk2[2]]
            path3_emb = z[walk3[0]] + z[walk3[1]] + z[walk3[2]] + z[walk3[3]]
            path4_emb = z[walk4[0]] + z[walk4[1]] + z[walk4[2]] + z[walk4[3]] + z[walk4[4]]
            path_emb = path2_emb + path3_emb + path4_emb

            path_batch = batch[walk2[0]]

            if self.readout == 'mean':
                g.append(global_mean_pool(path_emb, path_batch))
            elif self.readout == 'max':
                g.append(global_max_pool(path_emb, path_batch))
            elif self.readout == 'sum':
                g.append(global_add_pool(path_emb, path_batch))

        g = torch.cat(g, dim=1)
        return g


class PathSimplifiedMLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_mlp_layers, activation):
        super(PathSimplifiedMLP, self).__init__()
        self.net = []
        self.net.append(torch.nn.Linear(input_dim, hidden_dim))

        for _ in range(num_mlp_layers - 1):
            self.net.append(torch.nn.Linear(hidden_dim, hidden_dim))

        self.net = torch.nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)


class PathPooling(torch.nn.Module):
    def __init__(self, gnn, mlp):
        super(PathPooling, self).__init__()
        self.gnn = gnn 
        self.mlp = mlp 
    
    def forward(self, x, edge_index, batch, device):
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(device)

        g = self.gnn(x, edge_index, batch)
        g = self.mlp(g)

        return g
