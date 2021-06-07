import torch
import torch.nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool, global_max_pool, global_mean_pool


class SimplifiedGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_gc_layers, activation, readout):
        super(SimplifiedGCN, self).__init__()
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

    def forward(self, x, batch):
        z = x
        g = []

        for i, conv in enumerate(self.convs):
            z = conv(z)

            if self.readout == 'mean':
                g.append(global_mean_pool(z, batch))
            elif self.readout == 'max':
                g.append(global_max_pool(z, batch))
            elif self.readout == 'sum':
                g.append(global_add_pool(z, batch))
    
        g = torch.cat(g, dim=1)
        return g


class SimplifiedMLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_mlp_layers, activation):
        super(SimplifiedMLP, self).__init__()
        self.net = []
        self.net.append(torch.nn.Linear(input_dim, hidden_dim))

        for _ in range(num_mlp_layers - 1):
            self.net.append(torch.nn.Linear(hidden_dim, hidden_dim))

        self.net = torch.nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)


class NodePooling(torch.nn.Module):
    def __init__(self, gnn, mlp):
        super(NodePooling, self).__init__()
        self.gnn = gnn
        self.mlp = mlp

    def forward(self, x, batch, device):
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(device)
        
        g = self.gnn(x, batch)
        g = self.mlp(g)

        return g
