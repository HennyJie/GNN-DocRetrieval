import torch
import torch.nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.nn import GATConv, EdgeConv


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_gc_layers, activation, readout):
        super(GCN, self).__init__()
        self.activation = activation()
        self.readout = readout
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(hidden_dim)] * num_gc_layers)
    
        for i in range(num_gc_layers):
            if i == 0:
                mlp = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, hidden_dim), self.activation, torch.nn.Linear(hidden_dim, hidden_dim))
            else:
                mlp = torch.nn.Sequential(
                    torch.nn.Linear(hidden_dim, hidden_dim), self.activation, torch.nn.Linear(hidden_dim, hidden_dim))
            conv = GINConv(mlp)
            self.convs.append(conv)

    def forward(self, x, edge_index, batch):
        z = x
        g = []

        for i, conv in enumerate(self.convs):
            z = conv(z, edge_index)
            z = self.activation(z)
            z = self.bns[i](z)

            if self.readout == 'mean':
                g.append(global_mean_pool(z, batch))
            elif self.readout == 'max':
                g.append(global_max_pool(z, batch))
            elif self.readout == 'sum':
                g.append(global_add_pool(z, batch))
        
        g = torch.cat(g, dim=1)
        return g


class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_gc_layers, first_heads, output_heads, dropout, activation, readout):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.activation = activation()
        self.readout = readout
        self.convs = torch.nn.ModuleList()
        for i in range(num_gc_layers):
            if i == 0:
                conv = GATConv(input_dim, hidden_dim, heads=first_heads, dropout=dropout)
            else:
                conv = GATConv(hidden_dim*first_heads, hidden_dim, heads=output_heads, concat=False, dropout=dropout)
            self.convs.append(conv)

    def forward(self, x, edge_index, batch):
        z = x
        g = []

        for i, conv in enumerate(self.convs):
            z = F.dropout(z, p=self.dropout, training=self.training)
            z = conv(z, edge_index)
            if i != len(self.convs)-1:
                z = self.activation(z)

            if self.readout == 'mean':
                g.append(global_mean_pool(z, batch))
            elif self.readout == 'max':
                g.append(global_max_pool(z, batch))
            elif self.readout == 'sum':
                g.append(global_add_pool(z, batch))
        
        g = torch.cat(g, dim=1)
        return g


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_mlp_layers, activation):
        super(MLP, self).__init__()
        self.net = []
        self.net.append(torch.nn.Linear(input_dim, hidden_dim))
        self.net.append(activation())

        for _ in range(num_mlp_layers - 1):
            self.net.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.net.append(activation())

        self.net = torch.nn.Sequential(*self.net)
        self.shortcut = torch.nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        return self.net(x) + self.shortcut(x)


class Net(torch.nn.Module):
    def __init__(self, gnn, mlp):
        super(Net, self).__init__()
        self.gnn = gnn
        self.mlp = mlp

    def forward(self, x, edge_index, batch, device):
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(device)

        g = self.gnn(x, edge_index, batch)
        g = self.mlp(g)

        return g
