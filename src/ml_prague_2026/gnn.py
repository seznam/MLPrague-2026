import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

import sympy
import scipy


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.3, aggr='mean'):
        super().__init__()
        self.dropout_rate = dropout
        self.convs = nn.ModuleList()
        
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
        
        self.head = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.mish(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        return self.head(x)


# BWGNN
def precompute_norm_adj(edge_index, num_nodes):
    row, col = edge_index
    deg = torch.zeros(num_nodes, dtype=torch.float, device=row.device)
    deg.scatter_add_(0, row, torch.ones(row.shape[0], device=row.device))
    deg_inv_sqrt = deg.clamp(min=1).pow(-0.5)

    weights = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    adj = torch.sparse_coo_tensor(
        torch.stack([row, col]), weights, (num_nodes, num_nodes)
    ).coalesce()
    return adj
    

class PolyConv(nn.Module):
    def __init__(self, theta):
        super().__init__()
        self._theta = theta
        self._k = len(theta)

    def forward(self, norm_adj, feat):
        """norm_adj: precomputed D^{-1/2} A D^{-1/2} sparse tensor"""
        h = self._theta[0] * feat
        for k in range(1, self._k):
            # L_norm @ feat = feat - D^{-1/2} A D^{-1/2} @ feat
            feat = feat - torch.sparse.mm(norm_adj, feat)
            h += self._theta[k] * feat
        return h


def calculate_theta2(d):
    thetas = []
    x = sympy.symbols('x')
    for i in range(d+1):
        f = sympy.poly((x/2) ** i * (1 - x/2) ** (d-i) / (scipy.special.beta(i+1, d+1-i)))
        coeff = f.all_coeffs()
        inv_coeff = []
        for i in range(d+1):
            inv_coeff.append(float(coeff[d-i]))
        thetas.append(inv_coeff)
    return thetas


class BWGNN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, norm_adj, d=2, dropout=0.0):
        super().__init__()
        self.norm_adj = norm_adj
        self.dropout_rate = dropout
        self.thetas = calculate_theta2(d=d)
        self.conv = nn.ModuleList(
            [PolyConv(theta) for theta in self.thetas]
        )
        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.linear3 = nn.Linear(h_feats * len(self.conv), h_feats)
        self.linear4 = nn.Linear(h_feats, num_classes)
        self.act = nn.ReLU()

    def forward(self, in_feat):
        h = F.dropout(self.act(self.linear(in_feat)), p=self.dropout_rate, training=self.training)
        h = F.dropout(self.act(self.linear2(h)), p=self.dropout_rate, training=self.training)
        h_parts = []
        for conv in self.conv:
            h_parts.append(conv(self.norm_adj, h))
        h = F.dropout(self.act(self.linear3(torch.cat(h_parts, dim=-1))), p=self.dropout_rate, training=self.training)
        return self.linear4(h)