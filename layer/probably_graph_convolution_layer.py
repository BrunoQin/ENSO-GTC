import gc
import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

from torch.nn import ModuleList, Sequential, Linear, ReLU, BatchNorm1d
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import dense_to_sparse, degree, add_self_loops
from torch.distributions import Normal

from layer.bayesian_layer import BLayer
from layer.memory_couple_layer import MCLayer


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='mean')
        self.mlp = Sequential(Linear(2 * in_channels, in_channels),
                              ReLU(inplace=True),
                              Linear(in_channels, out_channels))

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, x_i, norm):
        return norm.view(-1, 1) * self.mlp(torch.cat([x_i, x_j - x_i], dim=-1))


class PGConv(MessagePassing):
    def __init__(self, embedding_size, in_channels, out_channels, overlap, phy_graph, node_num, rank, k, samples, noise_tol=.1):
        super(PGConv, self).__init__(aggr='mean')
        self.node_num = node_num
        self.gtv = GraphTotalVariation()
        self.phy_graph = phy_graph
        self.noise_tol = noise_tol
        self.samples = samples
        self.mlp = Sequential(Linear(2 * in_channels, in_channels),
                              ReLU(inplace=True),
                              Linear(in_channels, out_channels))
        self.blayer = BLayer(out_channels, 1, overlap, phy_graph, node_num, rank)
        self.mclayer = MCLayer(embedding_size, out_channels, 16, 16, out_channels)
        self.ag_out = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index, attention_feature, memory):
        batch_size = x.size(0)
        tv = 0
        output = torch.zeros(x.size()).to(x.device)
        pf = torch.zeros(x.size()).to(x.device)
        log_prior = 0
        log_post = 0
        log_like = 0

        for i in range(self.samples):
            self.propagate(edge_index, x=x, memory=memory)
            for j in range(batch_size):
                pf[j] = self.ag_out(x[j], dense_to_sparse(self.prob_graph[j])[0])
                _graph = self.prob_graph[j] + self.phy_graph.to(self.prob_graph[j].device)
                _feature = pf[j] + attention_feature[j]
                _graph, _ = dense_to_sparse(_graph)
                tv += torch.sum(self.gtv(_feature.unsqueeze(0), _graph)).squeeze(0)
            output += pf
            log_prior += self.blayer.log_prior()
            log_post += self.blayer.log_post()
            log_like += Normal(tv, self.noise_tol).log_prob(torch.Tensor(0).to(tv.device)).sum()
        elbo = (log_post - log_prior - log_like) / self.samples
        return output / self.samples, elbo

    def message(self, x_i, x_j, memory):
        edge_feature = self.mlp(torch.cat([x_i, x_j - x_i], dim=-1))
        memory = memory.unsqueeze(1)
        edge_feature = self.mclayer([edge_feature, memory])
        self.prob_graph = self.blayer(edge_feature)
        return edge_feature


class GraphTotalVariation(MessagePassing):
    def __init__(self):
        super(GraphTotalVariation, self).__init__(aggr='mean')

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        return torch.norm(x_i - x_j, p=1, dim=2).permute(1, 0)
