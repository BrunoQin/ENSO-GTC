import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import ModuleList
from torch_geometric.utils import dense_to_sparse

from layer.probably_graph_convolution_layer import PGConv, GCNConv


class _GCNLayer(nn.Module):
    def __init__(self, embedding_size, in_channels, out_channels, head, concat, dropout, add_self_loops, overlap, phy_graph, node_num, rank, k, samples):
        super(_GCNLayer, self).__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.phy_graph = phy_graph
        self.phy_edge_index, _ = dense_to_sparse(phy_graph)
        # self.phy_edge_index = self.phy_edge_index.to(self.device)
        self.fully_edge_index, _ = dense_to_sparse(torch.ones((node_num, node_num)))
        # self.fully_edge_index = self.fully_edge_index.to(self.device)
        self.ag = GCNConv(in_channels, out_channels)
        self.pg = PGConv(embedding_size, in_channels, out_channels, overlap, phy_graph, node_num, rank, k, samples)

    def forward(self, input):
        x = input[0]
        memory = input[1]
        batch_size = x.size(0)
        attention_feature = torch.stack([self.ag(x[i], self.phy_edge_index.to(x[i].device)) for i in range(batch_size)], dim=0)
        probability_feature, elbo = self.pg(x, self.fully_edge_index.to(x.device), attention_feature, memory)
        out = attention_feature + probability_feature
        return out, elbo


class GCN(nn.Module):
    def __init__(self, embedding_size, in_channels, out_channels, head, concat, dropout, add_self_loops, overlap, phy_graph, node_num, rank, k, samples, num_layers):
        super(GCN, self).__init__()
        self.GCNLayer_list = ModuleList([_GCNLayer(embedding_size, in_channels, out_channels, head, concat, dropout, add_self_loops, overlap, phy_graph, node_num, rank, k, samples) for i in range(num_layers)])

    def forward(self, input):
        x = input[0]
        memory = input[1]
        elbo = 0
        for layer in self.GCNLayer_list:
            x, layer_elbo = layer([x, memory])
            elbo += layer_elbo
        return x, elbo
