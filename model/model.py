import torch
import torch.nn as nn
import numpy as np
from torch_geometric.utils import dense_to_sparse

from model.encoder import Encoder
from model.decoder import Decoder
from model.probably_graph import GCN
from layer.spatial_temporal_memory_layer import STMLayer


class Model(nn.Module):
    def __init__(self, loc, phy_graph, sequence_length, dropout,
                       encoder_growth_rate, encoder_bn_size, encoder_theta, encoder_dense_config, encoder_pool_size,
                       embedding_size, graph_channel, head, concat, add_self_loops, overlap, rank, k, samples, num_layers,
                       decoder_init_feature, decoder_reduce_rate, decoder_bn_size, decoder_theta, decoder_reduce_config, decoder_pool_size):
        super(Model, self).__init__()

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.phy_graph = torch.Tensor(phy_graph).to(self.device)
        self.phy_edge_index, _ = dense_to_sparse(self.phy_graph)
        self.embedding_size = embedding_size
        self.sequence_length = sequence_length
        self.node_num = int(np.sum(loc) * self.sequence_length)
        self.rank = rank

        self.stm = STMLayer(in_seqlen=self.sequence_length, in_channels=self.node_num, dropout=dropout)
        self.encoder = Encoder(growth_rate=encoder_growth_rate, bn_size=encoder_bn_size, theta=encoder_theta, dense_config=encoder_dense_config, pool_size=encoder_pool_size, dropout=dropout)
        self.gcn = GCN(embedding_size=embedding_size, in_channels=graph_channel, out_channels=graph_channel, head=head, concat=concat, dropout=dropout, add_self_loops=add_self_loops, overlap=overlap, phy_graph=self.phy_graph, node_num=self.node_num, rank=rank, k=k, samples=samples, num_layers=num_layers)
        self.decoder = Decoder(init_feature=decoder_init_feature, reduce_rate=decoder_reduce_rate, bn_size=decoder_bn_size, theta=decoder_theta, reduce_config=decoder_reduce_config, pool_size=decoder_pool_size, dropout=dropout)

    def forward(self, input):

        memory = self.stm(input)

        x = self.encoder(input)
        batch_size, sxn, c, h, w = x.size()
        x = x.flatten(start_dim=2)

        x, elbo = self.gcn([x, memory])

        x = x.view(batch_size, self.sequence_length, int(sxn / self.sequence_length), c, h, w)
        x = torch.mean(x, dim=1)

        x = self.decoder(x)
        return x.squeeze(dim=2), elbo
