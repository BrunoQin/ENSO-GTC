import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Linear, ReLU, Sigmoid


class MCLayer(nn.Module):
    def __init__(self, embedding_size, in_features, emb_features, hid_features, out_features):
        super(MCLayer, self).__init__()

        self.embedding = Linear(embedding_size, emb_features)

        self.reset_gate_x = nn.Linear(in_features, hid_features, bias=True)
        self.reset_gate_m = nn.Linear(emb_features, hid_features, bias=True)

        self.update_gate_x = nn.Linear(in_features, hid_features, bias=True)
        self.update_gate_m = nn.Linear(emb_features, hid_features, bias=True)

        self.select_gate_x = nn.Linear(in_features, hid_features, bias=True)
        self.select_gate_m = nn.Linear(emb_features, hid_features, bias=True)

        self.output_gate = nn.Linear(hid_features, out_features, bias=True)

    def forward(self, input):
        x = input[0]
        memory = input[1]

        memory = ReLU(inplace=True)(self.embedding(memory))
        Z = Sigmoid()(self.reset_gate_x(x) + self.reset_gate_m(memory))
        R = Sigmoid()(self.update_gate_x(x) + self.update_gate_m(memory))
        m_tilda = torch.tanh(self.select_gate_x(x) + self.select_gate_m(R * memory))
        H = Z * memory + (1 - Z) * m_tilda
        x = self.output_gate(H)

        return x
