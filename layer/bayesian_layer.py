import gc
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Softmax, ReLU
from torch.autograd import Function
from torch.distributions import Normal


class MySign(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return (1.0 - torch.tanh(input) ** 2.0) * grad_output


class BayesianLinear(nn.Module):
    def __init__(self, input_features, output_features, prior_var=1.):
        super(BayesianLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.w_mu = nn.Parameter(torch.zeros(output_features, input_features))
        self.w_rho = nn.Parameter(torch.zeros(output_features, input_features))

        self.b_mu =  nn.Parameter(torch.zeros(output_features))
        self.b_rho = nn.Parameter(torch.zeros(output_features))

        self.prior = torch.distributions.Normal(0, prior_var)

    def forward(self, input):
        w_epsilon = Normal(0, .1).sample(self.w_mu.shape)
        w = self.w_mu + torch.log(1 + torch.exp(self.w_rho)) * w_epsilon.to(self.w_mu.device)

        b_epsilon = Normal(0, .1).sample(self.b_mu.shape)
        b = self.b_mu + torch.log(1 + torch.exp(self.b_rho)) * b_epsilon.to(self.b_mu.device)

        w_log_prior = self.prior.log_prob(w)
        b_log_prior = self.prior.log_prob(b)
        self.log_prior = torch.sum(w_log_prior) + torch.sum(b_log_prior)

        w_post = Normal(self.w_mu.data, torch.log(1 + torch.exp(self.w_rho)))
        b_post = Normal(self.b_mu.data, torch.log(1 + torch.exp(self.b_rho)))
        self.log_post = w_post.log_prob(w).sum() + b_post.log_prob(b).sum()

        return F.linear(input, w, b)


class BLayer(nn.Module):
    def __init__(self, input_features, output_features, overlap, phy_graph, node_num, rank, prior_var=1.):

        super(BLayer, self).__init__()
        self.overlap = overlap
        self.phy_graph = phy_graph
        self.node_num = node_num
        self.rank = rank
        self.input_features = input_features
        self.output_features = output_features

        self.hidden = BayesianLinear(self.input_features, 16, prior_var=prior_var)
        self.out = BayesianLinear(16, self.output_features, prior_var=prior_var)
        self.sftm = Softmax(dim=1)
        self.sign = MySign.apply
        self.relu1 = ReLU(inplace=True)
        self.relu2 = ReLU(inplace=True)

    def forward(self, input):
        batch_size = input.size(0)

        prob_graph = self.relu2(self.out(self.relu1(self.hidden(input))))
        prob_graph = self.sftm(prob_graph.view(batch_size, self.node_num, self.node_num))

        if self.overlap:
            prob_graph = prob_graph * torch.stack([1 - self.phy_graph.to(prob_graph.device)] * batch_size, dim=0)

        topk, indices = torch.topk(prob_graph, self.rank, dim=2, sorted=True)
        sign = self.sign(prob_graph - topk[:, :, -1].unsqueeze(2))
        prob_graph = torch.relu(sign)

        return prob_graph

    def log_prior(self):
        return self.hidden.log_prior + self.out.log_prior

    def log_post(self):
        return self.hidden.log_post + self.out.log_post


if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = BNNModel(100, 1, False, None, 10, 3).to(device)
    x = torch.rand(4, 100, 100).to(device)
    y = model(x)
