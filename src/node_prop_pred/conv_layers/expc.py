from typing import Tuple

import torch
from torch.nn import Linear, ReLU, Sequential, Tanh
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops


class ExpC(MessagePassing):
    def __init__(self, hidden, num_aggr, **kwargs):
        super(ExpC, self).__init__(aggr="add", **kwargs)
        self.hidden = hidden
        self.num_aggr = num_aggr

        self.fea_mlp = Sequential(
            Linear(hidden * self.num_aggr, hidden), ReLU(), Linear(hidden, hidden), ReLU()
        )

        self.aggr_mlp = Sequential(Linear(hidden * 2, self.num_aggr), Tanh())
        self.edge_encoder = torch.nn.Linear(2, hidden)

    def forward(self, x: torch.TensorType, edge_index: Tuple[int, int], edge_attr):
        edge_attr = self.edge_encoder(edge_attr)
        edge_index, _ = remove_self_loops(edge_index)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out

    def message(self, x_i, x_j, edge_attr):
        xe = x_j + edge_attr
        aggr_emb = self.aggr_mlp(torch.cat([x_i, xe], dim=-1))
        feature2d = (
            torch.matmul(aggr_emb.unsqueeze(-1), xe.unsqueeze(-1).transpose(-1, -2))
            .squeeze(-1)
            .view(-1, self.hidden * self.num_aggr)
        )
        return self.fea_mlp(feature2d)

    def update(self, aggr_out, x):
        root_emb = self.aggr_mlp(torch.cat([x, x], dim=-1))
        feature2d = (
            torch.matmul(root_emb.unsqueeze(-1), x.unsqueeze(-1).transpose(-1, -2))
            .squeeze(-1)
            .view(-1, self.hidden * self.num_aggr)
        )
        return aggr_out + self.fea_mlp(feature2d)

    def __repr__(self):
        return self.__class__.__name__


class ExpC_star(MessagePassing):
    def __init__(self, hidden, num_aggr, **kwargs):
        super(ExpC_star, self).__init__(aggr="add", **kwargs)
        self.hidden = hidden
        self.num_aggr = num_aggr

        self.fea_mlp = Sequential(
            Linear(hidden * self.num_aggr, hidden), ReLU(), Linear(hidden, hidden), ReLU()
        )

        self.aggr_mlp = Sequential(Linear(hidden * 2, self.num_aggr), Tanh())
        self.edge_encoder = torch.nn.Linear(2, hidden)

    def forward(self, x: torch.TensorType, edge_index: Tuple[int, int], edge_attr):
        edge_attr = self.edge_encoder(edge_attr)
        edge_index, _ = remove_self_loops(edge_index)
        out = self.fea_mlp(self.propagate(edge_index, x=x, edge_attr=edge_attr))
        return out

    def message(self, x_i, x_j, edge_attr):
        xe = x_j + edge_attr
        aggr_emb = self.aggr_mlp(torch.cat([x_i, xe], dim=-1))
        feature2d = (
            torch.matmul(aggr_emb.unsqueeze(-1), xe.unsqueeze(-1).transpose(-1, -2))
            .squeeze(-1)
            .view(-1, self.hidden * self.num_aggr)
        )
        return feature2d

    def update(self, aggr_out, x):
        root_emb = self.aggr_mlp(torch.cat([x, x], dim=-1))
        feature2d = (
            torch.matmul(root_emb.unsqueeze(-1), x.unsqueeze(-1).transpose(-1, -2))
            .squeeze(-1)
            .view(-1, self.hidden * self.num_aggr)
        )
        return aggr_out + feature2d

    def __repr__(self):
        return self.__class__.__name__
