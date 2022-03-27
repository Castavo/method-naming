from typing import Tuple

import torch
from torch.nn import Linear, ReLU, Sequential, Tanh
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops


class CombC(MessagePassing):
    def __init__(self, hidden, **kwargs):
        super(CombC, self).__init__(aggr="add", **kwargs)

        self.fea_mlp = Sequential(Linear(hidden, hidden), ReLU(), Linear(hidden, hidden), ReLU())

        self.aggr_mlp = Sequential(Linear(hidden * 2, hidden), Tanh())

        # edge_attr is two dimensional after augment_edge transformation
        self.edge_encoder = torch.nn.Linear(2, hidden)

    def forward(self, x: torch.TensorType, edge_index: Tuple[int, int], edge_attr):
        edge_attr = self.edge_encoder(edge_attr)
        edge_index, _ = remove_self_loops(edge_index)
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out

    def message(self, x_i, x_j, edge_attr):
        xe = x_j + edge_attr
        aggr_emb = self.aggr_mlp(torch.cat([x_i, xe], dim=-1))
        return self.fea_mlp(aggr_emb * xe)

    def update(self, aggr_out, x):
        root_emb = self.aggr_mlp(torch.cat([x, x], dim=-1))
        return aggr_out + self.fea_mlp(root_emb * x)

    def __repr__(self):
        return self.__class__.__name__


class CombC_star(MessagePassing):
    def __init__(self, hidden, **kwargs):
        super(CombC_star, self).__init__(aggr="add", **kwargs)

        self.fea_mlp = Sequential(Linear(hidden, hidden), ReLU(), Linear(hidden, hidden), ReLU())

        self.aggr_mlp = Sequential(Linear(hidden * 2, hidden), Tanh())

        # edge_attr is two dimensional after augment_edge transformation
        self.edge_encoder = torch.nn.Linear(2, hidden)

    def forward(self, x: torch.TensorType, edge_index: Tuple[int, int], edge_attr):
        edge_attr = self.edge_encoder(edge_attr)
        edge_index, _ = remove_self_loops(edge_index)
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        out = self.fea_mlp(self.propagate(edge_index, x=x, edge_attr=edge_attr))
        return out

    def message(self, x_i, x_j, edge_attr):
        xe = x_j + edge_attr
        aggr_emb = self.aggr_mlp(torch.cat([x_i, xe], dim=-1))
        return aggr_emb * xe

    def update(self, aggr_out, x):
        root_emb = self.aggr_mlp(torch.cat([x, x], dim=-1))
        return aggr_out + root_emb * x

    def __repr__(self):
        return self.__class__.__name__
