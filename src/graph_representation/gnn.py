import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GINConv, GraphConv


class GNN(nn.Module):
    def __init__(
        self,
        num_layer,
        emb_dim,
        node_encoder,
        drop_ratio=0.5,
        JK="last",
        residual=False,
        gnn_type="gin",
    ):

        super().__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.node_encoder = node_encoder

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        if gnn_type == "gin":
            self.mlp = nn.Sequential(
                nn.Linear(emb_dim, emb_dim),
                nn.ReLU(),
                nn.Linear(emb_dim, emb_dim),
                nn.ReLU(),
                nn.Linear(emb_dim, emb_dim),
                nn.ReLU(),
            )

        for _ in range(num_layer):
            if gnn_type == "gin":
                self.convs.append(GINConv(self.mlp, "sum"))
            elif gnn_type == "gcn":
                self.convs.append(GraphConv(emb_dim, emb_dim))
            else:
                raise ValueError(f"Undefined GNN type called {gnn_type}")

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_graph):

        nodes_embeddings = [
            self.node_encoder(batched_graph.ndata["feat"], batched_graph.ndata["depth"].view(-1))
        ]

        for layer in range(self.num_layer):

            h = self.convs[layer](batched_graph, nodes_embeddings[-1])
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            if self.residual:
                h += nodes_embeddings[layer]

            nodes_embeddings.append(h)

        if self.JK == "last":
            nodes_representation = nodes_embeddings[-1]
        elif self.JK == "sum":
            nodes_representation = sum(nodes_embeddings)

        return nodes_representation
