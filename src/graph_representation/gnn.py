import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GINConv, GraphConv
from dgl import DGLGraph


class GNN(nn.Module):
    def __init__(
        self,
        num_layer: int,
        emb_dim: int,
        node_encoder,
        drop_ratio=0.5,
        JK="last",
        residual=False,
        gnn_type="gin",
        virtual_node=False,
    ):

        super().__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK  # Jumping Knowledge
        self.virtual_node = virtual_node
        self.residual = residual  # add residual connection or not

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.node_encoder = node_encoder

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        if self.virtual_node:
            self.virtual_node_embedding = torch.nn.Parameter(torch.zeros(1, emb_dim))

        for _ in range(num_layer):
            if gnn_type == "gin":
                self.convs.append(
                    GINConv(
                        nn.Sequential(
                            nn.Linear(emb_dim, emb_dim),
                            nn.BatchNorm1d(emb_dim),
                            nn.ReLU(),
                            nn.Linear(emb_dim, emb_dim),
                            nn.BatchNorm1d(emb_dim),
                            nn.ReLU(),
                            nn.Linear(emb_dim, emb_dim),
                            nn.BatchNorm1d(emb_dim),
                            nn.ReLU(),
                        ),
                        "sum",
                    )
                )
            elif gnn_type == "gcn":
                self.convs.append(GraphConv(emb_dim, emb_dim))
            else:
                raise ValueError(f"Undefined GNN type called {gnn_type}")

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    @staticmethod
    def add_virtual_nodes(batched_graph: DGLGraph):
        n_nodes = batched_graph.number_of_nodes()
        graph_num_nodes, graph_num_edges = (
            batched_graph.batch_num_nodes(),
            batched_graph.batch_num_edges(),
        )
        virtual_nodes_idxs = []
        for i, graph_size in enumerate(graph_num_nodes):
            virtual_nodes_idxs += [n_nodes + i] * graph_size
        batched_graph.add_edges(
            np.array(virtual_nodes_idxs, np.int32), np.arange(n_nodes, dtype=np.int32)
        )
        batched_graph.add_edges(
            np.arange(n_nodes, dtype=np.int32), np.array(virtual_nodes_idxs, np.int32)
        )
        return graph_num_nodes, graph_num_edges

    @staticmethod
    def remove_virtual_nodes(batched_graph: DGLGraph, graph_num_nodes, graph_num_edges):
        n_nodes = batched_graph.number_of_nodes()
        batched_graph.remove_nodes(np.arange(n_nodes - len(graph_num_nodes), n_nodes))
        batched_graph.set_batch_num_edges(graph_num_edges)
        batched_graph.set_batch_num_nodes(graph_num_nodes)

    def forward(self, batched_graph: DGLGraph):
        nodes_embeddings = [
            self.node_encoder(batched_graph.ndata["feat"], batched_graph.ndata["depth"].view(-1))
        ]
        if self.virtual_node:
            # I have to store the batch size because it disappears when the graph is edited
            graph_num_nodes, graph_num_edges = self.add_virtual_nodes(batched_graph)
            nodes_embeddings[0] = torch.cat(
                [nodes_embeddings[0], self.virtual_node_embedding.repeat(len(graph_num_nodes), 1)],
                dim=0,
            )

        for layer in range(self.num_layer):
            h = self.convs[layer](batched_graph, nodes_embeddings[-1])
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            if self.residual:
                h += nodes_embeddings[-1]

            nodes_embeddings.append(h)

        if self.virtual_node:
            self.remove_virtual_nodes(batched_graph, graph_num_nodes, graph_num_edges)
            for i in range(self.num_layer + 1):
                nodes_embeddings[i] = nodes_embeddings[i][: -len(graph_num_nodes)]

        if self.JK == "last":
            nodes_representation = nodes_embeddings[-1]
        elif self.JK == "sum":
            nodes_representation = sum(nodes_embeddings)

        return nodes_representation
