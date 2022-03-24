import torch
from torch_geometric.nn import (
    GlobalAttention,
    Set2Set,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

from src.node_prop_pred import GNN_node_prop, GNNVirtual_node_prop, ASTNodeEncoder


class GNN_graph_prop(torch.nn.Module):
    def __init__(
        self,
        num_vocab: int,
        max_seq_len: int,
        node_encoder: ASTNodeEncoder,
        num_layer=5,
        emb_dim=300,
        gnn_type="gin",
        virtual_node=True,
        residual=False,
        drop_ratio=0.5,
        JK="last",
        graph_pooling="mean",
    ):
        """
        num_tasks (int): number of labels to be predicted
        virtual_node (bool): whether to add virtual node or not
        """

        super().__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        # JK c'est comment on produit le vecteur pour le graphe entier à la fin, je crois
        self.emb_dim = emb_dim
        self.num_vocab = num_vocab
        self.max_seq_len = max_seq_len
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn_node_prop = GNNVirtual_node_prop(
                num_layer,
                emb_dim,
                node_encoder,
                JK=JK,
                drop_ratio=drop_ratio,
                residual=residual,
                gnn_type=gnn_type,
            )
        else:
            self.gnn_node_prop = GNN_node_prop(
                num_layer,
                emb_dim,
                node_encoder,
                JK=JK,
                drop_ratio=drop_ratio,
                residual=residual,
                gnn_type=gnn_type,
            )

        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(
                gate_nn=torch.nn.Sequential(
                    torch.nn.Linear(emb_dim, 2 * emb_dim),
                    torch.nn.BatchNorm1d(2 * emb_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(2 * emb_dim, 1),
                )
            )
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")

        self.graph_pred_linear_list = torch.nn.ModuleList()

        if graph_pooling == "set2set":
            for _ in range(max_seq_len):
                self.graph_pred_linear_list.append(torch.nn.Linear(2 * emb_dim, self.num_vocab))

        else:
            for _ in range(max_seq_len):
                self.graph_pred_linear_list.append(torch.nn.Linear(emb_dim, self.num_vocab))

    def forward(self, batched_data):
        """
        Return:
            A list of predictions.
            i-th element represents prediction at i-th position of the sequence.
        """

        h_node_prop = self.gnn_node_prop(batched_data)

        h_graph_prop = self.pool(h_node_prop, batched_data.batch)

        pred_list = []

        for i in range(self.max_seq_len):
            pred_list.append(self.graph_pred_linear_list[i](h_graph_prop))

        return pred_list
