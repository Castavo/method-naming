import torch
import torch.nn.functional as F

from legacy.node_prop_pred.conv_layers.combc import CombC, CombC_star
from legacy.node_prop_pred.conv_layers.expc import ExpC, ExpC_star
from legacy.node_prop_pred.conv_layers.gin import GINConv
from legacy.node_prop_pred.conv_layers.gcn import GCNConv


### GNN to generate node embedding
class GNN_node_prop(torch.nn.Module):
    """
    Output:
        node representations
    """

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
        """
        emb_dim (int): node embedding dimensionality
        num_layer (int): number of GNN message passing layers

        """

        super(GNN_node_prop, self).__init__()
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

        for _ in range(num_layer):
            if gnn_type == "gin":
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == "gcn":
                self.convs.append(GCNConv(emb_dim))
            elif gnn_type[:2] == "EB":
                self.convs.append(ExpC(emb_dim, int(gnn_type[2:])))
            elif gnn_type[:2] == "EA":
                self.convs.append(ExpC_star(emb_dim, int(gnn_type[2:])))
            elif gnn_type == "CB":
                self.convs.append(CombC(emb_dim))
            elif gnn_type == "CA":
                self.convs.append(CombC_star(emb_dim))
            else:
                raise ValueError(f"Undefined GNN type called {gnn_type}")

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data):
        x, edge_index, edge_attr, node_depth, batch = (
            batched_data.x,
            batched_data.edge_index,
            batched_data.edge_attr,
            batched_data.node_depth,
            batched_data.batch,
        )

        ### computing input node embedding

        h_list = [
            self.node_encoder(
                x,
                node_depth.view(
                    -1,
                ),
            )
        ]
        for layer in range(self.num_layer):

            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        return node_representation
