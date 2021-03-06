from dgl.nn.pytorch.glob import AvgPooling, GlobalAttentionPooling, MaxPooling, SumPooling
from torch import nn

from src.graph_representation import GNN, ASTNodeEncoder


class MethodNamePredictor(nn.Module):
    def __init__(
        self,
        num_vocab: int,
        max_seq_len: int,
        node_encoder: ASTNodeEncoder,
        num_layers=5,
        emb_dim=300,
        gnn_type="gin",
        residual=False,
        drop_ratio=0.5,
        JK="last",  # Jumping Knowledge
        graph_pooling="mean",
        virtual_node: bool = False,
    ):
        """
        num_tasks (int): number of labels to be predicted
        virtual_node (bool): whether to add virtual node or not
        """

        super().__init__()

        if num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(
            num_layers,
            emb_dim,
            node_encoder,
            JK=JK,
            drop_ratio=drop_ratio,
            residual=residual,
            gnn_type=gnn_type,
            virtual_node=virtual_node,
        )

        if graph_pooling == "sum":
            self.pool = SumPooling()
        elif graph_pooling == "mean":
            self.pool = AvgPooling()
        elif graph_pooling == "max":
            self.pool = MaxPooling()
        elif graph_pooling == "attention":
            self.pool = GlobalAttentionPooling(
                nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, 1))
            )
        else:
            raise ValueError("Invalid graph pooling type.")

        self.predict_method_name = nn.ModuleList(
            [nn.Linear(emb_dim, num_vocab) for _ in range(max_seq_len)]
        )

    def forward(self, batched_data):
        """
        Return:
            A list of predictions.
            i-th element represents prediction at i-th position of the sequence.
        """

        node_features = self.gnn(batched_data)

        graph_representations = self.pool(batched_data, node_features)

        pred_list = [predictor(graph_representations) for predictor in self.predict_method_name]

        return pred_list
