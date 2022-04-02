import torch
from src.graph_representation import GNN, ASTNodeEncoder


class MethodNamePredictor(torch.nn.Module):
    def __init__(
        self,
        num_vocab: int,
        max_seq_len: int,
        node_encoder: ASTNodeEncoder,
        num_layer=5,
        emb_dim=300,
        gnn_type="gin",
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
        self.emb_dim = emb_dim
        self.num_vocab = num_vocab
        self.max_seq_len = max_seq_len
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(
            num_layer,
            emb_dim,
            node_encoder,
            JK=JK,
            drop_ratio=drop_ratio,
            residual=residual,
            gnn_type=gnn_type,
        )

        self.predict_method_name = torch.nn.ModuleList(
            [torch.nn.Linear(emb_dim, num_vocab) for _ in range(max_seq_len)]
        )

    def forward(self, batched_data):
        """
        Return:
            A list of predictions.
            i-th element represents prediction at i-th position of the sequence.
        """

        graph_representation = self.gnn(batched_data)

        pred_list = [predictor(graph_representation) for predictor in self.predict_method_name]

        return pred_list