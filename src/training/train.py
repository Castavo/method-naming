import argparse
import os
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from ogb.graphproppred import DglGraphPropPredDataset, Evaluator
from sklearn.metrics import roc_auc_score  # pylint: disable=unused-import

from src.data_loaders import get_data_loaders
from src.graph_representation import ASTNodeEncoder
from src.method_name_prediction import MethodNamePredictor
from src.training.evaluate import evaluate
from src.training.train_epoch import train_epoch


def train():
    # Training settings
    parser = argparse.ArgumentParser(
        description="GNN baselines on ogbg-code2 data with Pytorch Geometrics"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="which gpu to use if any (default: 0)",
    )
    parser.add_argument(
        "--gnn",
        default="gcn-virtual",
        help="GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gcn-virtual)",
    )
    parser.add_argument(
        "--drop_ratio",
        type=float,
        default=0,
        help="dropout ratio (default: 0)",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=5,
        help="maximum sequence length to predict (default: 5)",
    )
    parser.add_argument(
        "--num_vocab",
        type=int,
        default=5000,
        help="the number of vocabulary used for sequence prediction (default: 5000)",
    )
    parser.add_argument(
        "--num_layer",
        type=int,
        default=5,
        help="number of GNN message passing layers (default: 5)",
    )
    parser.add_argument(
        "--emb_dim",
        type=int,
        default=300,
        help="dimensionality of hidden units in GNNs (default: 300)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="input batch size for training (default: 128)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="number of epochs to train (default: 25)",
    )
    parser.add_argument("--random_split", action="store_true")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="number of workers (default: 0)",
    )
    parser.add_argument(
        "--filename",
        default="",
        help="filename to output result (default: )",
    )
    parser.add_argument(
        "--model_path",
        default="",
        help="path to save model (default: )",
    )
    args = parser.parse_args()

    device = (
        torch.device("cuda:" + str(args.gpu)) if torch.cuda.is_available() else torch.device("cpu")
    )

    ### automatic dataloading and splitting
    dataset = DglGraphPropPredDataset(name="ogbg-code2")

    train_loader, valid_loader, test_loader, vocab2idx, idx2vocab = get_data_loaders(
        dataset=dataset,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        random_split=args.random_split,
        num_vocab=args.num_vocab,
        num_workers=args.num_workers,
    )

    nodetypes_mapping = pd.read_csv(os.path.join(dataset.root, "mapping", "typeidx2type.csv.gz"))
    nodeattributes_mapping = pd.read_csv(
        os.path.join(dataset.root, "mapping", "attridx2attr.csv.gz")
    )

    print(nodeattributes_mapping)

    ### Encoding node features into emb_dim vectors.
    ### The following three node features are used.
    # 1. node type
    # 2. node attribute
    # 3. node depth
    node_encoder = ASTNodeEncoder(
        args.emb_dim,
        num_nodetypes=len(nodetypes_mapping["type"]),
        num_nodeattributes=len(nodeattributes_mapping["attr"]),
        max_depth=20,
    )

    if args.gnn in ["gin", "gin-virtual"]:
        gnn_type = "gin"
    elif args.gnn in ["gcn", "gcn-virtual"]:
        gnn_type = "gcn"
    else:
        raise ValueError("Invalid GNN type")

    # virtual_node = args.gnn in ["gin-virtual", "gcn-virtual"]

    model = MethodNamePredictor(
        num_vocab=len(vocab2idx),
        max_seq_len=args.max_seq_len,
        node_encoder=node_encoder,
        num_layer=args.num_layer,
        gnn_type=gnn_type,
        emb_dim=args.emb_dim,
        drop_ratio=args.drop_ratio,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"#Params: {sum(p.numel() for p in model.parameters())}")

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(name="ogbg-code2")

    valid_curve = []
    test_curve = []
    train_curve = []

    best_valid = -float("inf")
    best_model_state_dict = deepcopy(model.state_dict())
    if args.model_path:
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        torch.save(best_model_state_dict, args.model_path)

    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        print(f"=====Epoch {epoch}")
        print("Training...")
        train_epoch(
            model,
            device,
            train_loader,
            optimizer,
            criterion,
            vocab2idx=vocab2idx,
            max_seq_len=args.max_seq_len,
        )

        print("Evaluating...")
        train_perf = evaluate(model, device, train_loader, evaluator, idx2vocab)
        valid_perf = evaluate(model, device, valid_loader, evaluator, idx2vocab)
        test_perf = evaluate(model, device, test_loader, evaluator, idx2vocab)

        print({"Train": train_perf, "Validation": valid_perf, "Test": test_perf})

        train_curve.append(train_perf[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])
        test_curve.append(test_perf[dataset.eval_metric])

        if valid_perf[dataset.eval_metric] > best_valid:
            best_valid = valid_perf[dataset.eval_metric]
            best_model_state_dict = deepcopy(model.state_dict())
            if args.model_path:
                torch.save(best_model_state_dict, args.model_path)

    best_val_epoch = np.argmax(np.array(valid_curve))
    best_train = max(train_curve)
    print("Finished training!")
    print(f"Best validation score: {valid_curve[best_val_epoch]}")
    print(f"Test score: {test_curve[best_val_epoch]}")

    if not args.filename == "":
        result_dict = {
            "Val": valid_curve[best_val_epoch],
            "Test": test_curve[best_val_epoch],
            "Train": train_curve[best_val_epoch],
            "BestTrain": best_train,
        }
        torch.save(result_dict, args.filename)

    if not args.model_path == "":
        torch.save(best_model_state_dict, args.model_path)


if __name__ == "__main__":
    train()
