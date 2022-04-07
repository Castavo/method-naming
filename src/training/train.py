import argparse
import json
import os
from copy import deepcopy

import matplotlib.pyplot as plt
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


def train(
    batch_size: int,
    num_epochs: int,
    max_seq_len: int,
    random_split: bool,
    num_vocab: int,
    emb_dim: int,
    gnn_type: str,
    num_layers: int,
    drop_ratio: float,
    num_workers: int,
    device: str,
    model_path: str,
    results_path: str,
    progress_path: str,
    data_path: str,
):
    ### automatic dataloading and splitting
    dataset = DglGraphPropPredDataset(name="ogbg-code2", root=data_path)

    train_loader, valid_loader, test_loader, vocab2idx, idx2vocab = get_data_loaders(
        dataset=dataset,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        random_split=random_split,
        num_vocab=num_vocab,
        num_workers=num_workers,
    )

    nodetypes_mapping = pd.read_csv(
        os.path.join(dataset.root, "mapping", "typeidx2type.csv.gz")
    )
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
        emb_dim,
        num_nodetypes=len(nodetypes_mapping["type"]),
        num_nodeattributes=len(nodeattributes_mapping["attr"]),
        max_depth=20,
    )

    if gnn_type in ["gin", "gin-virtual"]:
        gnn_type = "gin"
    elif gnn_type in ["gcn", "gcn-virtual"]:
        gnn_type = "gcn"
    else:
        raise ValueError("Invalid GNN type")

    # virtual_node = gnn_type in ["gin-virtual", "gcn-virtual"]

    model = MethodNamePredictor(
        num_vocab=len(vocab2idx),
        max_seq_len=max_seq_len,
        node_encoder=node_encoder,
        num_layer=num_layers,
        gnn_type=gnn_type,
        emb_dim=emb_dim,
        drop_ratio=drop_ratio,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"#Params: {sum(p.numel() for p in model.parameters())}")

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(name="ogbg-code2")

    valid_curve = []
    train_curve = []

    best_valid = -float("inf")
    best_model_state_dict = deepcopy(model.state_dict())
    if model_path:
        torch.save(best_model_state_dict, model_path)

    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, num_epochs + 1):
        print(f"=====Epoch {epoch}")
        print("Training...")
        train_perf = train_epoch(
            model=model,
            device=device,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            evaluator=evaluator,
            vocab2idx=vocab2idx,
            idx2vocab=idx2vocab,
            max_seq_len=max_seq_len,
        )

        print("Evaluating...")
        valid_perf = evaluate(model, device, valid_loader, evaluator, idx2vocab)

        print({"Train": train_perf, "Validation": valid_perf})

        train_curve.append(train_perf[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])
        if progress_path:
            progress_curve(train_curve, valid_curve, progress_path, evaluator.eval_metric)

        if valid_perf[dataset.eval_metric] > best_valid:
            best_valid = valid_perf[dataset.eval_metric]
            best_model_state_dict = deepcopy(model.state_dict())
            if model_path:
                torch.save(best_model_state_dict, model_path)

    best_val_epoch = np.argmax(np.array(valid_curve))
    best_train = max(train_curve)
    print("Finished training!")
    model.load_state_dict(best_model_state_dict)
    test_perf = evaluate(model, device, test_loader, evaluator, idx2vocab)
    print(f"Best validation score: {valid_curve[best_val_epoch]}")
    print(f"Test score: {test_perf}")

    if results_path != "":
        result_dict = {
            "Val": valid_curve[best_val_epoch],
            "Test": test_perf,
            "Train": train_curve[best_val_epoch],
            "BestTrain": best_train,
        }
        json.dump(result_dict, open(results_path, "w"), indent=4)
    
    return model


def progress_curve(train_curve, val_curve, path, eval_metric):
    plt.figure(figsize=(12, 8))
    plt.title(f"Training Curve ({eval_metric})")
    plt.plot(train_curve, label="Train")
    plt.plot(val_curve, label="Validation")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid()
    plt.savefig(path)
    plt.close()


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(
        description="GNN baselines on ogbg-code2 data with Pytorch Geometrics"
    )
    parser.add_argument(
        "--gpu", type=int, default=0, help="which gpu to use if any (default: 0)",
    )
    parser.add_argument(
        "--gnn_type",
        default="gcn-virtual",
        help="GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gcn-virtual)",
    )
    parser.add_argument(
        "--drop_ratio", type=float, default=0, help="dropout ratio (default: 0)",
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
        "--num_layers",
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
        "--num_workers", type=int, default=0, help="number of workers (default: 0)",
    )
    parser.add_argument("--data_path", default=".", help="path to store and/or fetch ogbg-code2 data")
    parser.add_argument("--results_path", default="", help="path to output results")
    parser.add_argument("--model_path", default="", help="path to save model")
    parser.add_argument(
        "--all_output_path",
        default="",
        help="path were to store the model, results and progress plots",
    )
    args = parser.parse_args()

    device = (
        torch.device("cuda:" + str(args.gpu))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    ### Prepare outputs
    if args.all_output_path:
        os.makedirs(args.all_output_path, exist_ok=True)
        model_path = os.path.join(
            args.all_output_path, os.path.basename(args.model_path) or "model.pt"
        )
        results_path = os.path.join(
            args.all_output_path, os.path.basename(args.results_path) or "results.json"
        )
        progress_path = os.path.join(args.all_output_path, "progress.png")
        config_path = os.path.join(args.all_output_path, "config.json")
        json.dump(vars(args), open(config_path, "w"), indent=4)
    else:
        model_path = args.model_path
        results_path = args.results_path
        progress_path = ""

    train(
        gnn_type=args.gnn_type,
        drop_ratio=args.drop_ratio,
        max_seq_len=args.max_seq_len,
        num_vocab=args.num_vocab,
        num_layers=args.num_layers,
        emb_dim=args.emb_dim,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        random_split=args.random_split,
        num_workers=args.num_workers,
        device=device,
        model_path=model_path,
        results_path=results_path,
        progress_path=progress_path,
        data_path=args.data_path,
    )
