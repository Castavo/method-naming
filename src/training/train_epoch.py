import torch
from dgl.dataloading import GraphDataLoader
from ogb.graphproppred import Evaluator
from torch.optim import Optimizer
from tqdm import tqdm

from src.vocab_utils import labels_to_tensor
from src.training.evaluate import empty_imput_dict, expand_input_dict

def train_epoch(
    model: torch.nn.Module,
    device: str,
    loader: GraphDataLoader,
    evaluator: Evaluator,
    optimizer: Optimizer,
    criterion,
    vocab2idx: dict,
    idx2vocab: dict,
    max_seq_len: int,
) -> None:
    model.train()

    loss_accum = 0
    evaluator_input_dict = empty_imput_dict()
    for batch in tqdm(loader, mininterval=30):
        batched_graph, labels = batch
        batched_graph = batched_graph.to(device)

        tensor_labels = labels_to_tensor(labels, vocab2idx, max_seq_len).to(device)

        pred_list = model(batched_graph)
        expand_input_dict(evaluator_input_dict, pred_list, labels, idx2vocab)

        optimizer.zero_grad()
        loss = 0
        for i, _ in enumerate(pred_list):
            loss += criterion(pred_list[i].to(torch.float32), tensor_labels[:, i])

        loss = loss / len(pred_list)

        loss.backward()
        optimizer.step()

        loss_accum += loss.item()

    print(f"Average training loss: {loss_accum / len(loader) }")
    return evaluator.eval(evaluator_input_dict)