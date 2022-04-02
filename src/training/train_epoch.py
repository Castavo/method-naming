import torch
from dgl.dataloading import GraphDataLoader
from torch.optim import Optimizer
from tqdm import tqdm
import dgl

from src.vocab_utils import labels_to_tensor
from src.data_loaders import augment_edge


def train_epoch(
    model: torch.nn.Module,
    device: str,
    loader: GraphDataLoader,
    optimizer: Optimizer,
    criterion,
    vocab2idx: dict,
    max_seq_len: int,
) -> None:
    model.train()

    loss_accum = 0
    for batch in tqdm(loader):
        batched_graph, labels = batch
        graphs = dgl.unbatch(batched_graph)
        for graph in graphs:
            augment_edge(graph)
        batched_graph = dgl.batch(graphs)
        batched_graph = batched_graph.to(device)

        labels = labels_to_tensor(labels, vocab2idx, max_seq_len)

        pred_list = model(batched_graph)
        optimizer.zero_grad()

        loss = 0
        for i, _ in enumerate(pred_list):
            loss += criterion(pred_list[i].to(torch.float32), labels[:, i])

        loss = loss / len(pred_list)

        loss.backward()
        optimizer.step()

        loss_accum += loss.item()

    print(f"Average training loss: {loss_accum / len(loader) }")
