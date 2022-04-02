import dgl
import torch
from dgl.dataloading import GraphDataLoader
from torch.optim import Optimizer
from tqdm import tqdm

from src.data_loaders import augment_edge
from src.vocab_utils import labels_to_tensor


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
    for batch in tqdm(loader, mininterval=30):
        batched_graph, labels = batch
        graphs = dgl.unbatch(batched_graph)
        for graph in graphs:
            augment_edge(graph)
        batched_graph = dgl.batch(graphs)
        batched_graph = batched_graph.to(device)

        labels = labels_to_tensor(labels, vocab2idx, max_seq_len).to(device)

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
