from typing import Any, Dict

import dgl
import torch
from dgl.dataloading import GraphDataLoader
from ogb.graphproppred import Evaluator
from tqdm import tqdm

from src.data_loaders import augment_edge
from src.vocab_utils import decode_arr_to_name_seq


def evaluate(
    model: torch.nn.Module,
    device: str,
    loader: GraphDataLoader,
    evaluator: Evaluator,
    idx2vocab: Dict[int, str],
) -> Any:
    model.eval()
    seq_ref_list = []
    seq_pred_list = []

    for batch in tqdm(loader, mininterval=15):
        batched_graph, labels = batch
        graphs = dgl.unbatch(batched_graph)
        for graph in graphs:
            augment_edge(graph)
        batched_graph = dgl.batch(graphs)
        batched_graph = batched_graph.to(device)

        with torch.no_grad():
            pred_list = model(batched_graph)

        mat = []
        for pred in pred_list:
            mat.append(torch.argmax(pred, dim=1).view(-1, 1))
        mat = torch.cat(mat, dim=1)

        seq_pred = [decode_arr_to_name_seq(arr, idx2vocab) for arr in mat]

        seq_ref_list.extend(labels)
        seq_pred_list.extend(seq_pred)

    input_dict = {"seq_ref": seq_ref_list, "seq_pred": seq_pred_list}

    return evaluator.eval(input_dict)
