from typing import Any, Dict

import torch
from dgl.dataloading import GraphDataLoader
from ogb.graphproppred import Evaluator
from tqdm import tqdm

from src.vocab_utils import decode_arr_to_name_seq
from src.training.train_epoch import train_epoch


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

    for _, batch in enumerate(tqdm(loader)):
        batched_graph, labels = batch
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
