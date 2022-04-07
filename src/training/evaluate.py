from typing import Any, Dict

import torch
from dgl.dataloading import GraphDataLoader
from ogb.graphproppred import Evaluator
from tqdm import tqdm

from src.vocab_utils import decode_arr_to_name_seq


def evaluate(
    model: torch.nn.Module,
    device: str,
    loader: GraphDataLoader,
    evaluator: Evaluator,
    idx2vocab: Dict[int, str],
) -> Any:
    model.eval()
    evaluator_input_dict = empty_imput_dict()

    for batch in tqdm(loader, mininterval=15):
        batched_graph, labels = batch
        batched_graph = batched_graph.to(device)

        with torch.no_grad():
            pred_list = model(batched_graph)

        expand_input_dict(evaluator_input_dict, pred_list, labels, idx2vocab)

    return evaluator.eval(evaluator_input_dict)


def empty_imput_dict():
    return {"seq_ref": [], "seq_pred": []}


def expand_input_dict(input_dict, pred_list, labels, idx2vocab):
    mat = []
    for pred in pred_list:
        mat.append(torch.argmax(pred, dim=1).view(-1, 1))
    mat = torch.cat(mat, dim=1)

    seq_pred = [decode_arr_to_name_seq(arr, idx2vocab) for arr in mat]

    input_dict["seq_ref"].extend(labels)
    input_dict["seq_pred"].extend(seq_pred)