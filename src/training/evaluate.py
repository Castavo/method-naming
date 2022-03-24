import torch
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from typing import Dict, Any
from src.vocab_utils import decode_arr_to_name_seq
from ogb.graphproppred import Evaluator


def evaluate(
    model: torch.nn.Module,
    device: str,
    loader: DataLoader,
    evaluator: Evaluator,
    idx2vocab: Dict[int, str],
) -> Any:
    model.eval()
    seq_ref_list = []
    seq_pred_list = []

    for _, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred_list = model(batch)

            mat = []
            for i in range(len(pred_list)):
                mat.append(torch.argmax(pred_list[i], dim=1).view(-1, 1))
            mat = torch.cat(mat, dim=1)

            seq_pred = [decode_arr_to_name_seq(arr, idx2vocab) for arr in mat]

            seq_ref = [batch.y[i] for i in range(len(batch.y))]

            seq_ref_list.extend(seq_ref)
            seq_pred_list.extend(seq_pred)

    input_dict = {"seq_ref": seq_ref_list, "seq_pred": seq_pred_list}

    return evaluator.eval(input_dict)
