import json
import os

import numpy as np
from dgl import load_graphs
from dgl.dataloading import GraphDataLoader
from ogb.graphproppred import collate_dgl

from src.vocab_utils import get_vocab_mapping


def get_data_loaders(
    data_path: str,
    batch_size: int,
    max_seq_len: int,
    num_vocab: int,
    num_workers: int,
):

    data_loaders = {}
    seq_len_list = []
    for partition in ["train", "valid", "test"]:
        graphs, _ = load_graphs(os.path.join(data_path, partition + ".bin"))
        labels = json.load(open(os.path.join(data_path, partition + ".json")))
        data_loaders[partition] = GraphDataLoader(
            list(zip(graphs, labels)),
            batch_size=batch_size,
            shuffle=(partition == "train"),
            num_workers=num_workers,
            collate_fn=collate_dgl,
        )

        seq_len_list += [len(seq) for seq in labels]

        if partition == "train":
            vocab2idx, idx2vocab = get_vocab_mapping(labels, num_vocab)

    seq_len_list = np.array(seq_len_list)
    print(
        f"Target sequence less or equal to {max_seq_len} is "
        f"{round(100 * np.sum(seq_len_list <= max_seq_len) / len(seq_len_list), 3)}% of the dataset."
    )

    return (
        data_loaders["train"],
        data_loaders["valid"],
        data_loaders["test"],
        vocab2idx,
        idx2vocab,
    )
