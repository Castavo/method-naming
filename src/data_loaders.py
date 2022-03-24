import numpy as np
import torch
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader
from torchvision import transforms

from src.vocab_utils import encode_y_to_arr, get_vocab_mapping


def get_data_loaders(
    dataset: PygGraphPropPredDataset,
    batch_size: int,
    max_seq_len: int,
    random_split: bool,
    num_vocab: int,
    num_workers: int,
):

    seq_len_list = np.array([len(seq) for seq in dataset.data.y])
    print(
        f"Target sequence less or equal to {max_seq_len} is "
        f"{round(100 * np.sum(seq_len_list <= max_seq_len) / len(seq_len_list), 3)}% of the dataset."
    )

    split_idx = dataset.get_idx_split()

    if random_split:
        print("Using random split")
        perm = torch.randperm(len(dataset))
        num_train, num_valid, num_test = (
            len(split_idx["train"]),
            len(split_idx["valid"]),
            len(split_idx["test"]),
        )
        split_idx["train"] = perm[:num_train]
        split_idx["valid"] = perm[num_train : num_train + num_valid]
        split_idx["test"] = perm[num_train + num_valid :]

        assert len(split_idx["train"]) == num_train
        assert len(split_idx["valid"]) == num_valid
        assert len(split_idx["test"]) == num_test

    vocab2idx, idx2vocab = get_vocab_mapping(
        [dataset.data.y[i] for i in split_idx["train"]], num_vocab
    )

    ### set the transform function
    # augment_edge: add next-token edge as well as inverse edges. add edge attributes.
    # encode_y_to_arr: add y_arr to PyG data object, indicating the array representation of a sequence.
    dataset.transform = transforms.Compose(
        [augment_edge, lambda data: encode_y_to_arr(data, vocab2idx, max_seq_len)]
    )

    train_loader = DataLoader(
        dataset[split_idx["train"]],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    valid_loader = DataLoader(
        dataset[split_idx["valid"]],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        dataset[split_idx["test"]],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, valid_loader, test_loader, vocab2idx, idx2vocab


def augment_edge(data: PygGraphPropPredDataset):
    """
    Input:
        data: PyG data object
    Output:
        data (edges are augmented in the following ways):
            data.edge_index: Added next-token edge. The inverse edges were also added.
            data.edge_attr (torch.Long):
                data.edge_attr[:,0]: whether it is AST edge (0) for next-token edge (1)
                data.edge_attr[:,1]: whether it is original direction (0) or inverse direction (1)
    """

    ##### AST edge
    edge_index_ast = data.edge_index
    edge_attr_ast = torch.zeros((edge_index_ast.size(1), 2))

    ##### Inverse AST edge
    edge_index_ast_inverse = torch.stack([edge_index_ast[1], edge_index_ast[0]], dim=0)
    edge_attr_ast_inverse = torch.cat(
        [
            torch.zeros(edge_index_ast_inverse.size(1), 1),
            torch.ones(edge_index_ast_inverse.size(1), 1),
        ],
        dim=1,
    )

    ##### Next-token edge

    attributed_node_idx_in_dfs_order = torch.where(
        data.node_is_attributed.view(
            -1,
        )
        == 1
    )[0]

    ## build next token edge
    # Given: attributed_node_idx_in_dfs_order
    #        [1, 3, 4, 5, 8, 9, 12]
    # Output:
    #    [[1, 3, 4, 5, 8, 9]
    #     [3, 4, 5, 8, 9, 12]
    edge_index_nextoken = torch.stack(
        [attributed_node_idx_in_dfs_order[:-1], attributed_node_idx_in_dfs_order[1:]], dim=0
    )
    edge_attr_nextoken = torch.cat(
        [torch.ones(edge_index_nextoken.size(1), 1), torch.zeros(edge_index_nextoken.size(1), 1)],
        dim=1,
    )

    ##### Inverse next-token edge
    edge_index_nextoken_inverse = torch.stack(
        [edge_index_nextoken[1], edge_index_nextoken[0]], dim=0
    )
    edge_attr_nextoken_inverse = torch.ones((edge_index_nextoken.size(1), 2))

    data.edge_index = torch.cat(
        [edge_index_ast, edge_index_ast_inverse, edge_index_nextoken, edge_index_nextoken_inverse],
        dim=1,
    )
    data.edge_attr = torch.cat(
        [edge_attr_ast, edge_attr_ast_inverse, edge_attr_nextoken, edge_attr_nextoken_inverse],
        dim=0,
    )

    return data
