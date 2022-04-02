import numpy as np
import torch
from dgl import DGLGraph
from dgl.dataloading import GraphDataLoader
from ogb.graphproppred import DglGraphPropPredDataset, collate_dgl

from src.vocab_utils import get_vocab_mapping


def get_data_loaders(
    dataset: DglGraphPropPredDataset,
    batch_size: int,
    max_seq_len: int,
    random_split: bool,
    num_vocab: int,
    num_workers: int,
):

    seq_len_list = np.array([len(seq) for seq in dataset.labels])
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
        [dataset.labels[i] for i in split_idx["train"]], num_vocab
    )

    # augment_edge: add next-token edge as well as inverse edges. add edge attributes.
    # encode_y_to_arr: add y_arr to Dgl data object, indicating the array representation of a sequence.
    # dataset = preprocess_dataset(dataset, [augment_edge, lambda data: encode_y_to_arr(data, vocab2idx, max_seq_len)])

    for graph, _ in dataset:
        augment_edge(graph)

    train_loader = GraphDataLoader(
        dataset[split_idx["train"]],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_dgl,
    )
    valid_loader = GraphDataLoader(
        dataset[split_idx["valid"]],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_dgl,
    )
    test_loader = GraphDataLoader(
        dataset[split_idx["test"]],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_dgl,
    )

    return train_loader, valid_loader, test_loader, vocab2idx, idx2vocab


def augment_edge(graph: DGLGraph):
    """
    Input:
        graph: DglGraph object
    Output:
        Modifies the graph inplace by adding the following edges and their attributes:
            graph.edges: Added next-token edge. The inverse edges were also added.
            graph.edata["attr"] (torch.long):
                graph.edata["attr"][:,0]: whether it is AST edge (0) or next-token edge (1)
                graph.edata["attr"][:,1]: whether it is original direction (0) or inverse direction (1)
    """

    ##### AST edge
    edge_index_ast = graph.all_edges()
    edge_attr_ast = torch.zeros((edge_index_ast[0].size(0), 2))

    graph.remove_edges(graph.all_edges("eid"))

    graph.add_edges(edge_index_ast[0], edge_index_ast[1], {"attr": edge_attr_ast})

    ##### Inverse AST edge
    edge_index_ast_inverse = (edge_index_ast[1], edge_index_ast[0])
    edge_attr_ast_inverse = torch.cat(
        [
            torch.zeros(edge_index_ast_inverse[0].size(0), 1),
            torch.ones(edge_index_ast_inverse[0].size(0), 1),
        ],
        dim=1,
    )

    graph.add_edges(
        edge_index_ast_inverse[0], edge_index_ast_inverse[1], {"attr": edge_attr_ast_inverse}
    )

    ##### Next-token edge
    attributed_node_idx_in_dfs_order = torch.where(graph.ndata["is_attributed"].view(-1) == 1)[0]

    ## build next token edge
    # Given: attributed_node_idx_in_dfs_order
    #        [1, 3, 4, 5, 8, 9, 12]
    # Output:
    #    [[1, 3, 4, 5, 8, 9]
    #     [3, 4, 5, 8, 9, 12]
    edge_index_nextoken = (
        attributed_node_idx_in_dfs_order[:-1],
        attributed_node_idx_in_dfs_order[1:],
    )
    edge_attr_nextoken = torch.cat(
        [
            torch.ones(edge_index_nextoken[0].size(0), 1),
            torch.zeros(edge_index_nextoken[0].size(0), 1),
        ],
        dim=1,
    )

    graph.add_edges(edge_index_nextoken[0], edge_index_nextoken[1], {"attr": edge_attr_nextoken})

    ##### Inverse next-token edge
    edge_index_nextoken_inverse = (edge_index_nextoken[1], edge_index_nextoken[0])
    edge_attr_nextoken_inverse = torch.ones((edge_index_nextoken[0].size(0), 2))

    graph.add_edges(
        edge_index_nextoken_inverse[0],
        edge_index_nextoken_inverse[1],
        {"attr": edge_attr_nextoken_inverse},
    )
