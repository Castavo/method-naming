import argparse
import json
import os

import torch
from dgl import DGLGraph, save_graphs
from ogb.graphproppred import DglGraphPropPredDataset
from tqdm import tqdm


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
        edge_index_ast_inverse[0],
        edge_index_ast_inverse[1],
        {"attr": edge_attr_ast_inverse},
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

    graph.set_batch_num_edges(torch.tensor([graph.number_of_edges()]))


def preprocess_dataset(data_path, dest_path):
    dataset = DglGraphPropPredDataset("ogbg-code2", data_path)
    # augment_edge: add next-token edge as well as inverse edges. add edge attributes.
    print("Augmenting edges...")

    for data in tqdm(dataset, mininterval=30):
        augment_edge(data[0])

    split_idx = dataset.get_idx_split()
    print("Saving graphs...")
    os.makedirs(dest_path, exist_ok=True)
    for partition in ["test", "train", "valid"]:
        save_graphs(
            os.path.join(dest_path, partition + ".bin"),
            [dataset.graphs[i] for i in split_idx[partition]],
        )
        json.dump(
            [dataset.labels[i] for i in split_idx[partition]],
            open(os.path.join(dest_path, partition + ".json"), "w"),
            indent=4,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess ogb code2 dataset")
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--dest_path", type=str)
    args = parser.parse_args()
    preprocess_dataset(args.data_path, args.dest_path)

    print("Done!")
