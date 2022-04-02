from typing import Dict, List, Tuple
from ogb.graphproppred import PygGraphPropPredDataset
import numpy as np
import torch


def get_vocab_mapping(
    name_seq_list: List[List[str]], num_vocab: int
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Input:
        name_seq_list: a list of sequences
        num_vocab: vocabulary size
    Output:
        vocab2idx:
            A dictionary that maps vocabulary into integer index.
            Additioanlly, we also index '__UNK__' and '__EOS__'
            '__UNK__' : out-of-vocabulary term
            '__EOS__' : end-of-sentence

        idx2vocab:
            A list that maps idx to actual vocabulary.

    """

    vocab_cnt = {}
    vocab_list = []
    for name_seq in name_seq_list:
        for w in name_seq:
            if w in vocab_cnt:
                vocab_cnt[w] += 1
            else:
                vocab_cnt[w] = 1
                vocab_list.append(w)

    cnt_list = np.array([vocab_cnt[w] for w in vocab_list])
    topvocab = np.argsort(-cnt_list, kind="stable")[:num_vocab]

    print("Coverage of top {} vocabulary:".format(num_vocab))
    print(float(np.sum(cnt_list[topvocab])) / np.sum(cnt_list))

    vocab2idx = {vocab_list[vocab_idx]: idx for idx, vocab_idx in enumerate(topvocab)}
    idx2vocab = [vocab_list[vocab_idx] for vocab_idx in topvocab]

    # print(topvocab)
    # print([vocab_list[v] for v in topvocab[:10]])
    # print([vocab_list[v] for v in topvocab[-10:]])

    vocab2idx["__UNK__"] = num_vocab
    idx2vocab.append("__UNK__")

    vocab2idx["__EOS__"] = num_vocab + 1
    idx2vocab.append("__EOS__")

    # test the correspondence between vocab2idx and idx2vocab
    for idx, vocab in enumerate(idx2vocab):
        assert idx == vocab2idx[vocab]

    # test that the idx of '__EOS__' is len(idx2vocab) - 1.
    # This fact will be used in decode_arr_to_seq, when finding __EOS__
    assert vocab2idx["__EOS__"] == len(idx2vocab) - 1

    return vocab2idx, idx2vocab


def encode_y_to_arr(
    data: PygGraphPropPredDataset, vocab2idx: Dict[str, int], max_seq_len: int
) -> PygGraphPropPredDataset:
    """
    Input:
        data: PyG graph object
        output: add y_arr to data
    """

    name_seq = data.y
    data.y_arr = encode_name_seq_to_arr(name_seq, vocab2idx, max_seq_len)

    return data


def encode_name_seq_to_arr(
    name_seq: List[str], vocab2idx: Dict[str, int], max_seq_len: int
) -> torch.TensorType:
    """
    Input:
        seq: A list of words
        output: add y_arr (torch.Tensor)
    """

    augmented_seq = name_seq[:max_seq_len] + ["__EOS__"] * max(0, max_seq_len - len(name_seq))
    return torch.tensor(
        [[vocab2idx[w] if w in vocab2idx else vocab2idx["__UNK__"] for w in augmented_seq]],
        dtype=torch.long,
    )


def decode_arr_to_name_seq(arr: List[int], idx2vocab: Dict[int, str]) -> List[int]:
    """
    Input: torch 1d array: y_arr
    Output: a sequence of words.
    """
    try:
        eos_index = arr.index(len(idx2vocab) - 1)
        cropped_arr = arr[:eos_index]
    except ValueError:
        cropped_arr = arr

    return list(map(lambda x: idx2vocab[x], cropped_arr.cpu()))
