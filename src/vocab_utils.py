from collections import defaultdict
from typing import Dict, List, Tuple

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
            Additionally, we also index '__UNK__' and '__EOS__'
            '__UNK__' : out-of-vocabulary term
            '__EOS__' : end-of-sentence

        idx2vocab:
            A list that maps idx to actual vocabulary.

    """

    vocab_counts = defaultdict(int)
    for name_seq in name_seq_list:
        for w in name_seq:
            vocab_counts[w] += 1

    counts = np.array(list(vocab_counts.values()))
    vocabulary = list(vocab_counts.keys())
    topvocab = np.argsort(counts)[::-1][:num_vocab]

    print(f"Coverage of top {num_vocab} vocabulary:")
    print(float(np.sum(counts[topvocab])) / np.sum(counts))

    vocab2idx = {vocabulary[vocab_idx]: idx for idx, vocab_idx in enumerate(topvocab)}
    idx2vocab = [vocabulary[vocab_idx] for vocab_idx in topvocab]

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


def labels_to_tensor(labels: List[List[str]], vocab2idx: Dict[str, int], max_seq_len: int):
    """
    Input:
        labels: Labels with strings
        output: Labels encoded in integers as a tensor
    """

    arrays = []
    for name_seq in labels:
        arrays.append(encode_name_seq_to_arr(name_seq, vocab2idx, max_seq_len))

    return torch.cat(arrays, dim=0)


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
        [[vocab2idx.get(w, vocab2idx["__UNK__"]) for w in augmented_seq]],
        dtype=torch.long,
    )


def decode_arr_to_name_seq(arr: torch.TensorType, idx2vocab: Dict[int, str]) -> List[int]:
    """
    Input: torch 1d array: y_arr
    Output: a sequence of words.
    """
    eos_idx_list = torch.nonzero(
        arr == len(idx2vocab) - 1, as_tuple=False
    )  # find the position of __EOS__ (the last vocab in idx2vocab)

    if len(eos_idx_list) > 0:
        clippted_arr = arr[: torch.min(eos_idx_list)]
    else:
        clippted_arr = arr

    return list(map(lambda x: idx2vocab[x], clippted_arr.cpu()))
