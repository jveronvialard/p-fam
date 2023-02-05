import numpy as np
import torch

from pfam.utils import reader


def preprocess(text: str, max_len: int, word2id: dict) -> torch.tensor:
    """
    Convert a sequence of amino-acid letters to its one-hot torch tensor encoding
    :param text: a sequence of amino-acid letters
    :param max_len: used to truncate a sequence of amino-acid letters to its first max_len letters or pad to max_len
    :param word2id: dictionary mapping (bijection) an amino-acid letter to a corresponding integer
    :return: one-hot torch tensor encoding of the amino acid sequence
    """
    seq = []

    # Encode into IDs
    for word in text[:max_len]:
        seq.append(word2id.get(word, word2id['<unk>']))

    # Pad to maximal length
    if len(seq) < max_len:
        seq += [word2id['<pad>'] for _ in range(max_len - len(seq))]

    # Convert list into tensor
    seq = torch.from_numpy(np.array(seq))

    # One-hot encode
    one_hot_seq = torch.nn.functional.one_hot(seq.to(torch.int64), num_classes=len(word2id), )

    # Permute channel (one-hot) dim first
    one_hot_seq = one_hot_seq.permute(1, 0)

    return one_hot_seq


class SequenceDataset(torch.utils.data.Dataset):
    """
    Torch dataset to handle sequences of amino-acid letters and their corresponding label (a.k.a. protein family)
    """

    def __init__(self, word2id: dict, fam2label: dict, max_len: int, data_path: str, split: str) -> None:
        """

        :param word2id: dictionary mapping (bijection) an amino-acid letter to a corresponding integer
        :param fam2label: dictionary mapping (bijection) a protein family to a corresponding integer
        :param max_len: used to truncate a sequence of amino-acid letters to its first max_len letters or pad to max_len
        :param data_path: location of the random_split/ data folder
        :param split: data split to use (train, dev or test)
        """
        self.word2id = word2id
        self.fam2label = fam2label
        self.max_len = max_len
        self.data, self.label = reader(split, data_path)

    def __len__(self) -> int:
        """

        :return: dataset length
        """
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        """

        :param index: index of item
        :return: dictionary composed, for the given index, of the sequence of amino-acid letters and its label
        """
        text = self.data.iloc[index]
        seq = preprocess(text, self.max_len, self.word2id)
        label = self.fam2label.get(self.label.iloc[index], self.fam2label['<unk>'])
        return {'sequence': seq, 'target': label}
