import json
import os
import pandas as pd
from typing import Tuple, List


def load_json(filename: str) -> dict:
    """
    Load json dictionary from file
    :param filename:
    :return:
    """
    with open(filename, "r") as f:
        return json.load(f)


def save_json(filename: str, obj: dict) -> None:
    """
    Write json dictionary to file
    :param filename:
    :param obj:
    :return:
    """
    with open(filename, "w") as f:
        f.write(json.dumps(obj))


def reader(partition: str, data_path: str) -> Tuple[List[str], List[str]]:
    """

    :param partition: data split to use (train, dev or test)
    :param data_path: location of the random_split/ data folder
    :return: list of amino acid sequences and list of protein families
    """
    data = []
    for file_name in os.listdir(os.path.join(data_path, partition)):
        with open(os.path.join(data_path, partition, file_name)) as file:
            data.append(pd.read_csv(file, index_col=None, usecols=["sequence", "family_accession"]))

    all_data = pd.concat(data)

    return all_data["sequence"], all_data["family_accession"]


def build_labels(targets: List[str]) -> dict:
    """

    :param targets: list of protein families
    :return: fam2label, dictionary mapping (bijection) a protein family to a corresponding integer
    """
    unique_targets = targets.unique()
    fam2label = {target: i for i, target in enumerate(unique_targets, start=1)}
    fam2label['<unk>'] = 0

    return fam2label


def build_vocab(data: List[str], rare_AAs: set) -> set:
    """

    :param data: list of amino acid sequences
    :param rare_AAs: set of amino acid letters to treat as unknown <unk>
    :return: word2id, dictionary mapping (bijection) an amino-acid letter to a corresponding integer
    """
    # Build the vocabulary
    voc = set()
    for sequence in data:
        voc.update(sequence)

    unique_AAs = sorted(voc - rare_AAs)

    # Build the mapping
    word2id = {w: i for i, w in enumerate(unique_AAs, start=2)}
    word2id['<pad>'] = 0
    word2id['<unk>'] = 1

    return word2id
