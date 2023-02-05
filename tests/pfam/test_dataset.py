import numpy as np

from pfam.dataset import preprocess, SequenceDataset


class TestDataset:

    def test_preprocess(self):
        text = "LLQKKIRVRP"
        array = preprocess(text, max_len=4, word2id={"<pad>": 0, "<unk>": 1, "L": 2, "Q": 3, "D": 4}).numpy()
        assert (array == np.array([[0, 0, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])).all()
        assert list(array.shape) == [5, 4]
        text = "LLQKKIRVRP"
        array = preprocess(text, max_len=4, word2id={"<pad>": 0, "<unk>": 1, "P": 2, "Q": 3}).numpy()
        assert (array == np.array([[0, 0, 0, 0], [1, 1, 0, 1], [0, 0, 0, 0], [0, 0, 1, 0]])).all()
        assert list(array.shape) == [4, 4]

    def test_iter(self, word2id, fam2label, test_data_path):
        dataset = SequenceDataset(word2id, fam2label, max_len=120, data_path=test_data_path + "/random_split",
                                  split="train")
        assert list(next(iter(dataset))['sequence'].shape) == [22, 120]
        dataset = SequenceDataset(word2id, fam2label, max_len=160, data_path=test_data_path + "/random_split",
                                  split="train")
        assert list(next(iter(dataset))['sequence'].shape) == [22, 160]
        dataset = SequenceDataset({"<pad>": 0, "<unk>": 1, "L": 2, "Q": 3, "D": 4}, fam2label, max_len=160,
                                  data_path=test_data_path + "/random_split", split="train")
        assert list(next(iter(dataset))['sequence'].shape) == [5, 160]
