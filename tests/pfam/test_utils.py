from pfam.utils import reader, build_labels, build_vocab


class TestUtils:

    def test_build_labels(self, test_data_path):
        train_data, train_targets = reader("train", test_data_path + "/random_split")
        fam2label = build_labels(train_targets)
        assert len(fam2label) == 101

    def test_build_vocab(self, test_data_path):
        train_data, train_targets = reader("train", test_data_path + "/random_split")
        word2id = build_vocab(train_data, rare_AAs={"B", "O", "U", "X", "Z"})
        assert len(word2id) == 22
