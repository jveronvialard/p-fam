import torch

from pfam.model import ProtCNN


class TestProtCNN:

    def test_forward(self):
        vocab_size = 4
        num_classes = 25
        seq_max_len = 80
        prot_cnn = ProtCNN(
            vocab_size=vocab_size, num_classes=num_classes, seq_max_len=seq_max_len,
            lr=1e-2, momentum=0.9, weight_decay=1e-2
        )
        batch_size = 7
        x = torch.zeros((batch_size, vocab_size, seq_max_len))
        y = prot_cnn(x)
        assert list(y.shape) == [batch_size, num_classes]
