import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from typing import Callable


class Lambda(torch.nn.Module):
    """
    Lambda Layer
    """

    def __init__(self, func: Callable) -> None:
        super().__init__()
        self.func = func

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.func(x)


class ResidualBlock(torch.nn.Module):
    """
    The residual block used by ProtCNN (https://www.biorxiv.org/content/10.1101/626507v3.full).
    """

    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1) -> None:
        """

        :param in_channels: number of channels (feature maps) of the incoming embedding
        :param out_channels: number of channels after the first convolution
        :param dilation: dilation rate of the first convolution
        """
        super().__init__()

        # Initialize the required layers
        self.skip = torch.nn.Sequential()

        self.bn1 = torch.nn.BatchNorm1d(in_channels)
        self.conv1 = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=3, bias=False, dilation=dilation, padding=dilation)
        self.bn2 = torch.nn.BatchNorm1d(out_channels)
        self.conv2 = torch.nn.Conv1d(in_channels=out_channels, out_channels=out_channels,
                                     kernel_size=3, bias=False, padding=1)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """

        :param x:
        :return:
        """
        # Execute the required layers and functions
        activation = F.relu(self.bn1(x))
        x1 = self.conv1(activation)
        x2 = self.conv2(F.relu(self.bn2(x1)))

        return x2 + self.skip(x)


class ProtCNN(pl.LightningModule):
    """
    Protein classifier.
    For each protein (sequence of amino acids), assign the corresponding Pfam family (i.e. protein family).
    """

    def __init__(self, vocab_size: int, num_classes: int, seq_max_len: int, lr: float, momentum: float,
                 weight_decay: float) -> None:
        """

        :param vocab_size: number of amino acids along with <pad> and <unk>
        :param num_classes: number of protein families
        :param seq_max_len: used to truncate a sequence of amino-acid letters to its first max_len letters or pad to max_len
        :param lr: optimizer learning rate
        :param momentum: optimizer momentum
        :param weight_decay: optimizer weight decay
        """
        super().__init__()
        self.save_hyperparameters()

        self.vocab_size, self.num_classes, self.seq_max_len = vocab_size, num_classes, seq_max_len
        self.lr, self.momentum, self.weight_decay = lr, momentum, weight_decay

        self.model = torch.nn.Sequential(
            torch.nn.Conv1d(vocab_size, 128, kernel_size=1, padding=0, bias=False),
            ResidualBlock(128, 128, dilation=2),
            ResidualBlock(128, 128, dilation=3),
            torch.nn.MaxPool1d(3, stride=2, padding=1),
            Lambda(lambda x: x.flatten(start_dim=1)),
            torch.nn.Linear(64 * seq_max_len, num_classes)
        )

        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def forward(self, x: torch.tensor) -> torch.tensor:
        """

        :param x:
        :return:
        """
        return self.model(x.float())

    def training_step(self, batch: int, batch_idx: int) -> torch.tensor:
        """
        Training loop. Monitoring train_loss and train_acc.
        :param batch:
        :param batch_idx:
        :return: training loss
        """
        x, y = batch['sequence'], batch['target']
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True)

        pred = torch.argmax(y_hat, dim=1)
        self.train_acc(pred, y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch: int, batch_idx: int) -> torch.tensor:
        """
        Validation loop. Monitoring valid_acc.
        :param batch:
        :param batch_idx:
        :return: validation accuracy
        """
        x, y = batch['sequence'], batch['target']
        y_hat = self(x)
        pred = torch.argmax(y_hat, dim=1)
        acc = self.valid_acc(pred, y)
        self.log('valid_acc', self.valid_acc, on_step=False, on_epoch=True)

        return acc

    def test_step(self, batch: int, batch_idx: int) -> torch.tensor:
        """
        Testing loop. Monitoring test_acc.
        :param batch:
        :param batch_idx:
        :return: test accuracy
        """
        x, y = batch['sequence'], batch['target']
        y_hat = self(x)
        pred = torch.argmax(y_hat, dim=1)
        acc = self.test_acc(pred, y)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)

        return acc

    def configure_optimizers(self) -> dict:
        """
        Configuring optimizer and learning rate scheduler.
        :return: dictionary containing the optimizer and learning rate scheduler configured
        """
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay
        )
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[5, 8, 10, 12, 14, 16, 18, 20], gamma=0.9
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }
